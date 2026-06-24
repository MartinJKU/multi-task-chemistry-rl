from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from .prompts import extract_xml_answer
from .rewards import moleculariq_diagnostics
from .tasks import get_task


def evaluate(
    model_path: str,
    task_name: str,
    num_samples: int | None = 200,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    save_path: str | Path | None = None,
    task_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run greedy evaluation on a task test split.

    Args:
        model_path: Model name or checkpoint path to evaluate.
        task_name: Registered task name to evaluate on.
        num_samples: Optional maximum number of test samples to use.
        batch_size: Number of prompts to generate per batch.
        max_new_tokens: Maximum number of completion tokens to generate.
        save_path: Optional JSON output path for metrics and per-sample results.
        task_kwargs: Optional task-specific keyword arguments.

    Returns:
        Metrics dictionary containing accuracy, counts, model path, task, and timestamp.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from .utils import load_tokenizer

    task = get_task(task_name, **(task_kwargs or {}))
    ds: Dataset = task.to_grpo_dataset(split="test", num_samples=num_samples)

    print(f"[eval] model   = {model_path}")
    print(f"[eval] task    = {task_name}")
    print(f"[eval] samples = {len(ds)}")

    tokenizer = load_tokenizer(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    results: list[dict] = []
    correct = 0
    diagnostic_rows: list[dict[str, float | bool | str]] = []
    eval_task_type = getattr(task, "task_type", None)

    for start in tqdm(range(0, len(ds), batch_size), desc="eval"):
        batch = ds[start : start + batch_size]
        prompts = batch["prompt"]
        gold = batch["answer"]
        questions = batch["question"]

        rendered = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_tokens = out[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for q, g, resp in zip(questions, gold, decoded):
            extracted = extract_xml_answer(resp)
            is_correct = bool(task.score_prediction(resp, g))
            correct += int(is_correct)
            row = {
                "question": q,
                "gold": g,
                "extracted": extracted,
                "response": resp,
                "correct": is_correct,
            }
            if task_name == "moleculariq" and eval_task_type is not None:
                diagnostics = moleculariq_diagnostics(resp, g, eval_task_type)
                row["diagnostics"] = diagnostics
                diagnostic_rows.append(diagnostics)
            results.append(row)

    total = len(results)
    answer_counts = Counter(row["extracted"] for row in results if row["extracted"])
    distinct_answers = len(answer_counts)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        # Fraction of unique extracted answers. A value near 0 means the model
        # collapsed to a (near) molecule-independent output -- the tell-tale sign
        # of reward-hacking on set-valued tasks rather than real per-molecule
        # reasoning.
        "distinct_answer_rate": distinct_answers / total if total else 0.0,
        "most_common_answer_rate": (
            max(answer_counts.values()) / total if total and answer_counts else 0.0
        ),
        "model_path": str(model_path),
        "task": task_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if diagnostic_rows:
        metrics.update(
            {
                "partial_score_mean": sum(
                    float(row["partial_score"]) for row in diagnostic_rows
                )
                / len(diagnostic_rows),
                "answer_present_rate": sum(
                    1 for row in diagnostic_rows if row["answer_present"]
                )
                / len(diagnostic_rows),
                "json_valid_rate": sum(1 for row in diagnostic_rows if row["json_valid"])
                / len(diagnostic_rows),
                "valid_smiles_rate": sum(
                    1 for row in diagnostic_rows if row["valid_smiles"]
                )
                / len(diagnostic_rows),
            }
        )
        _add_index_metrics(metrics, diagnostic_rows)
        _add_constraint_metrics(metrics, diagnostic_rows)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"[eval] wrote {save_path}")

    print(f"[eval] accuracy = {metrics['accuracy']:.2%} ({correct}/{total})")
    return metrics


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """Average numeric/boolean diagnostic values for rows that contain key."""
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _add_index_metrics(
    metrics: dict[str, Any],
    diagnostic_rows: list[dict[str, float | bool | str]],
) -> None:
    """Add precision/recall and over-prediction metrics for index tasks."""
    index_rows = [row for row in diagnostic_rows if "index_empty_gold" in row]
    empty_gold_rows = [
        row for row in index_rows if bool(row.get("index_empty_gold", False))
    ]
    nonempty_gold_rows = [
        row for row in index_rows if not bool(row.get("index_empty_gold", False))
    ]
    if index_rows:
        metrics["index_gold_empty_rate"] = len(empty_gold_rows) / len(index_rows)
        metrics["index_empty_gold_total"] = len(empty_gold_rows)
        metrics["index_nonempty_total"] = len(nonempty_gold_rows)
    if empty_gold_rows:
        metrics["index_empty_gold_accuracy"] = _mean(empty_gold_rows, "exact_match")
    if nonempty_gold_rows:
        metrics["index_nonempty_accuracy"] = _mean(nonempty_gold_rows, "exact_match")
        metrics["index_nonempty_partial_score_mean"] = _mean(
            nonempty_gold_rows,
            "partial_score",
        )
        metrics["index_nonempty_precision_mean"] = _mean(
            nonempty_gold_rows,
            "index_precision",
        )
        metrics["index_nonempty_recall_mean"] = _mean(
            nonempty_gold_rows,
            "index_recall",
        )

    metric_map = {
        "index_precision": "index_precision_mean",
        "index_recall": "index_recall_mean",
        "index_pred_len": "avg_index_pred_len",
        "index_gold_len": "avg_index_gold_len",
        "index_false_positives": "avg_index_false_positives",
        "index_false_negatives": "avg_index_false_negatives",
        "index_empty_gold_nonempty_pred": "empty_gold_nonempty_rate",
        "index_superset": "superset_rate",
        "index_subset": "subset_rate",
    }
    for source, target in metric_map.items():
        value = _mean(diagnostic_rows, source)
        if value is not None:
            metrics[target] = value


def _add_constraint_metrics(
    metrics: dict[str, Any],
    diagnostic_rows: list[dict[str, float | bool | str]],
) -> None:
    """Add constraint satisfaction and SMILES expressivity metrics."""
    for source, target in (
        ("constraint_satisfied_fraction", "constraint_satisfied_fraction_mean"),
        ("trivial_alkane", "trivial_alkane_rate"),
    ):
        value = _mean(diagnostic_rows, source)
        if value is not None:
            metrics[target] = value

    ring_requested = [
        row for row in diagnostic_rows if bool(row.get("ring_requested", False))
    ]
    if ring_requested:
        metrics["ringless_when_ring_requested_rate"] = sum(
            1 for row in ring_requested if row.get("ringless_when_ring_requested")
        ) / len(ring_requested)

    canonical = [
        str(row["canonical_smiles"])
        for row in diagnostic_rows
        if row.get("canonical_smiles")
    ]
    if canonical:
        metrics["canonical_smiles_distinct_rate"] = len(set(canonical)) / len(
            canonical
        )
        metrics["canonical_smiles_most_common_rate"] = max(
            Counter(canonical).values()
        ) / len(canonical)

    successful_canonical = [
        str(row["canonical_smiles"])
        for row in diagnostic_rows
        if row.get("canonical_smiles") and float(row.get("exact_match", 0.0)) >= 1.0
    ]
    if successful_canonical:
        metrics["successful_canonical_smiles_distinct_rate"] = len(
            set(successful_canonical)
        ) / len(successful_canonical)
        metrics["successful_canonical_smiles_most_common_rate"] = max(
            Counter(successful_canonical).values()
        ) / len(successful_canonical)

    target_groups: dict[str, list[float]] = defaultdict(list)
    for row in diagnostic_rows:
        signature = row.get("constraint_target_signature")
        if not signature:
            continue
        target_groups[str(signature)].append(float(row.get("exact_match", 0.0)))
    if target_groups:
        target_breakdown = {
            signature: {
                "accuracy": sum(scores) / len(scores),
                "correct": int(sum(scores)),
                "total": len(scores),
            }
            for signature, scores in sorted(target_groups.items())
        }
        target_accuracies = [
            float(item["accuracy"]) for item in target_breakdown.values()
        ]
        metrics["constraint_target_count"] = len(target_breakdown)
        metrics["constraint_target_macro_accuracy"] = sum(target_accuracies) / len(
            target_accuracies
        )
        metrics["worst_constraint_target_accuracy"] = min(target_accuracies)
        metrics["constraint_target_breakdown"] = target_breakdown
