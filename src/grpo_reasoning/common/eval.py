from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .passk import bootstrap_passk_ci as _bootstrap_passk_ci
from .passk import pass_at_k as _pass_at_k
from .prompts import extract_xml_answer
from .rewards import moleculariq_diagnostics
from .tasks import get_task
from .utils import load_tokenizer


@torch.no_grad()
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
    task = get_task(task_name, **(task_kwargs or {}))
    ds: Dataset = task.to_grpo_dataset(split="test", num_samples=num_samples)

    print(f"[eval] model   = {model_path}")
    print(f"[eval] task    = {task_name}")
    print(f"[eval] samples = {len(ds)}")

    tokenizer = load_tokenizer(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    results: list[dict] = []
    correct = 0
    diagnostic_rows: list[dict[str, float | bool]] = []
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
    distinct_answers = len({row["extracted"] for row in results if row["extracted"]})
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        # Fraction of unique extracted answers. A value near 0 means the model
        # collapsed to a (near) molecule-independent output -- the tell-tale sign
        # of reward-hacking on set-valued tasks rather than real per-molecule
        # reasoning.
        "distinct_answer_rate": distinct_answers / total if total else 0.0,
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

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"[eval] wrote {save_path}")

    print(f"[eval] accuracy = {metrics['accuracy']:.2%} ({correct}/{total})")
    return metrics


# ---------------------------------------------------------------------------
# pass@k evaluation
#
# pass@1 (greedy `evaluate` above) measures how often the single most-likely
# answer is correct. It cannot tell whether RL *expanded* a model's reachable
# solution set or merely *sharpened* the base model's existing distribution. To
# distinguish elicitation from expansion (Yue et al., 2025), sample many
# completions per item, score each with the official MolecularIQ verifier, and
# estimate pass@k for a range of k. Run this on both training task types and
# held-out task types: if a fine-tuned model's pass@k stays above the base model
# even at large k on a held-out task, transfer reflects genuine new capability;
# if the base model catches up, the apparent transfer was only elicitation.
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pass_at_k(
    model_path: str,
    task_name: str = "moleculariq",
    num_samples: int | None = 100,
    n_completions: int = 256,
    k_values: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    prompt_batch_size: int = 8,
    samples_per_round: int = 16,
    max_new_tokens: int = 512,
    save_path: str | Path | None = None,
    task_kwargs: dict | None = None,
    seed: int = 0,
    model_label: str | None = None,
) -> dict[str, Any]:
    """Estimate pass@k on a task test split by sampling many completions.

    For each test item, ``n_completions`` completions are sampled (in rounds of
    ``samples_per_round`` to bound memory), scored with the task's official
    verifier, and reduced to a correct-count ``c``. Corpus pass@k is the mean of
    the per-item unbiased estimator. Use the same ``task_kwargs`` (task_type +
    properties + filters) the model was, or was not, trained on so a held-out
    task is scored on the same tractable distribution as training.

    Args:
        model_path: Model name or checkpoint path to evaluate.
        task_name: Registered task name to evaluate on.
        num_samples: Optional maximum number of test items.
        n_completions: Number of completions to sample per item.
        k_values: Values of k to report (each must be <= n_completions).
        temperature: Sampling temperature (must be > 0 for pass@k).
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling cutoff.
        prompt_batch_size: Number of distinct prompts per generation batch.
        samples_per_round: Completions per prompt generated per round.
        max_new_tokens: Maximum number of completion tokens to generate.
        save_path: Optional JSON output path for metrics and per-item counts.
        task_kwargs: Optional task-specific keyword arguments.
        seed: RNG seed for sampling.
        model_label: Optional human-readable label stored in the summary.

    Returns:
        Metrics dictionary with per-k pass@k, bootstrap CIs, and per-item counts.
    """
    if temperature <= 0:
        raise ValueError("pass@k requires temperature > 0 for diverse sampling.")
    valid_ks = sorted({int(k) for k in k_values if 1 <= int(k) <= n_completions})
    if not valid_ks:
        raise ValueError(
            f"No k in {tuple(k_values)} satisfies 1 <= k <= n_completions={n_completions}."
        )

    task = get_task(task_name, **(task_kwargs or {}))
    ds: Dataset = task.to_grpo_dataset(split="test", num_samples=num_samples)
    eval_task_type = getattr(task, "task_type", None)

    print(f"[passk] model       = {model_path}")
    print(f"[passk] task        = {task_name} ({eval_task_type})")
    print(f"[passk] items       = {len(ds)}")
    print(f"[passk] completions = {n_completions} @ T={temperature}")

    torch.manual_seed(seed)
    tokenizer = load_tokenizer(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    correct_counts = [0] * len(ds)

    for start in tqdm(range(0, len(ds), prompt_batch_size), desc="passk"):
        batch = ds[start : start + prompt_batch_size]
        prompts = batch["prompt"]
        gold = batch["answer"]
        n_prompts = len(prompts)

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
        prompt_len = inputs["input_ids"].shape[1]

        remaining = n_completions
        while remaining > 0:
            rounds = min(samples_per_round, remaining)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=rounds,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_tokens = out[:, prompt_len:]
            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            # generate() returns prompt-major order: [p0_s0..p0_s(r-1), p1_s0..].
            for bi in range(n_prompts):
                for si in range(rounds):
                    resp = decoded[bi * rounds + si]
                    if task.score_prediction(resp, gold[bi]):
                        correct_counts[start + bi] += 1
            remaining -= rounds

    pass_at_k = {}
    for k in valid_ks:
        per_item = [_pass_at_k(n_completions, c, k) for c in correct_counts]
        mean = float(np.mean(per_item)) if per_item else 0.0
        low, high = _bootstrap_passk_ci(correct_counts, n_completions, k, seed=seed)
        pass_at_k[str(k)] = {"mean": mean, "ci_low": low, "ci_high": high}

    metrics = {
        "model_path": str(model_path),
        "model_label": model_label or Path(str(model_path)).name,
        "task": task_name,
        "task_type": eval_task_type,
        "properties": list(getattr(task, "properties", []) or []),
        "num_items": len(ds),
        "n_completions": n_completions,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "k_values": valid_ks,
        "pass_at_k": pass_at_k,
        "correct_counts": correct_counts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[passk] wrote {save_path}")

    summary = "  ".join(
        f"pass@{k}={pass_at_k[str(k)]['mean']:.3f}" for k in valid_ks
    )
    print(f"[passk] {summary}")
    return metrics
