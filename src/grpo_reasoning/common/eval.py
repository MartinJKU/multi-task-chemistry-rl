from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

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
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
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
