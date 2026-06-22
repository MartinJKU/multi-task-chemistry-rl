from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SFTArgs:
    """Configuration for supervised warm-start fine-tuning.

    The SFT trainer consumes the same cached datasets used by GRPO. It builds a
    deterministic assistant completion from each row's gold answer and masks the
    prompt tokens, so loss is applied only to the target completion.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/miq_curriculum_02_simple_index_train"
    output_dir: str = "outputs/miq-sft-index-warmstart"

    task_type_filter: list[str] = field(
        default_factory=lambda: ["single_index", "multi_index"]
    )
    task_id_filter: list[str] = field(default_factory=list)
    num_samples: int | None = None
    shuffle: bool = True

    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_prompt_length: int = 384
    max_completion_length: int = 384
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    resume_from_checkpoint: str | None = None
    save_on_interrupt: bool = True

    training_overrides: dict[str, Any] = field(default_factory=dict)


def _json_dumps(value: Any) -> str:
    """Serialize compact JSON deterministically for answer blocks."""
    return json.dumps(value, sort_keys=True, separators=(",", ": "))


def _parse_answer(answer: str) -> Any | None:
    """Parse a JSON answer string, returning None when malformed."""
    try:
        return json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        return None


def _format_list(values: list[Any]) -> str:
    """Return a readable comma-separated list or an empty-list marker."""
    if not values:
        return "empty"
    return ", ".join(str(value) for value in values)


def _index_reasoning(parsed: dict[str, Any], smiles: str | None = None) -> str:
    """Build a compact oracle rationale for index targets."""
    parts = [
        "Use 0-based heavy-atom indices in SMILES order and return no extra atoms."
    ]
    if smiles:
        parts.append(f"The molecule is {smiles}.")
    for key, values in parsed.items():
        values = values if isinstance(values, list) else []
        if values:
            parts.append(f"{key} contains {len(values)} atom(s): {_format_list(values)}.")
        else:
            parts.append(f"No atoms match {key}, so {key} is an empty list.")
    return " ".join(parts)


def _count_reasoning(parsed: dict[str, Any], smiles: str | None = None) -> str:
    """Build a compact oracle rationale for count targets."""
    parts = ["Read the molecule and report the exact requested integer count(s)."]
    if smiles:
        parts.append(f"The molecule is {smiles}.")
    for key, value in parsed.items():
        parts.append(f"{key} = {value}.")
    return " ".join(parts)


def _completion_from_row(row: dict[str, Any]) -> str | None:
    """Create a supervised assistant completion for one cached dataset row."""
    parsed = _parse_answer(row.get("answer", ""))
    task_type = row.get("task_type")
    if task_type in {"single_index", "multi_index"} and isinstance(parsed, dict):
        reasoning = _index_reasoning(parsed, row.get("smiles"))
        answer = _json_dumps(parsed)
    elif task_type in {"single_count", "multi_count"} and isinstance(parsed, dict):
        reasoning = _count_reasoning(parsed, row.get("smiles"))
        answer = _json_dumps(parsed)
    else:
        return None
    return f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"


def _resolve_resume_checkpoint(cfg: SFTArgs) -> str | None:
    """Resolve latest-checkpoint aliases for SFT resume."""
    value = cfg.resume_from_checkpoint
    if value is None or value is False:
        return None
    if value is True or str(value).lower() == "latest":
        output_dir = Path(cfg.output_dir)
        checkpoints = []
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            checkpoints.append((step, path))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint-* directory found under {cfg.output_dir}."
            )
        return str(max(checkpoints, key=lambda item: item[0])[1])
    return str(value)


def _select_rows(dataset, cfg: SFTArgs):
    """Filter and optionally cap the cached dataset for SFT."""
    selected = dataset
    if cfg.task_type_filter:
        allowed = set(cfg.task_type_filter)
        selected = selected.filter(lambda row: row.get("task_type") in allowed)
    if cfg.task_id_filter:
        allowed_ids = set(cfg.task_id_filter)
        selected = selected.filter(lambda row: row.get("task_id") in allowed_ids)
    if cfg.shuffle:
        selected = selected.shuffle(seed=cfg.seed)
    if cfg.num_samples is not None:
        selected = selected.select(range(min(int(cfg.num_samples), len(selected))))
    return selected


class _CausalLMCollator:
    """Right-pad causal-LM SFT examples while masking prompt labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_train(cfg: SFTArgs) -> str:
    """Run supervised fine-tuning and save the warm-start checkpoint."""
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from .utils import load_tokenizer, set_seed

    import torch

    set_seed(cfg.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    raw_dataset = load_from_disk(cfg.dataset_path)
    dataset = _select_rows(raw_dataset, cfg)
    if len(dataset) == 0:
        raise ValueError(
            "SFT dataset is empty after filters. Check task_type_filter/task_id_filter."
        )

    print(f"[sft] model   = {cfg.model_name}")
    print(f"[sft] dataset = {cfg.dataset_path} (selected={len(dataset)})")
    print(f"[sft] output  = {cfg.output_dir}")

    tokenizer = load_tokenizer(
        cfg.model_name,
        model_max_length=cfg.max_prompt_length + cfg.max_completion_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        completion = _completion_from_row(row)
        if completion is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )["input_ids"]
        completion_text = completion + (tokenizer.eos_token or "")
        completion_ids = tokenizer(
            completion_text,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_completion_length,
        )["input_ids"]
        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 0)
    if len(tokenized) == 0:
        raise ValueError("No SFT rows could be converted to supervised completions.")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    args = dict(
        output_dir=cfg.output_dir,
        run_name=Path(cfg.output_dir).name,
        learning_rate=cfg.learning_rate,
        optim=cfg.optim,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=cfg.bf16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.seed,
    )
    args.update(cfg.training_overrides or {})

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**args),
        "train_dataset": tokenized,
        "data_collator": _CausalLMCollator(tokenizer.pad_token_id),
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    resume_from_checkpoint = _resolve_resume_checkpoint(cfg)
    if resume_from_checkpoint:
        print(f"[sft] resume  = {resume_from_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n[sft] interrupted by user.")
        if not cfg.save_on_interrupt:
            raise
        checkpoint_dir = Path(cfg.output_dir) / f"checkpoint-{trainer.state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(checkpoint_dir))
        trainer.save_state()
        print(f"[sft] interrupt checkpoint saved to {checkpoint_dir}")
        raise SystemExit(130)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[sft] done. Model saved to {cfg.output_dir}")
    return cfg.output_dir
