from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from .rewards import (
    format_reward,
    make_exact_match_reward,
    make_moleculariq_reward,
    make_moleculariq_shaped_reward,
    make_moleculariq_multitask_reward,
    soft_format_reward,
)
from .utils import load_tokenizer, set_seed


@dataclass
class TrainArgs:
    """Training configuration parsed from YAML.

    Args:
        model_name: Hugging Face model name or local checkpoint path.
        dataset_path: Path to a preprocessed dataset saved with `save_to_disk`.
        output_dir: Directory where checkpoints and final model files are written.
        task_name: Registered task name used to select rewards.
        learning_rate: Optimizer learning rate.
        beta: GRPO KL-to-reference coefficient.
        num_generations: Number of completions sampled per prompt.
        temperature: Sampling temperature for GRPO generation (higher = more
            exploration, useful for generative tasks prone to mode collapse).
        per_device_train_batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        max_prompt_length: Maximum prompt token length.
        max_completion_length: Maximum completion token length.
        num_train_epochs: Number of training epochs.
        max_steps: Optional step cap; negative values disable the cap.
        warmup_ratio: Learning-rate warmup ratio.
        weight_decay: Optimizer weight decay.
        max_grad_norm: Gradient clipping norm.
        lr_scheduler_type: Learning-rate scheduler name.
        seed: Random seed.
        logging_steps: Trainer logging interval.
        save_steps: Trainer checkpoint interval.
        bf16: Whether to load/train with bfloat16.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        optim: Trainer optimizer name.
        use_soft_format_reward: Whether to include loose format partial credit.
        correctness_weight: Reward weight for correctness.
        use_shaped_moleculariq_reward: Whether to add MolecularIQ partial credit.
        shaped_moleculariq_weight: Maximum reward for shaped MolecularIQ partial credit.
        smiles_validity_weight: Extra reward for valid generated SMILES.
        resume_from_checkpoint: Optional checkpoint path, or "latest".
        save_on_interrupt: Whether Ctrl+C should save an interrupt checkpoint.
        moleculariq_task_type: MolecularIQ task type used by the chemistry reward.
        grpo_overrides: Extra keyword arguments merged into `GRPOConfig`.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/miq_multitask_pooled_train"
    output_dir: str = "outputs/miq-multitask-pooled-grpo"
    task_name: str = "moleculariq"

    learning_rate: float = 1e-5
    beta: float = 0.005
    num_generations: int = 4
    temperature: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 512
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    lr_scheduler_type: str = "cosine"

    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    use_soft_format_reward: bool = False
    correctness_weight: float = 2.0
    use_shaped_moleculariq_reward: bool = True
    shaped_moleculariq_weight: float = 1.0
    smiles_validity_weight: float = 0.5
    resume_from_checkpoint: str | None = None
    save_on_interrupt: bool = True

    moleculariq_task_type: str = "single_count"

    grpo_overrides: dict[str, Any] = field(default_factory=dict)


def _build_training_args(cfg: TrainArgs) -> GRPOConfig:
    """Build TRL GRPO configuration from project training args.

    Args:
        cfg: Project-level training configuration.

    Returns:
        Initialized `GRPOConfig` for `GRPOTrainer`.
    """
    base = dict(
        output_dir=cfg.output_dir,
        run_name=Path(cfg.output_dir).name,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        optim=cfg.optim,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        max_grad_norm=cfg.max_grad_norm,
        report_to="none",
        log_on_each_node=False,
        use_vllm=False,  # not supported on Windows
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.seed,
    )
    base.update(cfg.grpo_overrides or {})
    return GRPOConfig(**base)


def _latest_checkpoint(output_dir: str | Path) -> str | None:
    """Return the numerically latest checkpoint under an output directory."""
    output_dir = Path(output_dir)
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
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def _resolve_resume_checkpoint(cfg: TrainArgs) -> str | None:
    """Resolve a configured checkpoint path or the latest checkpoint alias."""
    value = cfg.resume_from_checkpoint
    if value is None or value is False:
        return None
    if value is True or str(value).lower() == "latest":
        latest = _latest_checkpoint(cfg.output_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint-* directory found under {cfg.output_dir}."
            )
        return latest
    return str(value)


def _save_interrupt_checkpoint(trainer: GRPOTrainer) -> str:
    """Best-effort save of a resumable checkpoint after Ctrl+C."""
    step = trainer.state.global_step
    checkpoint_dir = Path(trainer.args.output_dir) / f"checkpoint-{step}"

    # Prefer Trainer's real checkpoint path because it preserves optimizer,
    # scheduler, RNG, and trainer state for resume.
    try:
        save_checkpoint = getattr(trainer, "_save_checkpoint")
        signature = inspect.signature(save_checkpoint)
        if "metrics" in signature.parameters:
            save_checkpoint(trainer.model, trial=None, metrics=None)
        else:
            save_checkpoint(trainer.model, trial=None)
    except Exception as exc:
        print(
            "[train] warning: full checkpoint save failed after interrupt; "
            f"falling back to model/state save only ({exc})"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(checkpoint_dir))

    trainer.save_state()
    return str(checkpoint_dir)


def train(cfg: TrainArgs) -> str:
    """Run GRPO training and save the resulting model.

    Args:
        cfg: Training configuration.

    Returns:
        Output directory containing the saved model and tokenizer.
    """
    set_seed(cfg.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_dataset = load_from_disk(cfg.dataset_path)

    print(f"[train] model     = {cfg.model_name}")
    print(f"[train] task      = {cfg.task_name}")
    print(f"[train] dataset   = {cfg.dataset_path} (n={len(train_dataset)})")
    print(f"[train] output    = {cfg.output_dir}")

    tokenizer = load_tokenizer(
        cfg.model_name,
        model_max_length=cfg.max_prompt_length + cfg.max_completion_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )

    training_args = _build_training_args(cfg)

    is_multitask_moleculariq = (
        cfg.task_name == "moleculariq" and "task_type" in train_dataset.column_names
    )
    shaped_reward = None
    if is_multitask_moleculariq:
        correctness = make_moleculariq_multitask_reward(weight=cfg.correctness_weight)
        if cfg.use_shaped_moleculariq_reward:
            shaped_reward = make_moleculariq_shaped_reward(
                weight=cfg.shaped_moleculariq_weight,
                smiles_validity_weight=cfg.smiles_validity_weight,
            )
        print("[train] reward   = moleculariq multitask dispatch")
    elif cfg.task_name == "moleculariq":
        correctness = make_moleculariq_reward(
            task_type=cfg.moleculariq_task_type, weight=cfg.correctness_weight
        )
        if cfg.use_shaped_moleculariq_reward:
            shaped_reward = make_moleculariq_shaped_reward(
                task_type=cfg.moleculariq_task_type,
                weight=cfg.shaped_moleculariq_weight,
                smiles_validity_weight=cfg.smiles_validity_weight,
            )
    else:
        correctness = make_exact_match_reward(cfg.correctness_weight)
    reward_funcs = [format_reward, correctness]
    if cfg.use_soft_format_reward:
        reward_funcs.insert(0, soft_format_reward)
    if shaped_reward is not None:
        reward_funcs.insert(-1, shaped_reward)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    resume_from_checkpoint = _resolve_resume_checkpoint(cfg)
    if resume_from_checkpoint:
        print(f"[train] resume   = {resume_from_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n[train] interrupted by user.")
        if not cfg.save_on_interrupt:
            print("[train] save_on_interrupt=false; exiting without extra checkpoint.")
            raise
        checkpoint = _save_interrupt_checkpoint(trainer)
        print(f"[train] interrupt checkpoint saved to {checkpoint}")
        print(
            "[train] resume later with: "
            f"python scripts/train.py --config <CONFIG> --resume-from-checkpoint {checkpoint}"
        )
        raise SystemExit(130)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train] done. Model saved to {cfg.output_dir}")
    return cfg.output_dir
