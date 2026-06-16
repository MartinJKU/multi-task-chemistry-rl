"""Run GRPO training from a YAML config.

Shared by both pipelines: the trainer auto-detects single-task vs multitask
datasets from their columns.

Usage:
    python scripts/train.py --config configs/single_task/miq_sc_ring_count.yaml
    python scripts/train.py --config configs/multitask/miq_multitask_pooled_train.yaml --max-steps 50
"""
from __future__ import annotations

from grpo_reasoning.common.cli import train_main


if __name__ == "__main__":
    train_main()
