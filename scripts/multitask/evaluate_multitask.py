"""Evaluate a model on every MolecularIQ subtask in a multitask config.

Usage:
    python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_pooled.yaml \
        --model outputs/miq-multitask-pooled-grpo --model-label pooled
"""
from __future__ import annotations

from grpo_reasoning.multitask.cli import evaluate_multitask_main


if __name__ == "__main__":
    evaluate_multitask_main()
