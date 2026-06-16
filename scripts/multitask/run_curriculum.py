"""Run staged MolecularIQ curriculum training.

Usage:
    python scripts/multitask/run_curriculum.py --config configs/multitask/miq_curriculum.yaml
    python scripts/multitask/run_curriculum.py --config configs/multitask/miq_curriculum.yaml --dataset-only
    python scripts/multitask/run_curriculum.py --config configs/multitask/miq_curriculum.yaml --max-steps-per-stage 50
"""
from __future__ import annotations

from grpo_reasoning.multitask.cli import curriculum_main


if __name__ == "__main__":
    curriculum_main()
