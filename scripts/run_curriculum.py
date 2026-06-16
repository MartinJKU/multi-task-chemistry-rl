"""Run staged MolecularIQ curriculum training.

Usage:
    python scripts/run_curriculum.py --config configs/miq_curriculum.yaml
    python scripts/run_curriculum.py --config configs/miq_curriculum.yaml --dataset-only
    python scripts/run_curriculum.py --config configs/miq_curriculum.yaml --max-steps-per-stage 50
"""
from __future__ import annotations

from grpo_reasoning.cli import curriculum_main


if __name__ == "__main__":
    curriculum_main()
