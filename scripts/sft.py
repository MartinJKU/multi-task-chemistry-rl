"""Run supervised warm-start fine-tuning from a YAML config.

Usage:
    python scripts/sft.py --config configs/multitask/miq_sft_index_warmstart.yaml
"""
from __future__ import annotations

from grpo_reasoning.common.cli import sft_main


if __name__ == "__main__":
    sft_main()
