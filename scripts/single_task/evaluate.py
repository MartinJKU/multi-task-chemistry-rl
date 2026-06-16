"""Evaluate baseline vs trained specialist model and write a comparison figure.

Usage:
    # Eval just one model:
    python scripts/single_task/evaluate.py --task moleculariq --task-type single_count \
        --properties ring_count --model Qwen/Qwen2.5-0.5B-Instruct --num-samples 200

    # Compare baseline vs trained checkpoint:
    python scripts/single_task/evaluate.py --task moleculariq --task-type single_count \
        --properties ring_count \
        --baseline Qwen/Qwen2.5-0.5B-Instruct \
        --trained outputs/miq-sc_ring_count-grpo \
        --num-samples 200 \
        --figure outputs/miq-sc_ring_count-grpo/figures/baseline_vs_trained.png
"""
from __future__ import annotations

from grpo_reasoning.single_task.cli import evaluate_main


if __name__ == "__main__":
    evaluate_main()
