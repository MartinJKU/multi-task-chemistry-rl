"""Plot training curves (loss / reward / KL) from a trainer_state.json.

Usage:
    python scripts/single_task/plot_training.py --output-dir outputs/miq-sc_ring_count-grpo
"""
from __future__ import annotations

from grpo_reasoning.single_task.cli import plot_training_main


if __name__ == "__main__":
    plot_training_main()
