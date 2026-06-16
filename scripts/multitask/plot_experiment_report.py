"""Create comparison plots from all discovered train/eval outputs.

Usage:
    python scripts/multitask/plot_experiment_report.py --outputs-dir outputs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grpo_reasoning.multitask.cli import report_main


if __name__ == "__main__":
    report_main()
