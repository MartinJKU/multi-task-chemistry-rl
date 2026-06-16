"""Audit MolecularIQ source-pool coverage for the configured task suite.

Usage:
    python scripts/multitask/inspect_moleculariq_dataset.py --config configs/multitask/miq_experiment_suite.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grpo_reasoning.multitask.cli import audit_moleculariq_main


if __name__ == "__main__":
    audit_moleculariq_main()
