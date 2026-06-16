from __future__ import annotations

import json

from grpo_reasoning.multitask.dataset import (
    MolecularIQTaskSpec,
    MultitaskDatasetConfig,
    _largest_remainder_counts,
    _read_task_accuracies,
)


def test_task_spec_parses_properties_as_list():
    """Verify YAML-style task specs are normalized into typed dataclasses."""
    spec = MolecularIQTaskSpec.from_dict(
        {
            "task_id": "sc_ring_count",
            "task_type": "single_count",
            "properties": ["ring_count"],
        }
    )
    assert spec.task_id == "sc_ring_count"
    assert spec.task_kwargs()["properties"] == ["ring_count"]


def test_multitask_config_requires_tasks():
    """Verify empty multitask configs are rejected early."""
    try:
        MultitaskDatasetConfig.from_dict({"out_dir": "data/x", "tasks": []})
    except ValueError as exc:
        assert "at least one task" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty multitask config.")


def test_largest_remainder_counts_keeps_total_and_positive_counts():
    """Verify proportional sample allocation is exact and non-empty per task."""
    counts = _largest_remainder_counts(
        ["easy", "hard", "medium"],
        {"easy": 1.0, "hard": 3.0, "medium": 2.0},
        total=12,
    )
    assert sum(counts.values()) == 12
    assert counts["hard"] >= counts["medium"] >= counts["easy"]
    assert min(counts.values()) > 0


def test_read_task_accuracies_from_multitask_summary(tmp_path):
    """Verify adaptive sampling can read evaluate_multitask summaries."""
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "sc_ring_count", "accuracy": 0.8},
                    {"task_id": "si_ring", "accuracy": 0.25},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _read_task_accuracies(path, "accuracy") == {
        "sc_ring_count": 0.8,
        "si_ring": 0.25,
    }
