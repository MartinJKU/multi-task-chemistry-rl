from __future__ import annotations

import json

from grpo_reasoning.multitask.dataset import (
    MolecularIQTaskSpec,
    MultitaskDatasetConfig,
    _largest_remainder_counts,
    _read_task_accuracies,
    _dataset_config_fingerprint,
    validate_saved_multitask_dataset,
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


def test_task_kwargs_threads_filters_to_task():
    """Verify difficulty filters reach the underlying task via task_kwargs.

    This is what keeps training and evaluation on the same molecule
    distribution: the eval path builds the task from task_kwargs, so filters
    must propagate.
    """
    spec = MolecularIQTaskSpec.from_dict(
        {
            "task_id": "si_ring",
            "task_type": "single_index",
            "properties": ["ring_index"],
            "candidate_multiplier": 6,
            "filters": {"max_smiles_length": 90, "max_answer_list_length": 10},
        }
    )
    kwargs = spec.task_kwargs(default_seed=42)
    assert kwargs["filters"] == {"max_smiles_length": 90, "max_answer_list_length": 10}
    assert kwargs["candidate_multiplier"] == 6


def test_task_kwargs_omits_filters_when_absent():
    """Verify count tasks without filters do not inject filter kwargs."""
    spec = MolecularIQTaskSpec.from_dict(
        {
            "task_id": "sc_ring_count",
            "task_type": "single_count",
            "properties": ["ring_count"],
        }
    )
    kwargs = spec.task_kwargs()
    assert "filters" not in kwargs
    assert "candidate_multiplier" not in kwargs


def test_task_applies_answer_list_length_filter():
    """Verify the task filters generated index examples by answer-list length.

    Uses a synthetic in-memory SMILES pool so no network/HF download is needed.
    """
    from datasets import Dataset

    from grpo_reasoning.common.tasks.moleculariq import MolecularIQTask

    smiles = [
        "c1ccccc1",
        "CCO",
        "C1CCCCC1",
        "c1ccc2ccccc2c1",
        "CC(=O)O",
        "c1ccncc1",
        "C1CCC2CCCCC2C1",
        "c1ccc(cc1)c1ccccc1",
    ] * 6

    class _PooledTask(MolecularIQTask):
        def load_raw(self, split):
            return Dataset.from_dict({"smiles": smiles})

    task = _PooledTask(
        task_type="single_index",
        properties=["ring_index"],
        seed=43,
        filters={"max_smiles_length": 90, "max_answer_list_length": 6},
        candidate_multiplier=6,
    )
    ds = task.to_grpo_dataset(split="train", num_samples=10)
    assert len(ds) > 0
    max_len = max(
        len(values)
        for answer in ds["answer"]
        for values in json.loads(answer).values()
    )
    assert max_len <= 6


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


def test_cached_dataset_manifest_rejects_stale_config(tmp_path):
    cfg = MultitaskDatasetConfig.from_dict(
        {
            "out_dir": str(tmp_path),
            "tasks": [
                {
                    "task_id": "si_ring",
                    "task_type": "single_index",
                    "properties": ["ring_index"],
                    "num_samples": 100,
                }
            ],
        }
    )
    (tmp_path / "multitask_manifest.json").write_text(
        json.dumps({"config_fingerprint": _dataset_config_fingerprint(cfg)}),
        encoding="utf-8",
    )
    validate_saved_multitask_dataset(tmp_path, cfg)

    changed = MultitaskDatasetConfig.from_dict(
        {
            "out_dir": str(tmp_path),
            "tasks": [
                {
                    "task_id": "si_ring",
                    "task_type": "single_index",
                    "properties": ["ring_index"],
                    "num_samples": 200,
                }
            ],
        }
    )
    try:
        validate_saved_multitask_dataset(tmp_path, changed)
    except ValueError as exc:
        assert "--overwrite-datasets" in str(exc)
    else:
        raise AssertionError("Expected stale dataset validation to fail.")
