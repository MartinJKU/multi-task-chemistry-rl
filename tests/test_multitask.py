from __future__ import annotations

import json
from pathlib import Path

import yaml

from grpo_reasoning.multitask.dataset import (
    MolecularIQTaskSpec,
    MultitaskDatasetConfig,
    _largest_remainder_counts,
    _read_task_accuracies,
)


def _load_yaml(path: Path) -> dict:
    # Load configs without importing grpo_reasoning.common.utils, which pulls in torch.
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "multitask"

# Property sets that uniquely identify each base task family, used to check that
# every curriculum stage after the first replays all previously-learned skills.
_COUNT_PROPERTY_SETS = {
    ("ring_count",),
    ("aromatic_ring_count",),
    ("fused_ring_count",),
    ("carbon_atom_count",),
    ("hetero_atom_count",),
    ("hba_count",),
    ("rotatable_bond_count",),
    ("aromatic_ring_count", "fused_ring_count", "ring_count"),
    ("carbon_atom_count", "halogen_atom_count", "hetero_atom_count"),
}


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


def _property_set(spec: MolecularIQTaskSpec) -> tuple[str, ...]:
    return tuple(sorted(spec.properties))


def test_curriculum_config_has_consolidation_stage_and_valid_stage_datasets():
    """The unified curriculum parses, ends in consolidation, and each stage is valid."""
    data = _load_yaml(_CONFIGS / "miq_curriculum.yaml")
    stages = data["stages"]
    stage_names = [stage["name"] for stage in stages]
    assert stage_names == [
        "01_counts",
        "02_simple_index",
        "03_full_index",
        "04_generation",
        "05_consolidation",
    ]
    # Every stage dataset must validate (this is what run_curriculum builds per stage).
    for stage in stages:
        MultitaskDatasetConfig.from_dict(stage["dataset"])


def test_every_stage_after_first_replays_all_count_tasks():
    """Full replay: stages 2+ must include all 9 count task families to avoid forgetting."""
    stages = _load_yaml(_CONFIGS / "miq_curriculum.yaml")["stages"]
    for stage in stages[1:]:
        specs = [MolecularIQTaskSpec.from_dict(t) for t in stage["dataset"]["tasks"]]
        present = {_property_set(spec) for spec in specs}
        missing = _COUNT_PROPERTY_SETS - present
        assert not missing, f"stage {stage['name']} missing count replay for {missing}"


def test_consolidation_stage_covers_all_sixteen_tasks():
    """The consolidation stage must train on every task family in the eval suite."""
    stages = _load_yaml(_CONFIGS / "miq_curriculum.yaml")["stages"]
    consolidation = next(s for s in stages if s["name"] == "05_consolidation")
    eval_cfg = MultitaskDatasetConfig.from_dict(
        _load_yaml(_CONFIGS / "miq_eval_suite.yaml")
    )
    consolidation_props = {
        _property_set(MolecularIQTaskSpec.from_dict(t))
        for t in consolidation["dataset"]["tasks"]
    }
    eval_props = {_property_set(spec) for spec in eval_cfg.tasks}
    assert eval_props == consolidation_props


def test_eval_suite_filters_all_index_tasks():
    """Index tasks in the shared eval suite must be filtered so exact match is achievable."""
    cfg = MultitaskDatasetConfig.from_dict(_load_yaml(_CONFIGS / "miq_eval_suite.yaml"))
    index_specs = [s for s in cfg.tasks if s.task_type in {"single_index", "multi_index"}]
    assert index_specs, "expected index tasks in the eval suite"
    for spec in index_specs:
        assert spec.filters, f"{spec.task_id} must define length filters for eval"
        assert spec.filters.get("max_answer_list_length")
