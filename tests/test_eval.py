from __future__ import annotations

from grpo_reasoning.common.eval import _add_constraint_metrics, _add_index_metrics


def test_index_metrics_separate_empty_and_nonempty_targets():
    """Verify exact index accuracy is not inflated by empty-target examples."""
    metrics = {}
    rows = [
        {
            "exact_match": 1.0,
            "partial_score": 1.0,
            "index_empty_gold": True,
            "index_precision": 1.0,
            "index_recall": 1.0,
        },
        {
            "exact_match": 0.0,
            "partial_score": 0.5,
            "index_empty_gold": False,
            "index_precision": 0.75,
            "index_recall": 0.5,
        },
        {
            "exact_match": 1.0,
            "partial_score": 1.0,
            "index_empty_gold": False,
            "index_precision": 1.0,
            "index_recall": 1.0,
        },
    ]

    _add_index_metrics(metrics, rows)

    assert metrics["index_gold_empty_rate"] == 1 / 3
    assert metrics["index_empty_gold_total"] == 1
    assert metrics["index_nonempty_total"] == 2
    assert metrics["index_empty_gold_accuracy"] == 1.0
    assert metrics["index_nonempty_accuracy"] == 0.5
    assert metrics["index_nonempty_partial_score_mean"] == 0.75
    assert metrics["index_nonempty_precision_mean"] == 0.875
    assert metrics["index_nonempty_recall_mean"] == 0.75


def test_constraint_metrics_surface_target_and_generation_collapse():
    """Verify generation metrics expose target misses and repeated SMILES."""
    metrics = {}
    rows = [
        {
            "exact_match": 1.0,
            "constraint_satisfied_fraction": 1.0,
            "canonical_smiles": "CC",
            "constraint_target_signature": "carbon_atom_count=2",
        },
        {
            "exact_match": 1.0,
            "constraint_satisfied_fraction": 1.0,
            "canonical_smiles": "CC",
            "constraint_target_signature": "carbon_atom_count=2",
        },
        {
            "exact_match": 0.0,
            "constraint_satisfied_fraction": 0.0,
            "canonical_smiles": "CC",
            "constraint_target_signature": "carbon_atom_count=3",
        },
        {
            "exact_match": 1.0,
            "constraint_satisfied_fraction": 1.0,
            "canonical_smiles": "CCC",
            "constraint_target_signature": "carbon_atom_count=3",
        },
    ]

    _add_constraint_metrics(metrics, rows)

    assert metrics["canonical_smiles_distinct_rate"] == 0.5
    assert metrics["canonical_smiles_most_common_rate"] == 0.75
    assert metrics["successful_canonical_smiles_distinct_rate"] == 2 / 3
    assert metrics["successful_canonical_smiles_most_common_rate"] == 2 / 3
    assert metrics["constraint_target_count"] == 2
    assert metrics["constraint_target_macro_accuracy"] == 0.75
    assert metrics["worst_constraint_target_accuracy"] == 0.5
    assert metrics["constraint_target_breakdown"]["carbon_atom_count=2"] == {
        "accuracy": 1.0,
        "correct": 2,
        "total": 2,
    }
