from __future__ import annotations

import json

from grpo_reasoning.multitask.gates import check_index_gate


def _summary(mode: float = 0.3) -> dict:
    return {
        "tasks": [
            {
                "task_id": "si_ring",
                "task_type": "single_index",
                "accuracy": 0.05,
                "partial_score_mean": 0.4,
                "distinct_answer_rate": 0.3,
                "most_common_answer_rate": mode,
                "index_empty_gold_accuracy": 0.2,
            },
            {
                "task_id": "si_carbon_atom",
                "task_type": "single_index",
                "accuracy": 0.02,
                "partial_score_mean": 0.3,
                "distinct_answer_rate": 0.2,
                "most_common_answer_rate": mode,
            },
        ]
    }


def test_index_gate_accepts_noncollapsed_checkpoint(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")
    metrics = check_index_gate(path)
    assert metrics["macro_accuracy"] > 0.01


def test_index_gate_rejects_dominant_answer_template(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(mode=0.95)), encoding="utf-8")
    try:
        check_index_gate(path)
    except ValueError as exc:
        assert "worst_mode" in str(exc)
    else:
        raise AssertionError("Expected collapsed checkpoint to fail the gate.")
