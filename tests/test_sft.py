from __future__ import annotations

from grpo_reasoning.common.sft import _completion_from_row


def test_sft_completion_for_index_row_teaches_no_extra_atoms():
    """Verify SFT index targets include exact-answer and anti-superset wording."""
    row = {
        "task_type": "single_index",
        "smiles": "CCC1CCC1",
        "answer": '{"ring_index": [2, 3, 4, 5]}',
    }
    completion = _completion_from_row(row)
    assert completion is not None
    assert "return no extra atoms" in completion
    assert "<answer>{\"ring_index\": [2,3,4,5]}</answer>" in completion


def test_sft_completion_for_empty_index_row_teaches_empty_list():
    """Verify empty index targets are explicitly supervised as empty lists."""
    row = {
        "task_type": "single_index",
        "smiles": "CCO",
        "answer": '{"ring_index": []}',
    }
    completion = _completion_from_row(row)
    assert completion is not None
    assert "empty list" in completion
    assert "<answer>{\"ring_index\": []}</answer>" in completion


def test_sft_completion_skips_constraint_generation_rows():
    """Constraint generation rows need molecule synthesis, not target echoing."""
    row = {
        "task_type": "constraint_generation",
        "answer": '[{"property": "ring_count", "operator": "=", "value": 1}]',
    }
    assert _completion_from_row(row) is None
