from __future__ import annotations

from grpo_reasoning.common.sft import (
    SFTArgs,
    _canonical_task_family,
    _completion_from_row,
)
from grpo_reasoning.common.smiles import smiles_atom_tokens


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
    assert "Atom map: 0:C; 1:C; 2:C; 3:C; 4:C; 5:C" in completion
    assert "ring-closure digits do not create atoms" in completion
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


def test_sft_completion_teaches_branch_and_multi_character_atom_indexing():
    row = {
        "task_type": "single_index",
        "smiles": "CC(=O)NCl",
        "answer": '{"hetero_atom_index": [2, 3, 4]}',
    }
    completion = _completion_from_row(row)
    assert completion is not None
    assert "Atom map: 0:C; 1:C; 2:O; 3:N; 4:Cl" in completion
    assert "hetero_atom_index selects exactly [2, 3, 4]" in completion


def test_smiles_atom_tokens_skip_syntax_but_keep_bracket_atoms():
    assert smiles_atom_tokens("C[C@@H](O)c1cc[nH]c1Cl") == [
        "C",
        "C",
        "O",
        "c",
        "c",
        "c",
        "n",
        "c",
        "Cl",
    ]


def test_sft_task_family_collapses_nonempty_replay_aliases():
    assert _canonical_task_family("si_ring_nonempty_replay") == "si_ring"
    assert SFTArgs().samples_per_task_family is None


def test_sft_completion_skips_constraint_generation_rows():
    """Constraint generation rows need molecule synthesis, not target echoing."""
    row = {
        "task_type": "constraint_generation",
        "answer": '[{"property": "ring_count", "operator": "=", "value": 1}]',
    }
    assert _completion_from_row(row) is None
