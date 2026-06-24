from __future__ import annotations

from grpo_reasoning.common.rewards import (
    format_reward,
    make_exact_match_reward,
    make_moleculariq_index_reasoning_reward,
    make_moleculariq_shaped_reward,
    make_reasoning_quality_reward,
    moleculariq_diagnostics,
    soft_format_reward,
)


def _conv(text: str):
    """Wrap a string completion as a conversational completion.

    Args:
        text: Assistant completion text.

    Returns:
        Completion object shaped like TRL conversational output.
    """
    return [{"role": "assistant", "content": text}]


def test_format_reward_strict_match():
    """Verify strict format reward accepts only exact scaffold matches.

    Args:
        None.

    Returns:
        None.
    """
    good = _conv("<reasoning>2+2=4</reasoning>\n<answer>4</answer>")
    bad = _conv("The answer is 4")
    assert format_reward([good, bad]) == [1.0, 0.0]


def test_format_reward_rejects_extra_text():
    """Verify strict format reward rejects trailing text.

    Args:
        None.

    Returns:
        None.
    """
    # Trailing text after </answer> should fail strict match
    msg = _conv("<reasoning>x</reasoning>\n<answer>4</answer> extra")
    assert format_reward([msg]) == [0.0]


def test_soft_format_partial_credit():
    """Verify soft format reward gives partial credit for both tag pairs.

    Args:
        None.

    Returns:
        None.
    """
    msg = _conv("<reasoning>x</reasoning> bla <answer>4</answer>")
    assert soft_format_reward([msg]) == [0.5]


def test_reasoning_quality_rejects_empty_and_echo_reasoning():
    """Verify the degenerate empty/echo reasoning collapse earns zero reward.

    Reproduces the failure seen in the index eval dumps, where every completion
    was ``<reasoning> aromatic_ring_index</reasoning>`` followed by a guess.
    """
    reward = make_reasoning_quality_reward(weight=0.5)
    empty = _conv("<reasoning></reasoning>\n<answer>{\"ring_index\": [0, 1]}</answer>")
    echo = _conv(
        "<reasoning> aromatic_ring_index</reasoning>\n"
        '<answer>{"aromatic_ring_index": [6, 7, 8]}</answer>'
    )
    assert reward(completions=[empty, echo]) == [0.0, 0.0]


def test_reasoning_quality_credits_substantive_reasoning():
    """Verify a real, number-bearing rationale earns the full weight."""
    reward = make_reasoning_quality_reward(weight=0.5)
    good = _conv(
        "<reasoning>Index the atoms of CCc1ccccc1: C=0, C=1, then the benzene"
        " ring spans atoms 2 through 7, so the ethyl carbons 0 and 1 are"
        " excluded.</reasoning>\n"
        '<answer>{"ring_index": [2, 3, 4, 5, 6, 7]}</answer>'
    )
    assert reward(completions=[good]) == [0.5]


def test_reasoning_quality_caps_so_padding_does_not_pay():
    """Verify credit is capped: a very long rationale cannot exceed the weight."""
    reward = make_reasoning_quality_reward(weight=0.5)
    padded = _conv(
        "<reasoning>" + " ".join(f"step{i} count 1" for i in range(200)) + "</reasoning>\n"
        '<answer>{"ring_index": [0]}</answer>'
    )
    assert reward(completions=[padded]) == [0.5]


def test_index_reasoning_reward_requires_actual_atom_map_and_answer_consistency():
    reward = make_moleculariq_index_reasoning_reward(weight=0.5)
    good = _conv(
        "<reasoning>Algorithm: scan left to right. Molecule: CC(=O)NCl."
        " Atom map: 0:C; 1:C; 2:O; 3:N; 4:Cl."
        " hetero_atom_index selects exactly [2, 3, 4]; return no extra atoms."
        "</reasoning>\n"
        '<answer>{"hetero_atom_index": [2, 3, 4]}</answer>'
    )
    generic = _conv(
        "<reasoning>Atom map: 0:C; 1:C; 2:C; 3:C; 4:C."
        " hetero_atom_index selects exactly [2, 3, 4].</reasoning>\n"
        '<answer>{"hetero_atom_index": [2, 3, 4]}</answer>'
    )
    out = reward(
        completions=[good, generic],
        task_type=["single_index", "single_index"],
        smiles=["CC(=O)NCl", "CC(=O)NCl"],
    )
    assert out[0] == 0.5
    assert out[1] < out[0]


def test_correctness_reward_exact_match():
    """Verify exact-match reward scores matching extracted answers.

    Args:
        None.

    Returns:
        None.
    """
    reward = make_exact_match_reward(weight=2.0)
    completions = [
        _conv("<reasoning>...</reasoning>\n<answer>4</answer>"),
        _conv("<reasoning>...</reasoning>\n<answer>7</answer>"),
    ]
    out = reward(completions=completions, answer=["4", "4"])
    assert out == [2.0, 0.0]


def test_correctness_reward_empty_extraction():
    """Verify exact-match reward handles missing answer tags.

    Args:
        None.

    Returns:
        None.
    """
    reward = make_exact_match_reward()
    completions = [_conv("no tags here")]
    assert reward(completions=completions, answer=["42"]) == [0.0]


def test_moleculariq_shaped_count_closeness():
    """Verify count tasks receive numeric partial credit."""
    reward = make_moleculariq_shaped_reward(task_type="single_count", weight=1.0)
    completions = [_conv('<reasoning>x</reasoning>\n<answer>{"ring_count": 8}</answer>')]
    out = reward(completions=completions, answer=['{"ring_count": 10}'])
    assert out == [1 / 3]


def test_moleculariq_shaped_multi_count_averages_keys():
    """Verify multi-count partial credit averages target keys."""
    reward = make_moleculariq_shaped_reward(task_type="multi_count", weight=1.0)
    completions = [
        _conv(
            "<reasoning>x</reasoning>\n"
            '<answer>{"ring_count": 2, "aromatic_ring_count": 0}</answer>'
        )
    ]
    out = reward(
        completions=completions,
        answer=['{"ring_count": 2, "aromatic_ring_count": 1}'],
    )
    assert 0.7 < out[0] < 0.8


def test_moleculariq_shaped_index_overlap():
    """Verify index tasks receive precision-biased set-overlap partial credit."""
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    completions = [
        _conv('<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 9]}</answer>')
    ]
    out = reward(completions=completions, answer=['{"ring_index": [0, 1, 2]}'])
    # Tversky: TP=2, FP=1, FN=1 -> 2 / (2 + 3 + 1)
    assert out == [2 / 6]


def test_shaped_index_overlap_punishes_overprediction():
    """Verify dumping a long consecutive list scores low (anti reward-hacking).

    Mirrors the degenerate "{ring_index: [1..N]}" policy seen in training: it has
    perfect recall but should be punished for over-prediction.
    """
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    dump = [
        _conv(
            "<reasoning>x</reasoning>\n"
            '<answer>{"ring_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}</answer>'
        )
    ]
    out = reward(completions=dump, answer=['{"ring_index": [0, 1, 2]}'])
    # Precision-biased Tversky: TP=3, FP=7, FN=0 -> 3 / (3 + 21)
    assert out[0] == 3 / 24


def test_shaped_index_penalizes_contiguous_guess_for_sparse_target():
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    out = reward(
        completions=[
            _conv(
                "<reasoning>x</reasoning>\n"
                '<answer>{"carbon_atom_index": [0, 1, 2, 3, 4]}</answer>'
            )
        ],
        answer=['{"carbon_atom_index": [0, 2, 4]}'],
    )
    # Base overlap is 3 / (3 + 3*2), then the contiguous shortcut gets x0.1.
    assert out == [(3 / 9) * 0.1]


def test_shaped_index_empty_gold_rejects_nonempty_prediction():
    """Verify empty index targets strongly reject non-empty predictions."""
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    out = reward(
        completions=[
            _conv('<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1]}</answer>')
        ],
        answer=['{"ring_index": []}'],
    )
    assert out == [0.0]


def test_moleculariq_shaped_constraint_valid_smiles_if_rdkit_available():
    """Verify constraint generation gives validity reward for parseable SMILES."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return

    reward = make_moleculariq_shaped_reward(
        task_type="constraint_generation",
        weight=1.0,
        smiles_validity_weight=0.5,
    )
    completions = [
        _conv('<reasoning>x</reasoning>\n<answer>{"smiles": "c1ccccc1"}</answer>')
    ]
    out = reward(
        completions=completions,
        answer=['[{"property": "ring_count", "operator": "=", "value": 1}]'],
    )
    assert out == [1.5]


def test_constraint_nontrivial_smiles_bonus_if_rdkit_available():
    """Verify non-trivial generated SMILES can receive an optional bonus."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return

    reward = make_moleculariq_shaped_reward(
        task_type="constraint_generation",
        weight=1.0,
        smiles_validity_weight=0.1,
        smiles_nontrivial_weight=0.2,
    )
    out = reward(
        completions=[
            _conv('<reasoning>x</reasoning>\n<answer>{"smiles": "CCO"}</answer>')
        ],
        answer=['[{"property": "carbon_atom_count", "operator": "=", "value": 2}]'],
    )
    assert out == [1.3]


def test_constraint_diagnostics_include_target_signature_if_rdkit_available():
    """Verify constraint diagnostics preserve target identity for reporting."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return

    diagnostics = moleculariq_diagnostics(
        '<reasoning>x</reasoning>\n<answer>{"smiles": "CCC"}</answer>',
        '[{"property": "carbon_atom_count", "operator": "=", "value": 3}]',
        "constraint_generation",
    )

    assert diagnostics["constraint_target_signature"] == "carbon_atom_count=3"
    assert diagnostics["constraint_target_property"] == "carbon_atom_count"
    assert diagnostics["constraint_target_operator"] == "="
    assert diagnostics["constraint_target_value"] == 3.0
    assert diagnostics["constraint_actual_value"] == 3.0


def test_moleculariq_diagnostics_reports_partial_score():
    """Verify diagnostic metrics expose parsing and partial score information."""
    completion = '<reasoning>x</reasoning>\n<answer>{"ring_count": 8}</answer>'
    out = moleculariq_diagnostics(completion, '{"ring_count": 10}', "single_count")
    assert out["answer_present"] is True
    assert out["json_valid"] is True
    assert out["partial_score"] == 1 / 3
    # Exact-match verdict and partial credit come from the same verifier call.
    assert out["exact_match"] == 0.0


def test_moleculariq_diagnostics_exact_match_on_perfect_answer():
    """Verify a perfect index answer reports exact_match=1.0 and partial=1.0."""
    completion = '<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 2]}</answer>'
    out = moleculariq_diagnostics(completion, '{"ring_index": [0, 1, 2]}', "single_index")
    assert out["exact_match"] == 1.0
    assert out["partial_score"] == 1.0
    assert out["index_precision"] == 1.0
    assert out["index_recall"] == 1.0


def test_moleculariq_diagnostics_reports_index_overprediction():
    """Verify diagnostics expose broad-span index failures."""
    completion = (
        '<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 2, 3]}</answer>'
    )
    out = moleculariq_diagnostics(completion, '{"ring_index": [1, 2]}', "single_index")
    assert out["index_precision"] == 0.5
    assert out["index_recall"] == 1.0
    assert out["index_false_positives"] == 2.0
    assert out["index_false_negatives"] == 0.0
    assert out["index_superset"] is True


def test_malformed_json_answer_gets_no_credit():
    """Verify a malformed JSON answer earns no reward even if the verifier is lenient.

    The model in an earlier run dropped the closing ``]`` (e.g.
    ``{"ring_index": [1, 2, 3}``); the official verifier still parsed the
    integers, so without a strict-JSON gate the model could farm partial credit
    on malformed output. Both the shaped reward and the diagnostics must reject it.
    """
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    malformed = [
        _conv('<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 2, 3}</answer>')
    ]
    out = reward(completions=malformed, answer=['{"ring_index": [0, 1, 2, 3]}'])
    assert out == [0.0]
    diag = moleculariq_diagnostics(
        malformed[0][0]["content"], '{"ring_index": [0, 1, 2, 3]}', "single_index"
    )
    assert diag["json_valid"] is False
    assert diag["exact_match"] == 0.0
    assert diag["partial_score"] == 0.0


def test_shaped_index_partial_is_dense_but_exact_is_zero():
    """Verify a near-miss index answer earns dense credit while exact match fails.

    This is the property that lets GRPO learn set-valued index tasks: the shaped
    reward is well above zero even though the all-or-nothing verdict is wrong.
    """
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    near_miss = [
        _conv('<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 2, 3]}</answer>')
    ]
    out = reward(completions=near_miss, answer=['{"ring_index": [0, 1, 2, 3, 4]}'])
    assert 0.0 < out[0] < 1.0
    diag = moleculariq_diagnostics(near_miss[0][0]["content"], '{"ring_index": [0, 1, 2, 3, 4]}', "single_index")
    assert diag["exact_match"] == 0.0
