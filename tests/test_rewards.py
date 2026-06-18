from __future__ import annotations

from grpo_reasoning.common.rewards import (
    format_reward,
    make_exact_match_reward,
    make_moleculariq_shaped_reward,
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
    assert 0.8 < out[0] < 0.9


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
    """Verify index tasks receive set-overlap partial credit."""
    reward = make_moleculariq_shaped_reward(task_type="single_index", weight=1.0)
    completions = [
        _conv('<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 9]}</answer>')
    ]
    out = reward(completions=completions, answer=['{"ring_index": [0, 1, 2]}'])
    assert out == [2 * 2 / 6]


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


def test_moleculariq_diagnostics_reports_partial_score():
    """Verify diagnostic metrics expose parsing and partial score information."""
    completion = '<reasoning>x</reasoning>\n<answer>{"ring_count": 8}</answer>'
    out = moleculariq_diagnostics(completion, '{"ring_count": 10}', "single_count")
    assert out["answer_present"] is True
    assert out["json_valid"] is True
    assert 0.8 < out["partial_score"] < 0.9
    # Exact-match verdict and partial credit come from the same verifier call.
    assert out["exact_match"] == 0.0


def test_moleculariq_diagnostics_exact_match_on_perfect_answer():
    """Verify a perfect index answer reports exact_match=1.0 and partial=1.0."""
    completion = '<reasoning>x</reasoning>\n<answer>{"ring_index": [0, 1, 2]}</answer>'
    out = moleculariq_diagnostics(completion, '{"ring_index": [0, 1, 2]}', "single_index")
    assert out["exact_match"] == 1.0
    assert out["partial_score"] == 1.0


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
