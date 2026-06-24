from __future__ import annotations

from grpo_reasoning.common.prompts import build_chat_prompt
from grpo_reasoning.common.tasks import get_task


def _user_turns(messages: list[dict]) -> list[str]:
    """Return the content of every user message in a chat prompt."""
    return [m["content"] for m in messages if m["role"] == "user"]


def test_build_chat_prompt_supports_multi_shot():
    """Verify multiple few-shot pairs render as alternating user/assistant turns."""
    messages = build_chat_prompt(
        question="Q3",
        task_instructions="do the thing",
        few_shot_examples=[("Q1", "A1"), ("Q2", "A2")],
    )
    roles = [m["role"] for m in messages]
    # system, then two (user, assistant) demo pairs, then the real user question.
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert _user_turns(messages) == ["Q1", "Q2", "Q3"]


def test_build_chat_prompt_examples_take_precedence_over_single_pair():
    """Verify few_shot_examples overrides the legacy single-shot pair."""
    messages = build_chat_prompt(
        question="Q",
        task_instructions="",
        few_shot_question="legacy_q",
        few_shot_answer="legacy_a",
        few_shot_examples=[("multi_q", "multi_a")],
    )
    assert "legacy_q" not in _user_turns(messages)
    assert "multi_q" in _user_turns(messages)


def test_index_task_uses_multi_shot_with_empty_list_demo():
    """Verify index tasks carry a multi-shot demo that includes the empty-list case.

    The single benzene example taught the degenerate "contiguous low range, never
    empty" policy; the multi-shot demo must show both an offset ring and [].
    """
    task = get_task("moleculariq", task_type="single_index", properties=["ring_index"])
    assert task.few_shot_examples is not None
    assert len(task.few_shot_examples) >= 2
    demo_answers = " ".join(answer for _, answer in task.few_shot_examples)
    assert "[]" in demo_answers  # empty-list case is demonstrated


def test_single_index_demos_match_requested_property():
    """Carbon indexing must not receive only ring-index demonstrations."""
    task = get_task(
        "moleculariq",
        task_type="single_index",
        properties=["carbon_atom_index"],
    )
    demo_text = " ".join(
        question + answer for question, answer in (task.few_shot_examples or [])
    )
    assert "carbon_atom_index" in demo_text
    assert "Atom map:" in demo_text
    assert "ring_index" not in demo_text


def test_count_task_keeps_single_shot_demo():
    """Verify non-index tasks retain the legacy single-shot demonstration."""
    task = get_task("moleculariq", task_type="single_count", properties=["ring_count"])
    assert task.few_shot_examples is None
    assert task.few_shot_question is not None
    assert task.few_shot_answer is not None


def test_constraint_generation_uses_property_aware_examples():
    """Verify ring generation demos include actual ring-containing SMILES."""
    task = get_task(
        "moleculariq",
        task_type="constraint_generation",
        properties=["ring_count"],
    )
    assert task.few_shot_examples is not None
    demo_answers = " ".join(answer for _, answer in task.few_shot_examples)
    assert "C1CCCCC1" in demo_answers
    assert "C1CCC2CCCCC2C1" in demo_answers
    assert task.few_shot_question is None
