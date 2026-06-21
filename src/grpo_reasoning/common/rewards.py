from __future__ import annotations

import json
import re
from numbers import Number
from typing import Callable

from .prompts import extract_xml_answer

# Strict R1-style format: exactly one <reasoning>...</reasoning>\n<answer>...</answer>,
_FORMAT_PATTERN = re.compile(
    r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$",
    re.DOTALL,
)


def _completion_text(completion) -> str:
    """Normalize a completion object to text.

    Args:
        completion: TRL completion object or plain string.

    Returns:
        Assistant content text when available, otherwise `str(completion)`.
    """
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def format_reward(completions, **_) -> list[float]:
    """Score strict R1-style output format.

    Args:
        completions: Batch of completion objects from TRL.
        **_: Extra reward-function keyword arguments ignored by this reward.

    Returns:
        Reward list with 1.0 for strict format matches and 0.0 otherwise.
    """
    texts = [_completion_text(c) for c in completions]
    return [1.0 if _FORMAT_PATTERN.match(t) else 0.0 for t in texts]


def soft_format_reward(completions, **_) -> list[float]:
    """Score loose presence of reasoning and answer tags.

    Args:
        completions: Batch of completion objects from TRL.
        **_: Extra reward-function keyword arguments ignored by this reward.

    Returns:
        Reward list with 0.5 when both tag pairs appear and 0.0 otherwise.
    """
    texts = [_completion_text(c) for c in completions]
    out = []
    for t in texts:
        has_reasoning = "<reasoning>" in t and "</reasoning>" in t
        has_answer = "<answer>" in t and "</answer>" in t
        out.append(0.5 if (has_reasoning and has_answer) else 0.0)
    return out


def make_exact_match_reward(weight: float = 2.0) -> Callable:
    """Create an exact-match correctness reward.

    Args:
        weight: Reward value assigned to each exact answer match.

    Returns:
        Reward function that compares extracted answers with gold answers.
    """

    def correctness_reward(completions, answer, **_) -> list[float]:
        """Score extracted answers by exact string match.

        Args:
            completions: Batch of completion objects from TRL.
            answer: Batch of gold answer strings.
            **_: Extra reward-function keyword arguments ignored by this reward.

        Returns:
            Reward list with `weight` for matches and 0.0 otherwise.
        """
        texts = [_completion_text(c) for c in completions]
        extracted = [extract_xml_answer(t) for t in texts]
        return [weight if e == a else 0.0 for e, a in zip(extracted, answer)]

    correctness_reward.__name__ = "correctness_reward"
    return correctness_reward


def make_moleculariq_reward(task_type: str, weight: float = 2.0) -> Callable:
    """Create a MolecularIQ correctness reward.

    Args:
        task_type: MolecularIQ task variant used for scoring.
        weight: Reward value assigned when MolecularIQ scoring returns a full match.

    Returns:
        Reward function that scores completions via `moleculariq_core.evaluate_answer`.
    """
    def correctness_reward(completions, answer, **_) -> list[float]:
        """Score MolecularIQ answers against JSON-encoded targets.

        Args:
            completions: Batch of completion objects from TRL.
            answer: Batch of JSON-encoded MolecularIQ targets.
            **_: Extra reward-function keyword arguments ignored by this reward.

        Returns:
            Reward list with `weight` for full MolecularIQ matches and 0.0 otherwise.
        """
        texts = [_completion_text(c) for c in completions]
        scores: list[float] = []
        for text, gold_json in zip(texts, answer):
            s = _score_moleculariq_completion(text, gold_json, task_type)
            scores.append(weight if s >= 1.0 else 0.0)
        return scores

    correctness_reward.__name__ = "moleculariq_correctness_reward"
    return correctness_reward


# ---------------------------------------------------------------------------
# Verifier-backed scoring
#
# A single source of truth: ``moleculariq_core.evaluate_answer`` is the official
# verifier. Both the exact-match verdict and the dense partial-credit signal are
# derived from one verifier call (``return_details=True``) so they can never
# disagree. The verifier returns a binary ``reward`` plus a ``details`` report
# that exposes its own canonically parsed predictions and targets; partial
# credit is computed on top of *those* values rather than re-parsing the model
# output, which keeps shaping consistent with how correctness is judged.
# ---------------------------------------------------------------------------


def _parse_json(value):
    """Parse JSON-like strings, returning None on malformed inputs."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _as_number(value) -> float | None:
    """Convert plain numeric values to float; reject bools and structured values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)
    return None


def _numeric_closeness(predicted, target) -> float:
    """Score numeric predictions with exact match or smooth distance-based credit."""
    pred = _as_number(predicted)
    gold = _as_number(target)
    if pred is None or gold is None:
        return 0.0
    if pred == gold:
        return 1.0
    scale = max(abs(gold), 1.0)
    return max(0.0, 1.0 / (1.0 + abs(pred - gold) / scale))


def _as_int_set(value) -> set[int] | None:
    """Convert an index list to a set of ints when possible."""
    if not isinstance(value, list):
        return None
    out: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.add(item)
    return out


def _index_jaccard(predicted, target) -> float:
    """Score predicted atom-index sets with Jaccard overlap (intersection/union).

    Jaccard is used instead of Dice/F1 because it penalizes over-prediction much
    harder: a molecule-independent "dump 0..N" policy has perfect recall and
    earns a misleadingly high F1, but its Jaccard is only |gold|/|prediction|.
    This keeps the partial score honest -- it stays low unless the predicted set
    is genuinely specific to the molecule.
    """
    pred = _as_int_set(predicted)
    gold = _as_int_set(target)
    if pred is None or gold is None:
        return 0.0
    if not pred and not gold:
        return 1.0
    union = len(pred | gold)
    if union == 0:
        return 0.0
    return len(pred & gold) / union


def _verifier_report(
    completion_text: str,
    gold_answer: str,
    task_type: str,
) -> dict | None:
    """Run the official MolecularIQ verifier once with detailed output.

    Args:
        completion_text: Full assistant completion text.
        gold_answer: JSON-encoded target (count/index) or constraint list.
        task_type: MolecularIQ task type for this example.

    Returns:
        The verifier's detail dictionary, or None when the answer is missing,
        not a well-formed JSON object, or the gold is unparseable. The dictionary
        always carries a ``reward`` key (the binary exact-match verdict) plus a
        ``details`` report with the verifier's own parsed predictions and targets.

    Note:
        The model answer must be a strict JSON object (the format every task
        prompt asks for). The verifier itself tolerates loose formats such as
        bare ``0,2,3`` strings, but accepting those during training lets the
        model drift to malformed output (e.g. dropping the closing ``]``), so we
        require valid JSON here to keep the format pressure that drives clean,
        parseable answers.
    """
    from moleculariq_core import evaluate_answer

    extracted = extract_xml_answer(completion_text)
    if not extracted:
        return None
    if not isinstance(_parse_json(extracted), dict):
        return None
    target = _parse_json(gold_answer)
    if target is None:
        return None

    try:
        if task_type == "constraint_generation":
            return evaluate_answer(
                task_type=task_type,
                predicted=extracted,
                constraints=target,
                return_details=True,
            )
        return evaluate_answer(
            task_type=task_type,
            predicted=extracted,
            target=target,
            return_details=True,
        )
    except Exception:
        return None


def _exact_score(report: dict | None) -> float:
    """Return the verifier's binary exact-match score from a report."""
    if not report:
        return 0.0
    try:
        return float(report.get("reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _count_graded(report: dict) -> float:
    """Average numeric closeness over the verifier's per-property count details."""
    details = report.get("details") or {}
    if not isinstance(details, dict) or not details:
        return 0.0
    scores = [
        _numeric_closeness(entry.get("predicted"), entry.get("target"))
        for entry in details.values()
        if isinstance(entry, dict)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _index_graded(report: dict) -> float:
    """Average atom-set F1 over the verifier's per-property index details."""
    details = report.get("details") or {}
    if not isinstance(details, dict) or not details:
        return 0.0
    scores = [
        _index_jaccard(entry.get("predicted"), entry.get("target"))
        for entry in details.values()
        if isinstance(entry, dict)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _constraint_graded(report: dict) -> tuple[float, bool]:
    """Score constraint generation from the verifier's per-constraint details.

    Returns:
        Tuple of partial-credit score in [0, 1] and a valid-SMILES flag. Credit
        combines per-constraint satisfaction (numeric closeness when a supported
        constraint is missed) and requires a valid, reasonable molecule.
    """
    valid_smiles = bool(report.get("valid_smiles"))
    details = report.get("details") or []
    total = report.get("total") or len(details)

    if not total:
        # No constraints to satisfy; credit a valid, reasonable molecule.
        return (1.0 if valid_smiles else 0.0), valid_smiles
    if not valid_smiles or not isinstance(details, list):
        return 0.0, valid_smiles

    scores: list[float] = []
    for entry in details:
        if not isinstance(entry, dict):
            scores.append(0.0)
        elif entry.get("satisfied"):
            scores.append(1.0)
        elif entry.get("supported"):
            constraint = entry.get("constraint") or {}
            scores.append(_numeric_closeness(entry.get("actual"), constraint.get("value")))
        else:
            scores.append(0.0)
    return sum(scores) / total, valid_smiles


def _graded_score(report: dict | None, task_type: str) -> tuple[float, bool]:
    """Return task-shaped partial credit and a SMILES validity flag.

    Partial credit is derived entirely from the official verifier's parsed
    output so it never diverges from the exact-match verdict.

    Args:
        report: Verifier detail report, or None when scoring was not possible.
        task_type: MolecularIQ task type for this example.

    Returns:
        Tuple of partial-credit score in [0, 1] and a valid-SMILES flag (only
        meaningful for constraint-generation tasks).
    """
    if report is None:
        return 0.0, False
    if task_type in {"single_count", "multi_count"}:
        return _count_graded(report), False
    if task_type in {"single_index", "multi_index"}:
        return _index_graded(report), False
    if task_type == "constraint_generation":
        return _constraint_graded(report)
    return 0.0, False


def _score_moleculariq_completion(
    completion_text: str,
    gold_answer: str,
    task_type: str,
) -> float:
    """Score one MolecularIQ completion with the official binary verifier.

    Args:
        completion_text: Full assistant completion text.
        gold_answer: JSON-encoded target or constraints.
        task_type: MolecularIQ task type for this example.

    Returns:
        1.0 when the verifier reports a full match, otherwise 0.0.
    """
    return _exact_score(_verifier_report(completion_text, gold_answer, task_type))


def make_moleculariq_multitask_reward(weight: float = 2.0) -> Callable:
    """Create a MolecularIQ correctness reward dispatched per example.

    Args:
        weight: Reward value assigned when a row's task-specific scorer fully matches.

    Returns:
        Reward function that reads the dataset `task_type` column for each row.
    """

    def correctness_reward(completions, answer, task_type=None, **_) -> list[float]:
        """Score mixed MolecularIQ completions against row-specific task types."""
        if task_type is None:
            raise ValueError(
                "Multitask MolecularIQ reward requires a `task_type` dataset column."
            )
        texts = [_completion_text(c) for c in completions]
        return [
            weight if _score_moleculariq_completion(text, gold, t_type) >= 1.0 else 0.0
            for text, gold, t_type in zip(texts, answer, task_type)
        ]

    correctness_reward.__name__ = "moleculariq_multitask_correctness_reward"
    return correctness_reward


def make_moleculariq_shaped_reward(
    task_type: str | None = None,
    weight: float = 1.0,
    smiles_validity_weight: float = 0.5,
) -> Callable:
    """Create a MolecularIQ partial-credit reward.

    Args:
        task_type: Fixed task type for single-task runs. If None, read per-row task_type.
        weight: Maximum task-shaped partial-credit reward.
        smiles_validity_weight: Extra reward for valid generated SMILES.

    Returns:
        Reward function for count, index, and constraint-generation tasks.
    """

    fixed_task_type = task_type

    def shaped_reward(
        completions,
        answer,
        task_type: list[str] | None = None,
        **_,
    ) -> list[float]:
        """Score MolecularIQ completions with task-specific partial credit."""
        row_task_types = task_type
        if row_task_types is None:
            if fixed_task_type is None:
                raise ValueError(
                    "Shaped MolecularIQ reward requires a fixed task_type or a "
                    "`task_type` dataset column."
                )
            row_task_types = [fixed_task_type] * len(completions)

        scores: list[float] = []
        for completion, gold, row_task_type in zip(completions, answer, row_task_types):
            report = _verifier_report(
                _completion_text(completion),
                gold,
                row_task_type,
            )
            score, valid_smiles = _graded_score(report, row_task_type)
            reward = weight * score
            if row_task_type == "constraint_generation" and valid_smiles:
                reward += smiles_validity_weight
            scores.append(reward)
        return scores

    shaped_reward.__name__ = "moleculariq_shaped_reward"
    return shaped_reward


def moleculariq_diagnostics(
    completion_text: str,
    gold_answer: str,
    task_type: str,
) -> dict[str, float | bool]:
    """Compute diagnostic metrics for one MolecularIQ completion.

    Args:
        completion_text: Full assistant completion text.
        gold_answer: JSON-encoded target answer.
        task_type: MolecularIQ task type for this example.

    Returns:
        Dictionary with answer parsing, exact-match verdict, partial score, and
        SMILES validity flags, all derived from the official verifier.
    """
    extracted = extract_xml_answer(completion_text)
    parsed = _parse_json(extracted) if extracted else None
    report = _verifier_report(completion_text, gold_answer, task_type)
    partial_score, valid_smiles = _graded_score(report, task_type)
    return {
        "answer_present": bool(extracted),
        "json_valid": parsed is not None,
        "exact_match": _exact_score(report),
        "partial_score": float(partial_score),
        "valid_smiles": bool(valid_smiles),
    }
