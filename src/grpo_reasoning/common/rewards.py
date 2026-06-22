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

# Inner text of the reasoning block, used to score reasoning substance.
_REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")

# A genuine count/index rationale uses several distinct words and references
# concrete numbers; the degenerate "<reasoning> aromatic_ring_index</reasoning>"
# collapse seen in training has only the echoed key. These thresholds gate that
# collapse without paying for runaway padding (credit is capped and concave).
_MIN_UNIQUE_REASONING_TOKENS = 4
_TARGET_UNIQUE_REASONING_TOKENS = 12

# The index evals showed a high-recall/low-precision failure: answers such as
# ``[0, 1, ..., 18]`` overlap many gold atoms but almost never exactly match.
# These weights keep dense credit useful while making false positives expensive.
_INDEX_FALSE_POSITIVE_PENALTY = 1.5
_INDEX_FALSE_NEGATIVE_PENALTY = 0.5


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


def _reasoning_text(text: str) -> str:
    """Return the inner text of the first `<reasoning>...</reasoning>` block."""
    match = _REASONING_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def _reasoning_quality_score(reasoning: str) -> float:
    """Score reasoning substance in [0, 1] to escape the empty-reasoning collapse.

    GRPO with only format/correctness rewards converges to a degenerate optimum
    that leaves `<reasoning>` empty (or echoes the answer key) and guesses, which
    is fatal for tasks needing real per-molecule work (atom indexing, wide-range
    counts). This signal rewards reasoning that contains several distinct words
    and a concrete number, the minimal hallmark of an actual rationale.

    The score is concave and capped so the model cannot farm it by padding: once
    the reasoning has enough distinct tokens, extra length earns nothing.

    Args:
        reasoning: Inner text of the `<reasoning>` block.

    Returns:
        Quality score in [0, 1].
    """
    reasoning = reasoning.strip()
    if not reasoning:
        return 0.0
    unique_tokens = {tok.lower() for tok in _WORD_PATTERN.findall(reasoning)}
    if len(unique_tokens) < _MIN_UNIQUE_REASONING_TOKENS:
        return 0.0
    richness = min(1.0, len(unique_tokens) / _TARGET_UNIQUE_REASONING_TOKENS)
    # Numbers are near-universal in genuine count/index rationales; prose with no
    # numeric content earns only partial credit so pure padding stays cheap.
    has_number = any(ch.isdigit() for ch in reasoning)
    return richness if has_number else 0.5 * richness


def make_reasoning_quality_reward(weight: float = 0.5) -> Callable:
    """Create a reward for substantive (non-empty, non-echo) reasoning.

    Args:
        weight: Maximum reward for a fully substantive reasoning block.

    Returns:
        Reward function that scores the `<reasoning>` block of each completion.
    """

    def reasoning_quality_reward(completions, **_) -> list[float]:
        """Score the reasoning substance of each completion."""
        texts = [_completion_text(c) for c in completions]
        return [weight * _reasoning_quality_score(_reasoning_text(t)) for t in texts]

    reasoning_quality_reward.__name__ = "reasoning_quality_reward"
    return reasoning_quality_reward


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


def _index_confusion(predicted, target) -> dict[str, int] | None:
    """Return atom-index set confusion counts, or None for malformed values."""
    pred = _as_int_set(predicted)
    gold = _as_int_set(target)
    if pred is None or gold is None:
        return None
    true_positive = len(pred & gold)
    return {
        "tp": true_positive,
        "fp": len(pred - gold),
        "fn": len(gold - pred),
        "pred_len": len(pred),
        "gold_len": len(gold),
    }


def _index_precision_biased_overlap(predicted, target) -> float:
    """Score atom-index sets with a false-positive-biased Tversky overlap.

    Plain Jaccard still gave broad contiguous ranges too much credit in practice.
    This variant keeps the dense signal but makes over-prediction more costly
    than under-prediction, which is exactly the observed index failure mode.
    """
    counts = _index_confusion(predicted, target)
    if counts is None:
        return 0.0
    if counts["gold_len"] == 0:
        return 1.0 if counts["pred_len"] == 0 else 0.0
    denom = (
        counts["tp"]
        + _INDEX_FALSE_POSITIVE_PENALTY * counts["fp"]
        + _INDEX_FALSE_NEGATIVE_PENALTY * counts["fn"]
    )
    if denom <= 0:
        return 0.0
    return counts["tp"] / denom


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
    """Average precision-biased atom-set overlap over index details."""
    details = report.get("details") or {}
    if not isinstance(details, dict) or not details:
        return 0.0
    scores = [
        _index_precision_biased_overlap(entry.get("predicted"), entry.get("target"))
        for entry in details.values()
        if isinstance(entry, dict)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _index_diagnostics(report: dict | None) -> dict[str, float | bool]:
    """Compute precision/recall and over-prediction diagnostics for index tasks."""
    if report is None:
        return {}
    details = report.get("details") or {}
    if not isinstance(details, dict) or not details:
        return {}

    totals = {"tp": 0, "fp": 0, "fn": 0, "pred_len": 0, "gold_len": 0}
    found = False
    for entry in details.values():
        if not isinstance(entry, dict):
            continue
        counts = _index_confusion(entry.get("predicted"), entry.get("target"))
        if counts is None:
            continue
        found = True
        for key in totals:
            totals[key] += counts[key]
    if not found:
        return {}

    pred_len = totals["pred_len"]
    gold_len = totals["gold_len"]
    precision = (
        totals["tp"] / pred_len
        if pred_len
        else (1.0 if gold_len == 0 else 0.0)
    )
    recall = (
        totals["tp"] / gold_len
        if gold_len
        else (1.0 if pred_len == 0 else 0.0)
    )
    return {
        "index_precision": float(precision),
        "index_recall": float(recall),
        "index_pred_len": float(pred_len),
        "index_gold_len": float(gold_len),
        "index_false_positives": float(totals["fp"]),
        "index_false_negatives": float(totals["fn"]),
        "index_empty_gold": gold_len == 0,
        "index_empty_pred": pred_len == 0,
        "index_empty_gold_nonempty_pred": gold_len == 0 and pred_len > 0,
        "index_superset": totals["fn"] == 0 and totals["fp"] > 0,
        "index_subset": totals["fp"] == 0 and totals["fn"] > 0,
    }


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


def _constraint_diagnostics(report: dict | None) -> dict[str, float | bool]:
    """Compute constraint satisfaction diagnostics beyond valid-SMILES rate."""
    if report is None:
        return {}
    details = report.get("details") or []
    if not isinstance(details, list) or not details:
        return {}

    supported = 0
    satisfied = 0
    ring_requested = False
    ringless_when_ring_requested = False
    for entry in details:
        if not isinstance(entry, dict):
            continue
        if entry.get("supported"):
            supported += 1
        if entry.get("satisfied"):
            satisfied += 1
        constraint = entry.get("constraint") or {}
        if not isinstance(constraint, dict):
            continue
        target_value = _as_number(constraint.get("value"))
        actual_value = _as_number(entry.get("actual"))
        if (
            constraint.get("property") == "ring_count"
            and target_value is not None
            and target_value > 0
        ):
            ring_requested = True
            if actual_value == 0:
                ringless_when_ring_requested = True

    return {
        "constraint_satisfied_fraction": (
            float(satisfied / supported) if supported else 0.0
        ),
        "ring_requested": ring_requested,
        "ringless_when_ring_requested": ringless_when_ring_requested,
    }


def _constraint_smiles_diagnostics(parsed_answer) -> dict[str, float | bool | str]:
    """Return optional RDKit-backed diversity diagnostics for generated SMILES."""
    if not isinstance(parsed_answer, dict):
        return {}
    smiles = parsed_answer.get("smiles")
    if not isinstance(smiles, str) or not smiles:
        return {}
    try:
        from rdkit import Chem
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    canonical = Chem.MolToSmiles(mol, canonical=True)
    ring_count = mol.GetRingInfo().NumRings()
    all_carbon = all(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
    all_single = all(
        bond.GetBondType() == Chem.BondType.SINGLE for bond in mol.GetBonds()
    )
    return {
        "canonical_smiles": canonical,
        "generated_ring_count": float(ring_count),
        "trivial_alkane": all_carbon and ring_count == 0 and all_single,
    }


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
) -> dict[str, float | bool | str]:
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
    diagnostics: dict[str, float | bool | str] = {
        "answer_present": bool(extracted),
        "json_valid": parsed is not None,
        "exact_match": _exact_score(report),
        "partial_score": float(partial_score),
        "valid_smiles": bool(valid_smiles),
    }
    if task_type in {"single_index", "multi_index"}:
        diagnostics.update(_index_diagnostics(report))
    elif task_type == "constraint_generation":
        diagnostics.update(_constraint_diagnostics(report))
        diagnostics.update(_constraint_smiles_diagnostics(parsed))
    return diagnostics
