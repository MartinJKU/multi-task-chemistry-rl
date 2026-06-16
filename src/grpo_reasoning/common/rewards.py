from __future__ import annotations

import json
import math
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


def _score_moleculariq_completion(
    completion_text: str,
    gold_answer: str,
    task_type: str,
) -> float:
    """Score one MolecularIQ completion with the official evaluator.

    Args:
        completion_text: Full assistant completion text.
        gold_answer: JSON-encoded target or constraints.
        task_type: MolecularIQ task type for this example.

    Returns:
        MolecularIQ score in [0, 1] when available, otherwise 0.
    """
    import json

    from moleculariq_core import evaluate_answer

    extracted = extract_xml_answer(completion_text)
    if not extracted:
        return 0.0
    try:
        target = json.loads(gold_answer) if isinstance(gold_answer, str) else gold_answer
    except json.JSONDecodeError:
        return 0.0

    try:
        if task_type == "constraint_generation":
            return float(
                evaluate_answer(
                    task_type=task_type,
                    predicted=extracted,
                    constraints=target,
                )
            )
        return float(
            evaluate_answer(
                task_type=task_type,
                predicted=extracted,
                target=target,
            )
        )
    except Exception:
        return 0.0


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


def _parse_json(value):
    """Parse JSON-like strings, returning None on malformed inputs."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _parse_extracted_answer(completion_text: str):
    """Extract and parse the JSON object inside <answer> tags."""
    extracted = extract_xml_answer(completion_text)
    if not extracted:
        return None
    return _parse_json(extracted)


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


def _dict_numeric_score(predicted, target) -> float:
    """Average numeric partial credit over target dictionary keys."""
    if not isinstance(predicted, dict) or not isinstance(target, dict) or not target:
        return 0.0
    scores = [
        _numeric_closeness(predicted.get(key), gold_value)
        for key, gold_value in target.items()
    ]
    return sum(scores) / len(scores)


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


def _index_f1(predicted, target) -> float:
    """Score predicted atom-index sets with F1 overlap."""
    pred = _as_int_set(predicted)
    gold = _as_int_set(target)
    if pred is None or gold is None:
        return 0.0
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    return 2.0 * overlap / (len(pred) + len(gold))


def _dict_index_score(predicted, target) -> float:
    """Average atom-index partial credit over target dictionary keys."""
    if not isinstance(predicted, dict) or not isinstance(target, dict) or not target:
        return 0.0
    scores = [_index_f1(predicted.get(key), gold_value) for key, gold_value in target.items()]
    return sum(scores) / len(scores)


def _rdkit_mol_from_smiles(smiles: str):
    """Parse SMILES with RDKit when available."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return None

    RDLogger.DisableLog("rdApp.error")
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _compute_rdkit_property(mol, property_name: str) -> float | None:
    """Compute a small set of MolecularIQ count properties directly with RDKit."""
    if mol is None:
        return None
    try:
        if property_name == "ring_count":
            return float(mol.GetRingInfo().NumRings())
        if property_name == "carbon_atom_count":
            return float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6))
        if property_name == "hetero_atom_count":
            return float(
                sum(
                    1
                    for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() not in {1, 6}
                )
            )
        if property_name == "halogen_atom_count":
            return float(
                sum(
                    1
                    for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() in {9, 17, 35, 53, 85}
                )
            )
        if property_name == "heavy_atom_count":
            return float(mol.GetNumHeavyAtoms())
        if property_name == "aromatic_ring_count":
            rings = mol.GetRingInfo().AtomRings()
            return float(
                sum(
                    1
                    for ring in rings
                    if ring and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
                )
            )
        if property_name == "hba_count":
            from rdkit.Chem import Lipinski

            return float(Lipinski.NumHAcceptors(mol))
        if property_name == "hbd_count":
            from rdkit.Chem import Lipinski

            return float(Lipinski.NumHDonors(mol))
        if property_name == "rotatable_bond_count":
            from rdkit.Chem import Lipinski

            return float(Lipinski.NumRotatableBonds(mol))
    except Exception:
        return None
    return None


def _constraint_satisfaction_score(predicted, target) -> tuple[float, bool]:
    """Score generated SMILES constraints with validity and property closeness."""
    if not isinstance(predicted, dict):
        return 0.0, False
    smiles = predicted.get("smiles")
    mol = _rdkit_mol_from_smiles(smiles)
    if mol is None:
        return 0.0, False

    constraints = target if isinstance(target, list) else [target]
    scores: list[float] = []
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        prop = constraint.get("property")
        op = constraint.get("operator", "=")
        gold = _as_number(constraint.get("value"))
        pred = _compute_rdkit_property(mol, str(prop))
        if gold is None or pred is None or not math.isfinite(pred):
            continue
        if op == "=":
            scores.append(_numeric_closeness(pred, gold))
        elif op == ">=":
            scores.append(1.0 if pred >= gold else _numeric_closeness(pred, gold))
        elif op == ">":
            scores.append(1.0 if pred > gold else _numeric_closeness(pred, gold))
        elif op == "<=":
            scores.append(1.0 if pred <= gold else _numeric_closeness(pred, gold))
        elif op == "<":
            scores.append(1.0 if pred < gold else _numeric_closeness(pred, gold))
        else:
            scores.append(_numeric_closeness(pred, gold))

    if not scores:
        return 0.0, True
    return sum(scores) / len(scores), True


def _moleculariq_shaped_score(
    completion_text: str,
    gold_answer: str,
    task_type: str,
) -> tuple[float, bool]:
    """Return task-shaped partial credit and SMILES validity flag."""
    predicted = _parse_extracted_answer(completion_text)
    target = _parse_json(gold_answer)
    if predicted is None or target is None:
        return 0.0, False

    if task_type in {"single_count", "multi_count"}:
        return _dict_numeric_score(predicted, target), False
    if task_type in {"single_index", "multi_index"}:
        return _dict_index_score(predicted, target), False
    if task_type == "constraint_generation":
        return _constraint_satisfaction_score(predicted, target)
    return 0.0, False


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
            score, valid_smiles = _moleculariq_shaped_score(
                _completion_text(completion),
                gold,
                row_task_type,
            )
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
    """Compute diagnostic partial metrics for one MolecularIQ completion.

    Args:
        completion_text: Full assistant completion text.
        gold_answer: JSON-encoded target answer.
        task_type: MolecularIQ task type for this example.

    Returns:
        Dictionary with answer parsing, partial score, and SMILES validity flags.
    """
    extracted = extract_xml_answer(completion_text)
    parsed = _parse_json(extracted) if extracted else None
    partial_score, valid_smiles = _moleculariq_shaped_score(
        completion_text,
        gold_answer,
        task_type,
    )
    return {
        "answer_present": bool(extracted),
        "json_valid": parsed is not None,
        "partial_score": float(partial_score),
        "valid_smiles": bool(valid_smiles),
    }
