"""
SMILES pool: https://huggingface.co/datasets/ml-jku/moleculariq-trainPool
Q&A generation: https://github.com/ml-jku/moleculariq-core
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset

from ..prompts import extract_xml_answer
from .base import Task, register_task

_SUPPORTED_TASK_TYPES = {
    "single_count",
    "multi_count",
    "single_index",
    "multi_index",
    "constraint_generation",
}

# Benzene (c1ccccc1) as the demo molecule
_FEW_SHOT_EXAMPLES: dict[str, tuple[str, str]] = {
    "single_count": (
        'How many rings does the molecule "c1ccccc1" have? Provide the result as'
        ' JSON with the exact key "ring_count".',
        "<reasoning>Benzene (c1ccccc1) consists of one 6-membered aromatic ring,"
        " so ring_count = 1.</reasoning>\n"
        '<answer>{"ring_count": 1}</answer>',
    ),
    "multi_count": (
        'For the molecule "c1ccccc1", report ring_count and aromatic_ring_count as JSON.',
        "<reasoning>Benzene has 1 ring total and 1 aromatic ring.</reasoning>\n"
        '<answer>{"aromatic_ring_count": 1, "ring_count": 1}</answer>',
    ),
    "single_index": (
        'Report the ring_index for the molecule "c1ccccc1". Return JSON with key'
        ' "ring_index" containing a list of 0-based atom indices (H excluded).',
        "<reasoning>Benzene is a 6-membered ring whose 6 heavy atoms have indices"
        " 0 through 5 in SMILES order.</reasoning>\n"
        '<answer>{"ring_index": [0, 1, 2, 3, 4, 5]}</answer>',
    ),
    "multi_index": (
        'For "c1ccccc1", report ring_index and aromatic_ring_index as JSON lists'
        " of 0-based atom indices.",
        "<reasoning>All 6 benzene atoms (0-5) are in the ring and all are"
        " aromatic.</reasoning>\n"
        '<answer>{"aromatic_ring_index": [0, 1, 2, 3, 4, 5],'
        ' "ring_index": [0, 1, 2, 3, 4, 5]}</answer>',
    ),
    "constraint_generation": (
        'Generate a molecule where ring_count = 1. Return JSON with key "smiles"'
        " containing a valid SMILES string.",
        "<reasoning>I need any molecule with exactly 1 ring. Benzene (c1ccccc1)"
        " satisfies ring_count = 1.</reasoning>\n"
        '<answer>{"smiles": "c1ccccc1"}</answer>',
    ),
}


def _flatten_answer_values(value) -> list:
    """Flatten nested JSON answer values for simple filter checks."""
    if isinstance(value, dict):
        out: list = []
        for item in value.values():
            out.extend(_flatten_answer_values(item))
        return out
    if isinstance(value, list):
        out = [value]
        for item in value:
            out.extend(_flatten_answer_values(item))
        return out
    return [value]


def answer_passes_filters(answer: str, filters: dict) -> bool:
    """Check JSON-answer constraints used to keep tasks tractable.

    Args:
        answer: JSON-encoded gold answer.
        filters: Mapping of supported filter names to thresholds.

    Returns:
        True when the answer satisfies every configured answer filter.
    """
    try:
        parsed = json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        return False

    values = _flatten_answer_values(parsed)
    lists = [value for value in values if isinstance(value, list)]
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    max_list_len = filters.get("max_answer_list_length")
    if max_list_len is not None and any(len(items) > int(max_list_len) for items in lists):
        return False

    min_list_len = filters.get("min_answer_list_length")
    if min_list_len is not None and lists and any(len(items) < int(min_list_len) for items in lists):
        return False

    if filters.get("require_nonempty_list") and lists and not any(lists):
        return False

    max_numeric = filters.get("max_answer_numeric_value")
    if max_numeric is not None and any(abs(value) > float(max_numeric) for value in numbers):
        return False

    return True


def row_passes_filters(smiles: str, question: str, answer: str, filters: dict) -> bool:
    """Check whether one generated example satisfies the task filters."""
    max_smiles_len = filters.get("max_smiles_length")
    if max_smiles_len is not None and len(smiles or "") > int(max_smiles_len):
        return False

    max_question_len = filters.get("max_question_length")
    if max_question_len is not None and len(question or "") > int(max_question_len):
        return False

    return answer_passes_filters(answer or "", filters)


def _format_instructions(task_type: str, system_prompt_style: str) -> str:
    """Return task-specific answer-format instructions.

    Args:
        task_type: MolecularIQ task variant.
        system_prompt_style: Prompt style selected for MolecularIQD.

    Returns:
        Instruction string describing the JSON shape inside `<answer>`.

    Raises:
        ValueError: If `task_type` is unsupported.
    """
    indexing = (
        "Atom indices are 0-based, in SMILES order, hydrogens excluded but "
        "isotopes ([2H], [3H]) included."
    )
    if task_type in {"single_count", "multi_count"}:
        return (
            "Put a single JSON object inside <answer>...</answer> with the EXACT "
            "key(s) named in the question and integer counts as values "
            '(0 if absent). Example: <answer>{"ring_count": 3}</answer>.'
        )
    if task_type in {"single_index", "multi_index"}:
        return (
            f"{indexing} Put a single JSON object inside <answer>...</answer> "
            "with the EXACT key(s) named in the question, each mapped to a list "
            "of atom indices (empty list [] if absent). "
            'Example: <answer>{"ring_index": [0, 1, 2]}</answer>.'
        )
    if task_type == "constraint_generation":
        return (
            "Return a single JSON object inside <answer>...</answer> with key "
            "'smiles' and a valid SMILES string. "
            'Example: <answer>{"smiles": "CCO"}</answer>.'
        )
    raise ValueError(f"Unsupported moleculariq task_type: {task_type}")


@dataclass
class MolecularIQTask(Task):
    """MolecularIQ chemistry reasoning task.

    Args:
        name: Registered task name.
        task_instructions: Additional system prompt instructions.
        task_type: MolecularIQ task variant.
        properties: Molecular properties to ask about.
        constraint_operator: Operator used for constraint generation tasks.
        seed: Random seed for raw-data shuffling and MolecularIQD.
        system_prompt_style: MolecularIQD system prompt style.
        filters: Optional difficulty filters applied to generated examples in
            both training and evaluation (e.g. max_answer_list_length).
        candidate_multiplier: Over-generation factor used before filtering so
            enough examples survive to reach num_samples.
        _repo: Hugging Face dataset repository for the SMILES pool.
    """

    name: str = "moleculariq"
    task_instructions: str = ""

    task_type: str = "single_count"
    properties: list[str] = field(default_factory=lambda: ["ring_count"])
    constraint_operator: str = "="

    seed: int = 42
    system_prompt_style: str = "with_key_hints"

    filters: dict | None = None
    candidate_multiplier: int = 1

    _repo: str = "ml-jku/moleculariq-trainPool"

    def __post_init__(self) -> None:
        """Validate task settings and fill derived prompt defaults.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If task type or property count is invalid.
        """
        if self.task_type not in _SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type={self.task_type!r}. "
                f"Pick one of {sorted(_SUPPORTED_TASK_TYPES)}."
            )
        if self.task_type.startswith("single_") and len(self.properties) != 1:
            raise ValueError(
                f"task_type={self.task_type} requires exactly 1 property, got {self.properties}"
            )
        if not self.task_instructions:
            self.task_instructions = _format_instructions(
                self.task_type, self.system_prompt_style
            )

        if self.few_shot_question is None and self.task_type in _FEW_SHOT_EXAMPLES:
            self.few_shot_question, self.few_shot_answer = _FEW_SHOT_EXAMPLES[
                self.task_type
            ]

    def load_raw(self, split: str):
        """Load the MolecularIQ SMILES pool.

        Args:
            split: Requested split name; currently ignored because the pool has one split.

        Returns:
            Hugging Face dataset containing raw SMILES rows.
        """
        return load_dataset(self._repo, split="train")

    def extract_question(self, row: dict) -> str:
        """Extract a placeholder question from a raw MolecularIQ row.

        Args:
            row: Raw MolecularIQ row.

        Returns:
            SMILES string from the row.
        """
        return row.get("smiles", "")

    def extract_answer(self, row: dict) -> str:
        """Extract a placeholder answer from a raw MolecularIQ row.

        Args:
            row: Raw MolecularIQ row.

        Returns:
            Empty string because generated MolecularIQ answers are produced later.
        """
        return ""

    def _make_generator(self):
        """Instantiate the MolecularIQ question generator.

        Args:
            None.

        Returns:
            Configured `moleculariq_core.MolecularIQD` instance.

        Raises:
            ImportError: If `moleculariq_core` is not installed.
        """
        try:
            from moleculariq_core import MolecularIQD
        except ImportError as e:
            raise ImportError(
                "moleculariq_core is required for the moleculariq task. Install with:\n"
                "  pip install git+https://github.com/ml-jku/moleculariq-core.git"
            ) from e
        return MolecularIQD(
            seed=self.seed,
            enable_random_phrasing=True,
            cache_properties=True,
            system_prompt_style=self.system_prompt_style,
        )

    def _generate_qa(self, mqd, smiles: str) -> tuple[str, str] | None:
        """Generate one MolecularIQ question-answer pair.

        Args:
            mqd: MolecularIQD generator instance.
            smiles: Input molecule as a SMILES string.

        Returns:
            Tuple of question and JSON-encoded answer, or `None` if generation fails.
        """
        try:
            if self.task_type in {"single_count", "multi_count"}:
                question, gt, _ = mqd.generate_count_question(
                    smiles=smiles, count_properties=self.properties
                )
            elif self.task_type in {"single_index", "multi_index"}:
                question, gt, _ = mqd.generate_index_question(
                    smiles=smiles, index_properties=self.properties
                )
            elif self.task_type == "constraint_generation":
                prop = self.properties[0]
                value = mqd.compute_property(smiles, prop)

                if isinstance(value, list):
                    value = len(value)
                if not isinstance(value, (int, float)):
                    return None
                constraints = [
                    {
                        "property": prop,
                        "operator": self.constraint_operator,
                        "value": value,
                    }
                ]
                question, _ = mqd.generate_constraint_question(constraints=constraints)
                gt = constraints
            else:
                raise ValueError(self.task_type)
        except Exception:
            return None

        try:
            answer_json = json.dumps(gt, sort_keys=True)
        except (TypeError, ValueError):
            return None
        return question, answer_json

    def to_grpo_dataset(
        self, split: str = "train", num_samples: int | None = None
    ) -> Dataset:
        """Generate a GRPO dataset for MolecularIQ.

        Args:
            split: Logical split name; `test` selects from the tail of the shuffled pool.
            num_samples: Optional maximum number of raw rows to process.

        Returns:
            Hugging Face dataset with prompts, generated answers, questions, and SMILES.
        """
        raw = self.load_raw(split)
        raw = raw.shuffle(seed=self.seed)
        if split == "test":
            tail_size = min(2000, len(raw))
            raw = raw.select(range(len(raw) - tail_size, len(raw)))

        # When filters are active, over-generate a candidate pool so enough
        # examples survive filtering to reach num_samples. The same filtering is
        # applied here for both train and eval, keeping their difficulty aligned.
        candidate_samples = num_samples
        if num_samples is not None and self.filters:
            candidate_samples = num_samples * max(1, self.candidate_multiplier)
        if candidate_samples is not None:
            raw = raw.select(range(min(candidate_samples, len(raw))))

        mqd = self._make_generator()

        questions: list[str] = []
        answers: list[str] = []
        smiles_kept: list[str] = []
        n_skipped = 0
        n_filtered = 0
        for row in raw:
            smi = row.get("smiles") or ""
            if not smi:
                continue
            qa = self._generate_qa(mqd, smi)
            if qa is None:
                n_skipped += 1
                continue
            q, a = qa
            if self.filters and not row_passes_filters(smi, q, a, self.filters):
                n_filtered += 1
                continue
            questions.append(q)
            answers.append(a)
            smiles_kept.append(smi)
            if num_samples is not None and len(questions) >= num_samples:
                break

        if n_skipped or n_filtered:
            print(
                f"[dataset] {n_skipped} SMILES skipped (generation failed), "
                f"{n_filtered} removed by filters; kept {len(questions)}"
            )
        if self.filters and not questions:
            raise ValueError(
                f"Filters {self.filters} removed all generated examples for "
                f"task_type={self.task_type!r}; relax the filters or raise "
                "candidate_multiplier."
            )
        prompts = [self.build_prompt(q) for q in questions]
        return Dataset.from_dict(
            {
                "prompt": prompts,
                "answer": answers,
                "question": questions,
                "smiles": smiles_kept,
            }
        )

    def score_prediction(self, prediction_text: str, gold_answer: str) -> bool:
        """Score a MolecularIQ prediction with the official evaluator.

        Args:
            prediction_text: Full generated completion text.
            gold_answer: JSON-encoded target answer or constraint specification.

        Returns:
            True when `moleculariq_core.evaluate_answer` reports a full match.
        """
        from moleculariq_core import evaluate_answer

        try:
            target = (
                json.loads(gold_answer) if isinstance(gold_answer, str) else gold_answer
            )
        except json.JSONDecodeError:
            return False

        extracted = extract_xml_answer(prediction_text)
        if not extracted:
            return False

        # Require a well-formed JSON object: every task prompt asks for one, and
        # the verifier's lenient parser would otherwise accept malformed answers
        # (e.g. a list missing its closing bracket) as correct.
        try:
            if not isinstance(json.loads(extracted), dict):
                return False
        except (TypeError, json.JSONDecodeError):
            return False

        try:
            if self.task_type == "constraint_generation":
                score = evaluate_answer(
                    task_type=self.task_type,
                    predicted=extracted,
                    constraints=target,
                )
            else:
                score = evaluate_answer(
                    task_type=self.task_type,
                    predicted=extracted,
                    target=target,
                )
        except Exception:
            return False
        return float(score) >= 1.0


register_task("moleculariq")(MolecularIQTask)
