from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from datasets import Dataset

from ..prompts import build_chat_prompt, extract_xml_answer


@dataclass
class Task(ABC):
    """Abstract base class for a single GRPO task.

    Args:
        name: Registered task name.
        task_instructions: Additional system prompt instructions.
        few_shot_question: Optional example user question (legacy single-shot).
        few_shot_answer: Optional example assistant answer (legacy single-shot).
        few_shot_examples: Optional list of (question, answer) pairs for
            multi-shot prompting; takes precedence over the single-shot pair.
    """

    name: str
    task_instructions: str = ""
    few_shot_question: str | None = None
    few_shot_answer: str | None = None
    few_shot_examples: list[tuple[str, str]] | None = None

    @abstractmethod
    def load_raw(self, split: str):
        """Load raw task data.

        Args:
            split: Dataset split name.

        Returns:
            Raw dataset object for the requested split.
        """
        ...

    @abstractmethod
    def extract_question(self, row: dict) -> str:
        """Extract the user question from a raw row.

        Args:
            row: Raw dataset row.

        Returns:
            Question string for prompt construction.
        """
        ...

    @abstractmethod
    def extract_answer(self, row: dict) -> str:
        """Extract the gold answer from a raw row.

        Args:
            row: Raw dataset row.

        Returns:
            Gold answer string.
        """
        ...

    def build_prompt(self, question: str) -> list[dict]:
        """Build a chat prompt for one task question.

        Args:
            question: Question text to present to the model.

        Returns:
            Chat-format prompt messages.
        """
        return build_chat_prompt(
            question=question,
            task_instructions=self.task_instructions,
            few_shot_question=self.few_shot_question,
            few_shot_answer=self.few_shot_answer,
            few_shot_examples=self.few_shot_examples,
        )

    def to_grpo_dataset(
        self, split: str = "train", num_samples: int | None = None
    ) -> Dataset:
        """Convert raw task data into a GRPO dataset.

        Args:
            split: Source split name.
            num_samples: Optional maximum number of rows to include.

        Returns:
            Hugging Face dataset with `prompt`, `answer`, and `question` columns.
        """
        raw = self.load_raw(split)
        if num_samples is not None:
            raw = raw.select(range(min(num_samples, len(raw))))

        cols = raw.column_names

        def _map(batch):
            """Map a raw batch to prompt, answer, and question columns.

            Args:
                batch: Column-oriented batch from Hugging Face `Dataset.map`.

            Returns:
                Dictionary of mapped columns.
            """
            n = len(next(iter(batch.values())))
            rows = [{c: batch[c][i] for c in cols} for i in range(n)]
            questions = [self.extract_question(r) for r in rows]
            answers = [self.extract_answer(r) for r in rows]
            prompts = [self.build_prompt(q) for q in questions]
            return {"prompt": prompts, "answer": answers, "question": questions}

        keep_cols = ["prompt", "answer", "question"]
        mapped = raw.map(_map, batched=True, batch_size=500)
        drop = [c for c in mapped.column_names if c not in keep_cols]
        return mapped.remove_columns(drop)

    def score_prediction(self, prediction_text: str, gold_answer: str) -> bool:
        """Score a model prediction against a gold answer.

        Args:
            prediction_text: Full generated completion text.
            gold_answer: Gold answer string.

        Returns:
            True when the extracted answer exactly matches the gold answer.
        """
        extracted = extract_xml_answer(prediction_text)
        return extracted == gold_answer


_REGISTRY: dict[str, Callable[..., Task]] = {}


def register_task(name: str):
    """Create a decorator that registers a task factory.

    Args:
        name: Registry name for the task.

    Returns:
        Decorator that stores a task factory under `name`.
    """
    def deco(factory: Callable[..., Task]):
        """Register one task factory.

        Args:
            factory: Callable that creates a `Task`.

        Returns:
            The original factory, unchanged.
        """
        _REGISTRY[name] = factory
        return factory

    return deco


def get_task(name: str, **kwargs) -> Task:
    """Instantiate a registered task.

    Args:
        name: Registered task name.
        **kwargs: Keyword arguments forwarded to the task constructor.

    Returns:
        Initialized task instance.

    Raises:
        KeyError: If `name` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_tasks() -> list[str]:
    """List registered task names.

    Args:
        None.

    Returns:
        Sorted task-name list.
    """
    return sorted(_REGISTRY)
