from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from ..common.tasks import get_task


def build_and_save(
    task_name: str,
    out_dir: Path | str,
    split: str = "train",
    num_samples: int | None = None,
    overwrite: bool = False,
    task_kwargs: dict | None = None,
) -> Path:
    """Build and persist a GRPO-ready dataset.

    Args:
        task_name: Registered task name to build.
        out_dir: Directory where the Hugging Face dataset should be saved.
        split: Source split to load from the task.
        num_samples: Optional maximum number of samples to keep.
        overwrite: Whether to rebuild an existing output directory.
        task_kwargs: Optional task-specific keyword arguments.

    Returns:
        Path to the saved dataset directory.

    Raises:
        FileExistsError: If `out_dir` exists and `overwrite` is false.
        KeyError: If `task_name` is not registered.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to rebuild.")

    task = get_task(task_name, **(task_kwargs or {}))
    ds: Dataset = task.to_grpo_dataset(split=split, num_samples=num_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    return out_dir
