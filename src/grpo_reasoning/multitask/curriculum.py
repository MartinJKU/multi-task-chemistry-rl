from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.train import TrainArgs, train
from ..common.utils import load_yaml
from .dataset import MultitaskDatasetConfig, build_and_save_multitask


@dataclass
class CurriculumStage:
    """One staged multitask training phase.

    Args:
        name: Stage name used for logging.
        dataset: Multitask dataset config dictionary for this stage.
        train_overrides: Overrides merged into the base training config.
    """

    name: str
    dataset: dict[str, Any]
    train_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumStage":
        """Build a curriculum stage from a YAML mapping."""
        if "name" not in data:
            raise ValueError(f"Curriculum stage is missing name: {data}")
        if "dataset" not in data:
            raise ValueError(f"Curriculum stage {data['name']!r} is missing dataset.")
        return cls(
            name=str(data["name"]),
            dataset=dict(data["dataset"]),
            train_overrides=dict(data.get("train", {}) or {}),
        )


@dataclass
class CurriculumConfig:
    """Configuration for staged curriculum training.

    Args:
        base_train_config: YAML training config used as the base for all stages.
        stages: Ordered curriculum stages.
        base_model: Optional model override for the first stage.
        overwrite_datasets: Whether to rebuild existing stage datasets.
    """

    base_train_config: str
    stages: list[CurriculumStage]
    base_model: str | None = None
    overwrite_datasets: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumConfig":
        """Build a curriculum config from YAML data."""
        stages = [CurriculumStage.from_dict(item) for item in data.get("stages", [])]
        if not stages:
            raise ValueError("Curriculum config must define at least one stage.")
        return cls(
            base_train_config=str(data["base_train_config"]),
            stages=stages,
            base_model=data.get("base_model"),
            overwrite_datasets=bool(data.get("overwrite_datasets", False)),
        )


def run_curriculum(
    cfg: CurriculumConfig,
    overwrite_datasets: bool | None = None,
    start_stage: str | None = None,
    dataset_only: bool = False,
    max_steps_per_stage: int | None = None,
) -> str | None:
    """Build datasets and train stages sequentially.

    Args:
        cfg: Curriculum configuration.
        overwrite_datasets: Optional command-line override for dataset rebuilds.
        start_stage: Optional stage name to start from.
        dataset_only: If true, only build stage datasets.
        max_steps_per_stage: Optional training step cap for every stage.

    Returns:
        Final stage output directory, or None when `dataset_only` is true.
    """
    base_train = load_yaml(cfg.base_train_config)
    rebuild = cfg.overwrite_datasets if overwrite_datasets is None else overwrite_datasets

    stages = list(cfg.stages)
    if start_stage is not None:
        names = [stage.name for stage in stages]
        if start_stage not in names:
            raise ValueError(f"Unknown start_stage={start_stage!r}. Available: {names}")
        stages = stages[names.index(start_stage) :]

    previous_output: str | None = None
    final_output: str | None = None
    for index, stage in enumerate(stages):
        print(f"\n[curriculum] stage {index + 1}/{len(stages)}: {stage.name}")
        ds_cfg = MultitaskDatasetConfig.from_dict(stage.dataset)
        out_dir = Path(ds_cfg.out_dir)
        if out_dir.exists() and not rebuild:
            # Datasets were pre-built (e.g. on a login node); reuse them.
            print(f"[curriculum] reusing existing dataset at {out_dir}")
            dataset_path: Path | str = out_dir
        else:
            dataset_path = build_and_save_multitask(ds_cfg, overwrite=rebuild)

        if dataset_only:
            continue

        train_cfg = dict(base_train)
        train_cfg.update(stage.train_overrides)
        train_cfg["dataset_path"] = str(dataset_path)

        if previous_output is not None:
            train_cfg["model_name"] = previous_output
        elif cfg.base_model is not None:
            train_cfg["model_name"] = cfg.base_model

        if max_steps_per_stage is not None:
            train_cfg["max_steps"] = max_steps_per_stage

        if "output_dir" not in train_cfg:
            out_name = stage.name.replace(" ", "_").replace("/", "_")
            train_cfg["output_dir"] = str(Path("outputs") / f"miq-curriculum-{out_name}")

        final_output = train(TrainArgs(**train_cfg))
        previous_output = final_output

    return final_output


def run_curriculum_from_file(
    path: str | Path,
    overwrite_datasets: bool | None = None,
    start_stage: str | None = None,
    dataset_only: bool = False,
    max_steps_per_stage: int | None = None,
) -> str | None:
    """Load and run a curriculum YAML file."""
    cfg = CurriculumConfig.from_dict(load_yaml(path))
    return run_curriculum(
        cfg,
        overwrite_datasets=overwrite_datasets,
        start_stage=start_stage,
        dataset_only=dataset_only,
        max_steps_per_stage=max_steps_per_stage,
    )
