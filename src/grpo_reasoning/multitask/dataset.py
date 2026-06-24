from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets

from ..common.tasks import get_task

_VALID_STRATEGIES = {"pooled", "balanced", "adaptive"}


def _coerce_properties(value: Any) -> list[str]:
    """Normalize MolecularIQ properties from YAML/JSON into a string list."""
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass(frozen=True)
class MolecularIQTaskSpec:
    """Configuration for one MolecularIQ subtask in a multitask experiment.

    Args:
        task_id: Stable identifier used in datasets, metrics, and plots.
        task_type: MolecularIQ task variant.
        properties: MolecularIQ property names used by the task.
        num_samples: Optional raw rows to generate for this subtask.
        sampling_weight: Optional sampling weight for weighted/adaptive mixes.
        seed: Optional per-task generation seed.
        system_prompt_style: MolecularIQ prompt style.
        constraint_operator: Operator used for constraint-generation tasks.
        filters: Optional dataset filters applied after question generation.
        candidate_multiplier: Extra generation factor used before filtered sampling.
    """

    task_id: str
    task_type: str
    properties: list[str]
    num_samples: int | None = None
    sampling_weight: float = 1.0
    seed: int | None = None
    system_prompt_style: str | None = None
    constraint_operator: str | None = None
    filters: dict[str, Any] | None = None
    candidate_multiplier: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MolecularIQTaskSpec":
        """Build a task spec from a YAML/JSON mapping."""
        if "task_id" not in data:
            raise ValueError(f"Missing task_id in multitask spec: {data}")
        if "task_type" not in data:
            raise ValueError(f"Missing task_type in multitask spec: {data}")
        if "properties" not in data:
            raise ValueError(f"Missing properties in multitask spec: {data}")
        return cls(
            task_id=str(data["task_id"]),
            task_type=str(data["task_type"]),
            properties=_coerce_properties(data["properties"]),
            num_samples=data.get("num_samples"),
            sampling_weight=float(data.get("sampling_weight", 1.0)),
            seed=data.get("seed"),
            system_prompt_style=data.get("system_prompt_style"),
            constraint_operator=data.get("constraint_operator"),
            filters=data.get("filters"),
            candidate_multiplier=int(data.get("candidate_multiplier", 1)),
        )

    def task_kwargs(self, default_seed: int | None = None) -> dict[str, Any]:
        """Return kwargs for constructing the underlying MolecularIQ task."""
        kwargs: dict[str, Any] = {
            "task_type": self.task_type,
            "properties": list(self.properties),
        }
        seed = self.seed if self.seed is not None else default_seed
        if seed is not None:
            kwargs["seed"] = seed
        if self.system_prompt_style is not None:
            kwargs["system_prompt_style"] = self.system_prompt_style
        if self.constraint_operator is not None:
            kwargs["constraint_operator"] = self.constraint_operator
        if self.filters:
            kwargs["filters"] = dict(self.filters)
            kwargs["candidate_multiplier"] = max(1, self.candidate_multiplier)
        return kwargs


@dataclass
class MultitaskDatasetConfig:
    """Configuration for building a multitask MolecularIQ dataset.

    Args:
        out_dir: Directory where the HF dataset is saved.
        tasks: Subtasks to include.
        split: Source split to build.
        strategy: Sampling strategy: pooled, balanced, or adaptive.
        seed: Dataset shuffle/sampling seed.
        total_samples: Optional final number of examples in the mixed dataset.
        samples_per_task: Optional fixed number of examples per task.
        shuffle: Whether to shuffle the final mixed dataset.
        adaptive_metrics_path: Optional per-task metrics JSON used by adaptive sampling.
        adaptive_accuracy_key: Accuracy key to read when metrics are nested.
        adaptive_floor: Lower bound added to error rates before normalization.
        adaptive_temperature: Exponent applied to adaptive error weights.
    """

    out_dir: str
    tasks: list[MolecularIQTaskSpec]
    split: str = "train"
    strategy: str = "pooled"
    seed: int = 42
    total_samples: int | None = None
    samples_per_task: int | None = None
    shuffle: bool = True
    adaptive_metrics_path: str | None = None
    adaptive_accuracy_key: str = "accuracy"
    adaptive_floor: float = 0.05
    adaptive_temperature: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultitaskDatasetConfig":
        """Build a multitask dataset config from a YAML/JSON mapping."""
        tasks = [MolecularIQTaskSpec.from_dict(item) for item in data.get("tasks", [])]
        if not tasks:
            raise ValueError("Multitask config must contain at least one task spec.")

        adaptive = data.get("adaptive", {}) or {}
        cfg = cls(
            out_dir=str(data["out_dir"]),
            tasks=tasks,
            split=str(data.get("split", "train")),
            strategy=str(data.get("strategy", "pooled")),
            seed=int(data.get("seed", 42)),
            total_samples=data.get("total_samples"),
            samples_per_task=data.get("samples_per_task"),
            shuffle=bool(data.get("shuffle", True)),
            adaptive_metrics_path=adaptive.get("metrics_path")
            or data.get("adaptive_metrics_path"),
            adaptive_accuracy_key=str(
                adaptive.get(
                    "accuracy_key",
                    data.get("adaptive_accuracy_key", "accuracy"),
                )
            ),
            adaptive_floor=float(adaptive.get("floor", data.get("adaptive_floor", 0.05))),
            adaptive_temperature=float(
                adaptive.get("temperature", data.get("adaptive_temperature", 1.0))
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate a multitask dataset configuration."""
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown multitask strategy {self.strategy!r}. "
                f"Choose one of {sorted(_VALID_STRATEGIES)}."
            )
        if self.total_samples is not None and self.total_samples <= 0:
            raise ValueError("total_samples must be positive when provided.")
        if self.samples_per_task is not None and self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive when provided.")
        if self.adaptive_floor < 0:
            raise ValueError("adaptive_floor must be non-negative.")
        if self.adaptive_temperature <= 0:
            raise ValueError("adaptive_temperature must be positive.")


def _metadata_columns(spec: MolecularIQTaskSpec, n: int) -> dict[str, list[str]]:
    """Return per-example metadata columns for one task dataset."""
    properties_json = json.dumps(list(spec.properties), sort_keys=True)
    return {
        "task_id": [spec.task_id] * n,
        "task_name": ["moleculariq"] * n,
        "task_type": [spec.task_type] * n,
        "properties": [properties_json] * n,
    }


def build_task_dataset(
    spec: MolecularIQTaskSpec,
    split: str,
    default_seed: int,
) -> Dataset:
    """Generate and annotate one MolecularIQ subtask dataset.

    Difficulty filters (and the matching candidate over-generation) are handled
    by the underlying task via ``task_kwargs``, so training and evaluation share
    one filtering implementation.
    """
    task = get_task("moleculariq", **spec.task_kwargs(default_seed=default_seed))
    ds = task.to_grpo_dataset(split=split, num_samples=spec.num_samples)
    for name, values in _metadata_columns(spec, len(ds)).items():
        if name in ds.column_names:
            ds = ds.remove_columns(name)
        ds = ds.add_column(name, values)
    return ds


def _take_with_replacement(ds: Dataset, n: int, seed: int) -> Dataset:
    """Select exactly n rows from a dataset, sampling with replacement if needed."""
    if n <= 0:
        raise ValueError("Cannot sample a non-positive number of rows.")
    if len(ds) == 0:
        raise ValueError("Cannot sample from an empty task dataset.")
    if n <= len(ds):
        return ds.shuffle(seed=seed).select(range(n))
    rng = random.Random(seed)
    indices = [rng.randrange(len(ds)) for _ in range(n)]
    return ds.select(indices)


def _largest_remainder_counts(
    task_ids: Iterable[str],
    weights: dict[str, float],
    total: int,
) -> dict[str, int]:
    """Allocate integer sample counts proportionally to task weights."""
    ids = list(task_ids)
    if total < len(ids):
        raise ValueError("total_samples must be at least the number of tasks.")
    clean_weights = {
        task_id: max(0.0, float(weights.get(task_id, 0.0))) for task_id in ids
    }
    if sum(clean_weights.values()) <= 0:
        clean_weights = {task_id: 1.0 for task_id in ids}

    weight_sum = sum(clean_weights.values())
    raw = {task_id: total * clean_weights[task_id] / weight_sum for task_id in ids}
    counts = {task_id: max(1, int(math.floor(value))) for task_id, value in raw.items()}

    remainder = total - sum(counts.values())
    order = sorted(
        ids,
        key=lambda task_id: raw[task_id] - math.floor(raw[task_id]),
        reverse=True,
    )
    i = 0
    while remainder > 0:
        counts[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    while remainder < 0:
        candidates = [task_id for task_id in reversed(order) if counts[task_id] > 1]
        if not candidates:
            raise ValueError("Could not allocate positive sample counts.")
        counts[candidates[0]] -= 1
        remainder += 1
    return counts


def _read_task_accuracies(path: str | Path, accuracy_key: str) -> dict[str, float]:
    """Read task accuracies from common multitask evaluation JSON shapes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    accuracies: dict[str, float] = {}

    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        for item in data["tasks"]:
            if not isinstance(item, dict):
                continue
            task_id = item.get("task_id") or item.get("name")
            if task_id is not None and accuracy_key in item:
                accuracies[str(task_id)] = float(item[accuracy_key])

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            task_id = item.get("task_id") or item.get("name")
            if task_id is None:
                continue
            if accuracy_key in item:
                accuracies[str(task_id)] = float(item[accuracy_key])
            elif "trained_accuracy" in item:
                accuracies[str(task_id)] = float(item["trained_accuracy"])

    if isinstance(data, dict) and not accuracies:
        for task_id, value in data.items():
            if isinstance(value, dict) and accuracy_key in value:
                accuracies[str(task_id)] = float(value[accuracy_key])

    return accuracies


def _adaptive_weights(cfg: MultitaskDatasetConfig) -> dict[str, float]:
    """Compute adaptive sampling weights from explicit weights or metrics."""
    base = {spec.task_id: spec.sampling_weight for spec in cfg.tasks}
    if not cfg.adaptive_metrics_path:
        return base
    if not Path(cfg.adaptive_metrics_path).exists():
        print(
            "[multitask] adaptive metrics not found; "
            f"using explicit sampling weights: {cfg.adaptive_metrics_path}"
        )
        return base

    accuracies = _read_task_accuracies(
        cfg.adaptive_metrics_path,
        accuracy_key=cfg.adaptive_accuracy_key,
    )
    weights: dict[str, float] = {}
    for spec in cfg.tasks:
        acc = accuracies.get(spec.task_id)
        if acc is None:
            weights[spec.task_id] = spec.sampling_weight
            continue
        error = max(0.0, 1.0 - float(acc))
        weights[spec.task_id] = (error + cfg.adaptive_floor) ** cfg.adaptive_temperature
    return weights


def _target_counts(
    cfg: MultitaskDatasetConfig,
    datasets: dict[str, Dataset],
) -> dict[str, int]:
    """Determine per-task sample counts for the selected sampling strategy."""
    task_ids = [spec.task_id for spec in cfg.tasks]
    if (
        cfg.strategy == "pooled"
        and cfg.total_samples is None
        and cfg.samples_per_task is None
    ):
        return {task_id: len(datasets[task_id]) for task_id in task_ids}

    if cfg.samples_per_task is not None:
        return {task_id: cfg.samples_per_task for task_id in task_ids}

    if cfg.strategy == "balanced":
        if cfg.total_samples is not None:
            return _largest_remainder_counts(
                task_ids,
                {task_id: 1.0 for task_id in task_ids},
                cfg.total_samples,
            )
        min_len = min(len(datasets[task_id]) for task_id in task_ids)
        return {task_id: min_len for task_id in task_ids}

    if cfg.total_samples is None:
        return {task_id: len(datasets[task_id]) for task_id in task_ids}

    weights = (
        _adaptive_weights(cfg)
        if cfg.strategy == "adaptive"
        else {spec.task_id: spec.sampling_weight for spec in cfg.tasks}
    )
    return _largest_remainder_counts(task_ids, weights, cfg.total_samples)


def build_moleculariq_multitask_dataset(cfg: MultitaskDatasetConfig) -> Dataset:
    """Build a mixed MolecularIQ dataset for pooled/balanced/adaptive training."""
    datasets = {
        spec.task_id: build_task_dataset(spec, split=cfg.split, default_seed=cfg.seed)
        for spec in cfg.tasks
    }
    counts = _target_counts(cfg, datasets)

    parts = []
    for offset, spec in enumerate(cfg.tasks):
        ds = _take_with_replacement(
            datasets[spec.task_id],
            counts[spec.task_id],
            seed=cfg.seed + offset,
        )
        parts.append(ds)

    mixed = concatenate_datasets(parts)
    if cfg.shuffle:
        mixed = mixed.shuffle(seed=cfg.seed)
    return mixed


def build_and_save_multitask(
    cfg: MultitaskDatasetConfig,
    overwrite: bool = False,
) -> Path:
    """Build and save a multitask MolecularIQ dataset."""
    out_dir = Path(cfg.out_dir)
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to rebuild.")

    ds = build_moleculariq_multitask_dataset(cfg)
    config_payload = _dataset_config_payload(cfg)
    manifest = {
        "config_fingerprint": _dataset_config_fingerprint(cfg),
        "config": config_payload,
        "strategy": cfg.strategy,
        "split": cfg.split,
        "seed": cfg.seed,
        "total_rows": len(ds),
        "task_counts": {
            task_id: ds["task_id"].count(task_id)
            for task_id in sorted(set(ds["task_id"]))
        },
        "tasks": [
            {
                "task_id": spec.task_id,
                "task_type": spec.task_type,
                "properties": spec.properties,
                "num_samples": spec.num_samples,
                "sampling_weight": spec.sampling_weight,
                "filters": spec.filters or {},
                "candidate_multiplier": spec.candidate_multiplier,
            }
            for spec in cfg.tasks
        ],
    }
    temp_dir = out_dir.with_name(f"{out_dir.name}.rebuild-{uuid.uuid4().hex[:8]}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(temp_dir))
    (temp_dir / "multitask_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    temp_dir.rename(out_dir)
    return out_dir


def _dataset_config_payload(cfg: MultitaskDatasetConfig) -> dict[str, Any]:
    """Return the training-relevant dataset configuration in canonical form."""
    return {
        "strategy": cfg.strategy,
        "split": cfg.split,
        "seed": cfg.seed,
        "total_samples": cfg.total_samples,
        "samples_per_task": cfg.samples_per_task,
        "shuffle": cfg.shuffle,
        "adaptive_metrics_path": cfg.adaptive_metrics_path,
        "adaptive_accuracy_key": cfg.adaptive_accuracy_key,
        "adaptive_floor": cfg.adaptive_floor,
        "adaptive_temperature": cfg.adaptive_temperature,
        "tasks": [
            {
                "task_id": spec.task_id,
                "task_type": spec.task_type,
                "properties": list(spec.properties),
                "num_samples": spec.num_samples,
                "sampling_weight": spec.sampling_weight,
                "seed": spec.seed,
                "system_prompt_style": spec.system_prompt_style,
                "constraint_operator": spec.constraint_operator,
                "filters": spec.filters or {},
                "candidate_multiplier": spec.candidate_multiplier,
            }
            for spec in cfg.tasks
        ],
    }


def _dataset_config_fingerprint(cfg: MultitaskDatasetConfig) -> str:
    """Hash the effective dataset config to detect stale cached datasets."""
    encoded = json.dumps(
        _dataset_config_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_saved_multitask_dataset(
    path: str | Path,
    cfg: MultitaskDatasetConfig,
) -> None:
    """Reject a cached dataset that was built from a different configuration."""
    manifest_path = Path(path) / "multitask_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"Cached dataset {path} has no multitask_manifest.json. "
            "Rebuild it with --overwrite-datasets."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cached dataset manifest is malformed: {manifest_path}. "
            "Rebuild it with --overwrite-datasets."
        ) from exc
    expected = _dataset_config_fingerprint(cfg)
    actual = manifest.get("config_fingerprint")
    if actual != expected:
        raise ValueError(
            f"Cached dataset {path} does not match the current curriculum config "
            f"(expected fingerprint {expected[:12]}, found "
            f"{str(actual)[:12] or 'missing'}). Rebuild with "
            "--overwrite-datasets before training."
        )
