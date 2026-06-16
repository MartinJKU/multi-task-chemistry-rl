from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ..common.tasks import get_task
from ..common.utils import load_yaml
from .dataset import MolecularIQTaskSpec


@dataclass
class PropertyStats:
    """Simple target-value summary for one property."""

    seen: int = 0
    nonzero: int = 0
    empty: int = 0
    numeric_sum: float = 0.0
    numeric_count: int = 0
    list_len_sum: int = 0
    list_count: int = 0

    def update(self, value: Any) -> None:
        self.seen += 1
        if isinstance(value, list):
            self.list_count += 1
            self.list_len_sum += len(value)
            if len(value) == 0:
                self.empty += 1
            else:
                self.nonzero += 1
            return

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.numeric_count += 1
            self.numeric_sum += float(value)
            if float(value) == 0.0:
                self.empty += 1
            else:
                self.nonzero += 1
            return

        if value in (None, "", []):
            self.empty += 1
        else:
            self.nonzero += 1

    def to_dict(self) -> dict[str, Any]:
        mean_numeric = (
            self.numeric_sum / self.numeric_count if self.numeric_count else None
        )
        mean_list_len = self.list_len_sum / self.list_count if self.list_count else None
        return {
            "seen": self.seen,
            "nonzero": self.nonzero,
            "empty_or_zero": self.empty,
            "mean_numeric_value": mean_numeric,
            "mean_list_length": mean_list_len,
        }


@dataclass
class TaskAudit:
    """Audit counters for one configured MolecularIQ task."""

    task_id: str
    task_type: str
    properties: list[str]
    generated: int = 0
    skipped: int = 0
    property_stats: dict[str, PropertyStats] = field(default_factory=dict)

    def as_row(self, raw_rows: int) -> dict[str, Any]:
        attempted = self.generated + self.skipped
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "properties": ",".join(self.properties),
            "raw_rows": raw_rows,
            "generated": self.generated,
            "skipped": self.skipped,
            "success_rate": self.generated / attempted if attempted else 0.0,
            "pooled_current_cap": min(self.generated, 1500),
        }


def audit_moleculariq_dataset(
    config_path: Path | str = "configs/multitask/miq_experiment_suite.yaml",
    out_dir: Path | str = "outputs/moleculariq_dataset_audit",
    split: str = "train",
    repo: str = "ml-jku/moleculariq-trainPool",
    seed: int = 42,
    max_raw_rows: int | None = None,
    progress_every: int = 5000,
) -> dict[str, Any]:
    """Audit full MolecularIQ source-pool coverage for configured tasks.

    Args:
        config_path: YAML containing a `tasks` list.
        out_dir: Directory where JSON/CSV outputs are written.
        split: MolecularIQ source split. The public pool currently uses `train`.
        repo: Hugging Face dataset id for the MolecularIQ SMILES pool.
        seed: Default MolecularIQ generator seed for task specs without a seed.
        max_raw_rows: Optional raw-row cap for smoke tests.
        progress_every: Print progress every N raw rows per task.

    Returns:
        The JSON-serializable audit summary.
    """
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_task_specs(config_path)
    raw = load_dataset(repo, split=split)
    if max_raw_rows is not None:
        raw = raw.select(range(min(max_raw_rows, len(raw))))

    smiles = [row.get("smiles") or "" for row in raw]
    nonempty_smiles = [smi for smi in smiles if smi]
    print(
        f"[audit] loaded {len(raw)} raw rows from {repo}:{split} "
        f"({len(nonempty_smiles)} non-empty SMILES)"
    )
    print(f"[audit] auditing {len(specs)} task specs from {config_path}")

    mqd_cache: dict[tuple[int, str], Any] = {}
    task_audits: list[TaskAudit] = []
    for spec_index, spec in enumerate(specs, start=1):
        task = get_task("moleculariq", **spec.task_kwargs(default_seed=seed))
        prompt_style = spec.system_prompt_style or getattr(
            task, "system_prompt_style", "with_key_hints"
        )
        task_seed = spec.seed if spec.seed is not None else seed
        cache_key = (int(task_seed), str(prompt_style))
        if cache_key not in mqd_cache:
            mqd_cache[cache_key] = task._make_generator()
        mqd = mqd_cache[cache_key]

        audit = TaskAudit(
            task_id=spec.task_id,
            task_type=spec.task_type,
            properties=list(spec.properties),
            property_stats={prop: PropertyStats() for prop in spec.properties},
        )
        print(
            f"[audit] {spec_index}/{len(specs)} {spec.task_id} "
            f"({spec.task_type}: {','.join(spec.properties)})"
        )

        for row_index, smi in enumerate(smiles, start=1):
            if not smi:
                audit.skipped += 1
                continue

            qa = task._generate_qa(mqd, smi)
            if qa is None:
                audit.skipped += 1
                continue

            audit.generated += 1
            _, answer = qa
            _update_property_stats(audit, answer, spec.task_type)

            if progress_every > 0 and row_index % progress_every == 0:
                print(
                    f"[audit]   {row_index}/{len(smiles)} rows, "
                    f"generated={audit.generated}, skipped={audit.skipped}"
                )

        task_audits.append(audit)

    summary = _build_summary(
        task_audits=task_audits,
        config_path=config_path,
        repo=repo,
        split=split,
        raw_rows=len(raw),
        nonempty_smiles=len(nonempty_smiles),
        unique_smiles=len(set(nonempty_smiles)),
        max_raw_rows=max_raw_rows,
    )
    _write_outputs(summary, out_dir)
    return summary


def _load_task_specs(config_path: Path) -> list[MolecularIQTaskSpec]:
    data = load_yaml(config_path)
    tasks = data.get("tasks", [])
    if not tasks:
        raise ValueError(f"No tasks found in {config_path}")
    return [MolecularIQTaskSpec.from_dict(item) for item in tasks]


def _update_property_stats(
    audit: TaskAudit,
    answer_json: str,
    task_type: str,
) -> None:
    try:
        parsed = json.loads(answer_json)
    except json.JSONDecodeError:
        return

    if task_type == "constraint_generation" and isinstance(parsed, list):
        for constraint in parsed:
            if not isinstance(constraint, dict):
                continue
            prop = constraint.get("property")
            if prop in audit.property_stats:
                audit.property_stats[prop].update(constraint.get("value"))
        return

    if isinstance(parsed, dict):
        for prop, stats in audit.property_stats.items():
            if prop in parsed:
                stats.update(parsed[prop])


def _build_summary(
    task_audits: list[TaskAudit],
    config_path: Path,
    repo: str,
    split: str,
    raw_rows: int,
    nonempty_smiles: int,
    unique_smiles: int,
    max_raw_rows: int | None,
) -> dict[str, Any]:
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_property: dict[str, Counter[str]] = defaultdict(Counter)
    task_rows = []
    property_rows = []

    for audit in task_audits:
        row = audit.as_row(raw_rows)
        task_rows.append(row)
        by_type[audit.task_type]["tasks"] += 1
        by_type[audit.task_type]["generated"] += audit.generated
        by_type[audit.task_type]["skipped"] += audit.skipped

        for prop, stats in audit.property_stats.items():
            stats_dict = stats.to_dict()
            property_rows.append(
                {
                    "task_id": audit.task_id,
                    "task_type": audit.task_type,
                    "property": prop,
                    **stats_dict,
                }
            )
            by_property[prop]["task_specs"] += 1
            by_property[prop]["seen"] += stats.seen
            by_property[prop]["nonzero"] += stats.nonzero
            by_property[prop]["empty_or_zero"] += stats.empty

    return {
        "config_path": str(config_path),
        "repo": repo,
        "split": split,
        "max_raw_rows": max_raw_rows,
        "raw_rows": raw_rows,
        "nonempty_smiles": nonempty_smiles,
        "unique_smiles": unique_smiles,
        "task_count": len(task_rows),
        "total_generated_across_task_specs": sum(row["generated"] for row in task_rows),
        "tasks": task_rows,
        "by_task_type": [
            {"task_type": task_type, **dict(counter)}
            for task_type, counter in sorted(by_type.items())
        ],
        "by_property": [
            {"property": prop, **dict(counter)}
            for prop, counter in sorted(by_property.items())
        ],
        "property_stats": property_rows,
    }


def _write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[audit] wrote {summary_path}")

    _write_csv(out_dir / "task_counts.csv", summary["tasks"])
    _write_csv(out_dir / "task_type_counts.csv", summary["by_task_type"])
    _write_csv(out_dir / "property_counts.csv", summary["by_property"])
    _write_csv(out_dir / "property_stats.csv", summary["property_stats"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[audit] wrote {path}")
