from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def check_index_gate(
    summary_path: str | Path,
    min_macro_accuracy: float = 0.01,
    min_macro_partial: float = 0.25,
    min_mean_distinct: float = 0.10,
    max_task_mode: float = 0.80,
    min_mean_empty_accuracy: float = 0.05,
) -> dict[str, float]:
    """Validate that an index warm-start is not an answer-template collapse."""
    path = Path(summary_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [
        row
        for row in data.get("tasks", [])
        if row.get("task_type") in {"single_index", "multi_index"}
    ]
    if not tasks:
        raise ValueError(f"No index tasks found in {path}.")

    def mean(key: str, rows: list[dict[str, Any]] = tasks) -> float:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            raise ValueError(f"Index gate metric {key!r} is missing from {path}.")
        return sum(values) / len(values)

    empty_rows = [row for row in tasks if "index_empty_gold_accuracy" in row]
    metrics = {
        "macro_accuracy": mean("accuracy"),
        "macro_partial": mean("partial_score_mean"),
        "mean_distinct": mean("distinct_answer_rate"),
        "worst_mode": max(float(row.get("most_common_answer_rate", 1.0)) for row in tasks),
        "mean_empty_accuracy": mean("index_empty_gold_accuracy", empty_rows),
    }
    failures = []
    if metrics["macro_accuracy"] < min_macro_accuracy:
        failures.append(
            f"macro_accuracy={metrics['macro_accuracy']:.3f} < {min_macro_accuracy:.3f}"
        )
    if metrics["macro_partial"] < min_macro_partial:
        failures.append(
            f"macro_partial={metrics['macro_partial']:.3f} < {min_macro_partial:.3f}"
        )
    if metrics["mean_distinct"] < min_mean_distinct:
        failures.append(
            f"mean_distinct={metrics['mean_distinct']:.3f} < {min_mean_distinct:.3f}"
        )
    if metrics["worst_mode"] > max_task_mode:
        failures.append(
            f"worst_mode={metrics['worst_mode']:.3f} > {max_task_mode:.3f}"
        )
    if metrics["mean_empty_accuracy"] < min_mean_empty_accuracy:
        failures.append(
            "mean_empty_accuracy="
            f"{metrics['mean_empty_accuracy']:.3f} < {min_mean_empty_accuracy:.3f}"
        )
    if failures:
        raise ValueError("Index warm-start gate failed: " + "; ".join(failures))
    return metrics


def index_gate_main() -> None:
    """CLI entry point for the index warm-start quality gate."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--min-macro-accuracy", type=float, default=0.01)
    parser.add_argument("--min-macro-partial", type=float, default=0.25)
    parser.add_argument("--min-mean-distinct", type=float, default=0.10)
    parser.add_argument("--max-task-mode", type=float, default=0.80)
    parser.add_argument("--min-mean-empty-accuracy", type=float, default=0.05)
    args = parser.parse_args()

    metrics = check_index_gate(
        args.summary,
        min_macro_accuracy=args.min_macro_accuracy,
        min_macro_partial=args.min_macro_partial,
        min_mean_distinct=args.min_mean_distinct,
        max_task_mode=args.max_task_mode,
        min_mean_empty_accuracy=args.min_mean_empty_accuracy,
    )
    formatted = ", ".join(f"{key}={value:.3f}" for key, value in metrics.items())
    print(f"[index-gate] passed: {formatted}")

