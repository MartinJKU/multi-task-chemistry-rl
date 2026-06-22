from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EvalSummary = dict[str, Any]
TrainingRun = dict[str, Any]

_TASK_TABLE_METRICS = (
    "accuracy",
    "partial_score_mean",
    "json_valid_rate",
    "answer_present_rate",
    "valid_smiles_rate",
    "distinct_answer_rate",
    "most_common_answer_rate",
    "index_precision_mean",
    "index_recall_mean",
    "avg_index_pred_len",
    "avg_index_gold_len",
    "avg_index_false_positives",
    "avg_index_false_negatives",
    "empty_gold_nonempty_rate",
    "superset_rate",
    "subset_rate",
    "constraint_satisfied_fraction_mean",
    "ringless_when_ring_requested_rate",
    "canonical_smiles_distinct_rate",
    "trivial_alkane_rate",
)


def create_experiment_report(
    outputs_dir: Path | str = "outputs",
    eval_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """Create comparison plots and tables from training/evaluation outputs.

    Args:
        outputs_dir: Root directory containing trainer output folders.
        eval_dir: Directory containing multitask eval subfolders with summary.json.
        out_dir: Directory where report artifacts should be written.

    Returns:
        A dictionary with generated artifact paths grouped by type.
    """
    outputs_dir = Path(outputs_dir)
    eval_dir = Path(eval_dir) if eval_dir else outputs_dir / "multitask_eval"
    out_dir = Path(out_dir) if out_dir else outputs_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_summaries = load_eval_summaries(eval_dir)
    training_runs = load_training_runs(outputs_dir)

    artifacts: dict[str, list[str]] = {"plots": [], "tables": [], "metadata": []}
    if eval_summaries:
        artifacts["tables"].extend(
            str(path) for path in write_eval_tables(eval_summaries, out_dir)
        )
        artifacts["plots"].extend(
            str(path) for path in plot_eval_comparisons(eval_summaries, out_dir)
        )
    else:
        print(f"[report] no model summary.json files found under {eval_dir}")

    if training_runs:
        artifacts["tables"].extend(
            str(path) for path in write_training_tables(training_runs, out_dir)
        )
        artifacts["plots"].extend(
            str(path) for path in plot_training_comparisons(training_runs, out_dir)
        )
    else:
        print(f"[report] no trainer_state.json files found under {outputs_dir}")

    summary_path = out_dir / "report_summary.json"
    summary_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    artifacts["metadata"].append(str(summary_path))
    print(f"[report] wrote {summary_path}")
    return artifacts


def load_eval_summaries(eval_dir: Path | str) -> list[EvalSummary]:
    """Load model-level multitask evaluation summaries."""
    eval_dir = Path(eval_dir)
    summaries: list[EvalSummary] = []
    for path in sorted(eval_dir.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[report] skip malformed JSON {path}: {exc}")
            continue

        if not _is_model_eval_summary(data):
            print(f"[report] skip non-model summary {path}")
            continue

        data = dict(data)
        data["_summary_path"] = str(path)
        data["_label"] = str(data.get("model_label") or path.parent.name)
        summaries.append(data)

    return sorted(summaries, key=lambda row: row["_label"])


def load_training_runs(outputs_dir: Path | str) -> list[TrainingRun]:
    """Load trainer logs from all output folders with trainer_state.json."""
    outputs_dir = Path(outputs_dir)
    runs: list[TrainingRun] = []
    for directory in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        state_path = _find_trainer_state(directory)
        if state_path is None:
            continue

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[report] skip malformed trainer state {state_path}: {exc}")
            continue

        log_history = state.get("log_history", [])
        if not isinstance(log_history, list):
            continue

        runs.append(
            {
                "label": directory.name,
                "output_dir": str(directory),
                "state_path": str(state_path),
                "global_step": state.get("global_step"),
                "log_history": log_history,
            }
        )

    return runs


def write_eval_tables(summaries: list[EvalSummary], out_dir: Path | str) -> list[Path]:
    """Write CSV tables for overall and per-task evaluation metrics."""
    out_dir = Path(out_dir)
    overall_path = out_dir / "eval_overall.csv"
    task_path = out_dir / "eval_by_task.csv"

    overall_fields = [
        "model",
        "macro_accuracy",
        "worst_task_accuracy",
        "macro_partial_score",
        "model_path",
        "summary_path",
    ]
    with overall_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "model": summary["_label"],
                    "macro_accuracy": _fmt(summary.get("macro_accuracy")),
                    "worst_task_accuracy": _fmt(summary.get("worst_task_accuracy")),
                    "macro_partial_score": _fmt(summary.get("macro_partial_score")),
                    "model_path": summary.get("model_path", ""),
                    "summary_path": summary.get("_summary_path", ""),
                }
            )

    labels = [summary["_label"] for summary in summaries]
    task_fields = ["task_id", "task_type", "properties"]
    for label in labels:
        task_fields.extend(f"{label}_{metric}" for metric in _TASK_TABLE_METRICS)
    task_fields.extend(["winner", "best_accuracy", "spread_pp"])

    task_rows = _task_rows(summaries)
    with task_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=task_fields)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({field: row.get(field, "") for field in task_fields})

    print(f"[report] wrote {overall_path}")
    print(f"[report] wrote {task_path}")
    return [overall_path, task_path]


def write_training_tables(runs: list[TrainingRun], out_dir: Path | str) -> list[Path]:
    """Write a compact CSV with final logged training metrics per run."""
    out_dir = Path(out_dir)
    path = out_dir / "training_final_metrics.csv"
    fields = [
        "run",
        "global_step",
        "last_logged_step",
        "loss",
        "reward",
        "kl",
        "learning_rate",
        "state_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            last = _last_metric_row(run["log_history"])
            writer.writerow(
                {
                    "run": run["label"],
                    "global_step": _fmt(run.get("global_step")),
                    "last_logged_step": _fmt(last.get("step")),
                    "loss": _fmt(last.get("loss")),
                    "reward": _fmt(last.get("reward")),
                    "kl": _fmt(last.get("kl")),
                    "learning_rate": _fmt(last.get("learning_rate")),
                    "state_path": run["state_path"],
                }
            )

    print(f"[report] wrote {path}")
    return [path]


def plot_eval_comparisons(
    summaries: list[EvalSummary],
    out_dir: Path | str,
) -> list[Path]:
    """Create evaluation comparison plots."""
    out_dir = Path(out_dir)
    paths = [
        _plot_overall_metrics(summaries, out_dir),
        _plot_task_heatmap(summaries, out_dir, "accuracy", "per_task_accuracy_heatmap.png"),
        _plot_task_wins(summaries, out_dir),
        _plot_task_type_accuracy(summaries, out_dir),
    ]

    if any(
        "partial_score_mean" in task
        for summary in summaries
        for task in summary.get("tasks", [])
    ):
        paths.append(
            _plot_task_heatmap(
                summaries,
                out_dir,
                "partial_score_mean",
                "per_task_partial_score_heatmap.png",
            )
        )

    return paths


def plot_training_comparisons(
    runs: list[TrainingRun],
    out_dir: Path | str,
) -> list[Path]:
    """Create overlaid training plots for all discovered runs."""
    out_dir = Path(out_dir)
    paths: list[Path] = []
    for key, title, ylabel, filename in (
        ("loss", "Training Loss", "loss", "training_loss.png"),
        ("reward", "Training Reward", "reward", "training_reward.png"),
        ("kl", "KL to Reference", "KL", "training_kl.png"),
    ):
        if any(_series(run["log_history"], key)[0] for run in runs):
            paths.append(_plot_training_series(runs, out_dir, key, title, ylabel, filename))

    component_keys = _reward_component_keys(runs)
    if component_keys:
        paths.append(_plot_reward_components(runs, out_dir, component_keys))

    return paths


def _is_model_eval_summary(data: object) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("tasks"), list)
        and "model_label" in data
        and "macro_accuracy" in data
    )


def _find_trainer_state(output_dir: Path) -> Path | None:
    direct = output_dir / "trainer_state.json"
    if direct.exists():
        return direct

    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=_checkpoint_sort_key,
    )
    for checkpoint in reversed(checkpoints):
        state = checkpoint / "trainer_state.json"
        if state.exists():
            return state
    return None


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _task_rows(summaries: list[EvalSummary]) -> list[dict[str, Any]]:
    labels = [summary["_label"] for summary in summaries]
    by_task: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for summary in summaries:
        label = summary["_label"]
        for task in summary.get("tasks", []):
            task_id = task.get("task_id")
            if not task_id:
                continue
            if task_id not in by_task:
                order.append(task_id)
                by_task[task_id] = {
                    "task_id": task_id,
                    "task_type": task.get("task_type", ""),
                    "properties": ",".join(task.get("properties", [])),
                }
            for metric in _TASK_TABLE_METRICS:
                by_task[task_id][f"{label}_{metric}"] = _fmt(task.get(metric))

    rows = []
    for task_id in order:
        row = by_task[task_id]
        accuracies = {
            label: _to_float(row.get(f"{label}_accuracy")) for label in labels
        }
        valid = {label: value for label, value in accuracies.items() if value is not None}
        if valid:
            best = max(valid.values())
            winners = [label for label, value in valid.items() if value == best]
            row["winner"] = "+".join(winners)
            row["best_accuracy"] = _fmt(best)
            row["spread_pp"] = _fmt((max(valid.values()) - min(valid.values())) * 100.0)
        rows.append(row)

    return rows


def _plot_overall_metrics(summaries: list[EvalSummary], out_dir: Path) -> Path:
    labels = [summary["_label"] for summary in summaries]
    metric_names = ["macro_accuracy", "worst_task_accuracy"]
    if any("macro_partial_score" in summary for summary in summaries):
        metric_names.append("macro_partial_score")

    values = np.array(
        [
            [_nan(summary.get(metric)) * 100.0 for summary in summaries]
            for metric in metric_names
        ]
    )
    x = np.arange(len(labels))
    width = min(0.8 / len(metric_names), 0.25)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    offsets = (np.arange(len(metric_names)) - (len(metric_names) - 1) / 2) * width
    for index, metric in enumerate(metric_names):
        bars = ax.bar(
            x + offsets[index],
            values[index],
            width=width,
            label=_pretty_metric(metric),
        )
        _annotate_bars(ax, bars)

    ax.set_title("Overall Multitask Evaluation")
    ax.set_ylabel("score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(100.0, float(np.nanmax(values)) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "overall_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _plot_task_heatmap(
    summaries: list[EvalSummary],
    out_dir: Path,
    metric: str,
    filename: str,
) -> Path:
    labels = [summary["_label"] for summary in summaries]
    task_ids = _ordered_task_ids(summaries)
    matrix = np.full((len(task_ids), len(labels)), np.nan)

    task_lookup = {
        (summary["_label"], task.get("task_id")): task
        for summary in summaries
        for task in summary.get("tasks", [])
    }
    for row, task_id in enumerate(task_ids):
        for col, label in enumerate(labels):
            matrix[row, col] = _nan(task_lookup.get((label, task_id), {}).get(metric))

    matrix_percent = matrix * 100.0
    fig, ax = plt.subplots(
        figsize=(max(7, len(labels) * 1.6), max(6, len(task_ids) * 0.38))
    )
    image = ax.imshow(matrix_percent, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_title(_pretty_metric(metric) + " by Task")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(task_ids)))
    ax.set_yticklabels(task_ids)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("score (%)")

    for row in range(matrix_percent.shape[0]):
        for col in range(matrix_percent.shape[1]):
            value = matrix_percent[row, col]
            text = "-" if math.isnan(value) else f"{value:.1f}"
            color = "white" if not math.isnan(value) and value < 45 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _plot_task_wins(summaries: list[EvalSummary], out_dir: Path) -> Path:
    labels = [summary["_label"] for summary in summaries]
    wins = Counter()
    for row in _task_rows(summaries):
        winner = row.get("winner", "")
        if not winner:
            continue
        for label in winner.split("+"):
            wins[label] += 1 / len(winner.split("+"))

    values = [wins.get(label, 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4.5))
    bars = ax.bar(labels, values, color="#4c78a8")
    _annotate_bars(ax, bars, fmt="{:.1f}")
    ax.set_title("Per-Task Wins")
    ax.set_ylabel("winning tasks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = out_dir / "task_wins.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _plot_task_type_accuracy(summaries: list[EvalSummary], out_dir: Path) -> Path:
    labels = [summary["_label"] for summary in summaries]
    task_types = sorted(
        {
            str(task.get("task_type"))
            for summary in summaries
            for task in summary.get("tasks", [])
            if task.get("task_type")
        }
    )
    values = np.full((len(task_types), len(labels)), np.nan)
    for row, task_type in enumerate(task_types):
        for col, summary in enumerate(summaries):
            scores = [
                _nan(task.get("accuracy"))
                for task in summary.get("tasks", [])
                if task.get("task_type") == task_type
            ]
            scores = [score for score in scores if not math.isnan(score)]
            if scores:
                values[row, col] = float(np.mean(scores)) * 100.0

    x = np.arange(len(task_types))
    width = min(0.8 / max(len(labels), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8, len(task_types) * 1.3), 5))
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width
    for col, label in enumerate(labels):
        ax.bar(x + offsets[col], values[:, col], width=width, label=label)

    ax.set_title("Accuracy by Task Type")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, rotation=20, ha="right")
    ax.set_ylim(0, max(100.0, float(np.nanmax(values)) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "task_type_accuracy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _plot_training_series(
    runs: list[TrainingRun],
    out_dir: Path,
    key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for run in runs:
        steps, values = _series(run["log_history"], key)
        if not steps:
            continue
        ax.plot(steps, values, label=run["label"], linewidth=1.8)
        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError(f"No training series found for {key}")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = out_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _plot_reward_components(
    runs: list[TrainingRun],
    out_dir: Path,
    component_keys: list[str],
) -> Path:
    cols = min(2, len(component_keys))
    rows = int(math.ceil(len(component_keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 3.2)), squeeze=False)

    for axis, key in zip(axes.ravel(), component_keys):
        for run in runs:
            steps, values = _series(run["log_history"], key)
            if steps:
                axis.plot(steps, values, label=run["label"], linewidth=1.5)
        axis.set_title(key.replace("rewards/", ""))
        axis.set_xlabel("step")
        axis.set_ylabel("reward")
        axis.grid(alpha=0.25)

    for axis in axes.ravel()[len(component_keys) :]:
        axis.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Reward Components", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path = out_dir / "training_reward_components.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[report] wrote {path}")
    return path


def _ordered_task_ids(summaries: list[EvalSummary]) -> list[str]:
    seen = set()
    task_ids = []
    for summary in summaries:
        for task in summary.get("tasks", []):
            task_id = task.get("task_id")
            if task_id and task_id not in seen:
                seen.add(task_id)
                task_ids.append(task_id)
    return task_ids


def _last_metric_row(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(log_history):
        if any(key in row for key in ("loss", "reward", "kl", "learning_rate")):
            return row
    return {}


def _series(log_history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    steps = []
    values = []
    for row in log_history:
        if key not in row or "step" not in row:
            continue
        value = _to_float(row.get(key))
        step = _to_float(row.get("step"))
        if value is None or step is None:
            continue
        steps.append(step)
        values.append(value)
    return steps, values


def _reward_component_keys(runs: list[TrainingRun]) -> list[str]:
    counts = Counter()
    for run in runs:
        for row in run["log_history"]:
            for key in row:
                if key.startswith("rewards/") and not key.endswith("/std"):
                    counts[key] += 1

    priority_terms = ("correct", "moleculariq", "shaped", "format", "valid")

    def sort_key(key: str) -> tuple[int, int, str]:
        priority = min(
            (idx for idx, term in enumerate(priority_terms) if term in key.lower()),
            default=len(priority_terms),
        )
        return priority, -counts[key], key

    return sorted(counts, key=sort_key)[:8]


def _annotate_bars(ax: plt.Axes, bars: Any, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _pretty_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def _fmt(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"{number:.8g}"


def _nan(value: Any) -> float:
    number = _to_float(value)
    return float("nan") if number is None else number


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
