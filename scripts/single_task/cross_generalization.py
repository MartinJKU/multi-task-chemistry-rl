"""Cross-task generalization matrix for the MolecularIQ GRPO models.

`auto_train_compare.py` trains one model per task configuration and evaluates
each model *only on its own task* (the diagonal of the matrix below).  This
script answers the complementary question:

    Does training on one task also help on the *other* tasks?

It takes every trained model and evaluates it against **every** task
configuration, producing a full (model × eval-task) accuracy matrix.  The
baseline (untrained) model is included as a reference row so off-diagonal
cells can be read as transfer / generalization relative to no training.

Rows are models, columns are evaluation subtasks, and the black-boxed cell in
each trained-model row marks the task that model was trained on.

The diagonal is the in-distribution accuracy (same numbers as
auto_train_compare).  The off-diagonal cells are out-of-distribution: a
positive delta over the baseline row there means training on task A also
improved task B.

Output files
------------
  outputs/generalization/matrix.json        — full results matrix (accuracies)
  outputs/generalization/heatmap.png        — accuracy + transfer-vs-baseline heatmaps
  outputs/generalization/<model>/<task>_eval.json — cached per-cell eval JSON

Usage
-----
    # Full matrix (reuses cached cells; baseline + specialists over the task suite):
    python scripts/single_task/cross_generalization.py

    # Quick smoke test:
    python scripts/single_task/cross_generalization.py --num-samples 50

    # Only some models / tasks:
    python scripts/single_task/cross_generalization.py --models sc_ring_count cg_ring_count
    python scripts/single_task/cross_generalization.py --tasks sc_ring_count si_ring

    # Regenerate the plots/JSON from already-cached eval files:
    python scripts/single_task/cross_generalization.py --report-only

The model and task names are the experiment names defined in
auto_train_compare.py.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNG without needing a display/GUI
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

from grpo_reasoning.common.eval import evaluate  # noqa: E402

# Reuse the exact experiment / task definitions used for training so the
# task_type + properties always match what each model was trained on.
from auto_train_compare import EXPERIMENTS, _BASELINE_MODEL, Experiment  # noqa: E402

_BASELINE_NAME = "baseline"
_OUT_DIR = _REPO_ROOT / "outputs" / "generalization"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _checkpoint_exists(output_dir: Path) -> bool:
    """Check whether a model checkpoint exists.

    Args:
        output_dir: Trainer output directory to inspect.

    Returns:
        True if a final model config or checkpoint directory exists.
    """
    if not output_dir.exists():
        return False
    return (output_dir / "config.json").exists() or any(output_dir.glob("checkpoint-*"))


def _eval_cell(
    model_path: str,
    task: Experiment,
    cache_path: Path,
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
) -> float:
    """Evaluate or load one model-task matrix cell.

    Args:
        model_path: Model name or checkpoint path to evaluate.
        task: Evaluation task definition.
        cache_path: JSON path used to cache the cell result.
        num_samples: Number of evaluation examples to generate.
        batch_size: Evaluation batch size.
        max_new_tokens: Maximum completion tokens to generate.

    Returns:
        Accuracy for the matrix cell in the range [0, 1].
    """
    if cache_path.exists():
        metrics = json.loads(cache_path.read_text())
        metrics = metrics.get("metrics", metrics)
        print(f"[skip] cached {cache_path.relative_to(_REPO_ROOT)} "
              f"-> acc={metrics['accuracy']:.2%}")
        return float(metrics["accuracy"])

    metrics = evaluate(
        model_path=str(model_path),
        task_name="moleculariq",
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        save_path=cache_path,
        task_kwargs={
            "task_type": task.task_type,
            "properties": list(task.properties),
        },
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(metrics["accuracy"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_matrix(
    model_names: list[str],
    task_labels: list[str],
    matrix: np.ndarray,
) -> None:
    """Print an accuracy matrix.

    Args:
        model_names: Row labels.
        task_labels: Column labels.
        matrix: Accuracy matrix in the range [0, 1].

    Returns:
        None.
    """
    col = 26
    cells = "".join(f"{lbl:>16}" for lbl in task_labels)
    row_header = "model \\ eval-task"
    header = f"{row_header:<{col}}{cells}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for i, name in enumerate(model_names):
        row = "".join(f"{matrix[i, j] * 100:>15.1f}%" for j in range(matrix.shape[1]))
        print(f"{name:<{col}}{row}")
    print(sep)
    print("(rows = model trained on that task; baseline = untrained reference)")


def _heatmap(
    ax,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    fmt: str,
    diag_rows: list[int | None],
    vmin: float,
    vmax: float,
    center0: bool = False,
) -> None:
    """Draw one annotated heatmap panel.

    Args:
        ax: Matplotlib axes to draw into.
        data: Matrix values to display.
        row_labels: Heatmap row labels.
        col_labels: Heatmap column labels.
        title: Panel title.
        cmap: Matplotlib colormap name.
        fmt: Format specifier for cell labels.
        diag_rows: Column index of the in-distribution cell for each row.
        vmin: Lower color scale bound.
        vmax: Upper color scale bound.
        center0: Whether the color scale is centered around zero.

    Returns:
        None.
    """
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("evaluation task")
    ax.set_ylabel("model")
    ax.set_title(title)

    span = (vmax - vmin) or 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="0.5")
                continue
            # Pick readable text color based on cell darkness.
            rel = (val - vmin) / span
            dark = rel < 0.35 or rel > 0.85
            color = "white" if dark and not center0 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=9, color=color)
            # Outline the in-distribution diagonal cell for this row.
            if diag_rows[i] == j:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=2.5,
                ))


def _plot(
    model_names: list[str],
    task_labels: list[str],
    matrix: np.ndarray,
    diag_rows: list[int | None],
    baseline_idx: int | None,
    save_path: Path,
) -> None:
    """Plot accuracy and transfer heatmaps.

    Args:
        model_names: Row labels.
        task_labels: Column labels.
        matrix: Accuracy matrix in the range [0, 1].
        diag_rows: Column index of the in-distribution cell for each row.
        baseline_idx: Optional row index for the baseline model.
        save_path: PNG output path.

    Returns:
        None.
    """
    acc = matrix * 100.0
    fig, axes = plt.subplots(
        1, 2 if baseline_idx is not None else 1,
        figsize=(8 * (2 if baseline_idx is not None else 1), 1.2 + 0.7 * len(model_names)),
        dpi=120, squeeze=False,
    )

    _heatmap(
        axes[0, 0], acc, model_names, task_labels,
        "Accuracy (%) — black box = in-distribution (trained on this task)",
        cmap="viridis", fmt=".1f", diag_rows=diag_rows, vmin=0, vmax=100,
    )

    if baseline_idx is not None:
        # Transfer: each trained model's accuracy minus the baseline row.
        delta = acc - acc[baseline_idx][None, :]
        m = float(np.nanmax(np.abs(delta))) if np.any(~np.isnan(delta)) else 1.0
        if not np.isfinite(m) or m == 0:
            m = 1.0
        _heatmap(
            axes[0, 1], delta, model_names, task_labels,
            "Improvement over baseline (pp) — off-diagonal = generalization",
            cmap="RdBu", fmt="+.1f", diag_rows=diag_rows, vmin=-m, vmax=m,
            center0=True,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] heatmap saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run cross-task generalization evaluation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
    """
    available = [e.name for e in EXPERIMENTS]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--num-samples", type=int, default=200,
                   help="Eval samples per cell (default: 200).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--models", nargs="+", metavar="NAME", default=None,
                   help=f"Only evaluate these trained models. Available: {available}")
    p.add_argument("--tasks", nargs="+", metavar="NAME", default=None,
                   help=f"Only evaluate against these tasks. Available: {available}")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip the untrained baseline reference row.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip evaluation; rebuild the matrix/plots from cached JSON.")
    p.add_argument("--out-dir", default=str(_OUT_DIR))
    args = p.parse_args()

    def _select(names: list[str] | None) -> list[Experiment]:
        """Select experiment definitions by name.

        Args:
            names: Optional requested experiment names.

        Returns:
            Matching experiment definitions, or all experiments if `names` is empty.

        Raises:
            SystemExit: If any requested name is unknown.
        """
        if not names:
            return list(EXPERIMENTS)
        unknown = set(names) - set(available)
        if unknown:
            p.error(f"Unknown name(s): {unknown}. Available: {available}")
        return [e for e in EXPERIMENTS if e.name in set(names)]

    model_exps = _select(args.models)
    task_exps = _select(args.tasks)
    out_dir = Path(args.out_dir)

    # Build the list of (display-name, model_path, source-experiment) rows.
    rows: list[tuple[str, str, Experiment | None]] = []
    if not args.no_baseline:
        rows.append((_BASELINE_NAME, _BASELINE_MODEL, None))
    for exp in model_exps:
        if not args.report_only and not _checkpoint_exists(exp.output_dir):
            print(f"[warn] no checkpoint for {exp.name} at {exp.output_dir} — skipping row")
            continue
        rows.append((exp.name, str(exp.output_dir), exp))

    if not rows:
        print("No models to evaluate — nothing to do.")
        return

    task_col_idx = {t.name: j for j, t in enumerate(task_exps)}
    matrix = np.full((len(rows), len(task_exps)), np.nan, dtype=float)
    # For each row, which column (if any) is its in-distribution task.
    diag_rows: list[int | None] = [
        task_col_idx.get(src.name) if src is not None else None
        for _, _, src in rows
    ]

    print(f"\nGeneralization matrix: {len(rows)} models × {len(task_exps)} tasks")
    print(f"  models: {[name for name, _, _ in rows]}")
    print(f"  tasks : {[t.name for t in task_exps]}")

    for i, (name, model_path, _src) in enumerate(rows):
        for j, task in enumerate(task_exps):
            cache_path = out_dir / name / f"{task.name}_eval.json"
            if args.report_only and not cache_path.exists():
                print(f"[warn] missing cached {cache_path.relative_to(_REPO_ROOT)} — leaving NaN")
                continue
            print(f"\n--- cell [{name}] on [{task.name}] "
                  f"({task.task_type} × {task.properties}) ---")
            matrix[i, j] = _eval_cell(
                model_path, task, cache_path,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )

    model_names = [name for name, _, _ in rows]
    task_labels = [t.label.replace("\n", " ") for t in task_exps]

    _print_matrix(model_names, task_labels, matrix)

    # Persist the machine-readable matrix.
    baseline_idx = model_names.index(_BASELINE_NAME) if _BASELINE_NAME in model_names else None
    matrix_data = {
        "models": model_names,
        "tasks": [t.name for t in task_exps],
        "task_labels": task_labels,
        "diagonal_col_per_model": diag_rows,
        "accuracy": [[None if np.isnan(v) else v for v in row] for row in matrix.tolist()],
    }
    if baseline_idx is not None:
        delta = matrix - matrix[baseline_idx][None, :]
        matrix_data["delta_vs_baseline_pp"] = [
            [None if np.isnan(v) else v * 100 for v in row] for row in delta.tolist()
        ]
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_json = out_dir / "matrix.json"
    matrix_json.write_text(json.dumps(matrix_data, indent=2))
    print(f"[matrix] JSON saved to {matrix_json}")

    _plot(model_names, task_labels, matrix, diag_rows, baseline_idx,
          out_dir / "heatmap.png")

    # Quick generalization read-out: average off-diagonal transfer per model.
    if baseline_idx is not None:
        print("\nAverage off-diagonal transfer over baseline (generalization):")
        for i, name in enumerate(model_names):
            if i == baseline_idx:
                continue
            diag = diag_rows[i]
            offs = [j for j in range(matrix.shape[1])
                    if j != diag and not np.isnan(matrix[i, j])]
            if not offs:
                continue
            avg = float(np.mean([(matrix[i, j] - matrix[baseline_idx, j]) for j in offs])) * 100
            print(f"  {name:<22} {avg:+.1f}pp over {len(offs)} other task(s)")


if __name__ == "__main__":
    main()
