#!/usr/bin/env python3
"""Build poster-ready figures for the two thesis hypotheses.

This is a thin, self-contained layer on top of the regular reporting outputs.
It reads the SAME ``outputs/multitask_eval/<label>/summary.json`` files that
``grpo-report`` consumes plus the per-run ``trainer_state.json`` logs, and emits
a small, clearly labelled set of figures keyed to:

  H1  Verifier-reward GRPO lifts a 0.5B LLM over its base model.
  H2  Dense (verifier-shaped) rewards are needed for GRPO to learn hard
      set-valued tasks; sparse exact-match rewards stall.

Default labels (override with flags):

  baseline -> the untrained Qwen2.5-0.5B-Instruct reference (auto-evaluated)
  shaped   -> miq_h2_shaped_train.yaml  (dense reward; also the H1 generalist)
  sparse   -> miq_h2_sparse_train.yaml  (exact-match-only reward)

Every panel degrades gracefully: if a summary or a log key is missing, the
panel is skipped with a printed note instead of crashing. That matters when you
are assembling the poster the night before and only some runs have finished.

Usage (run from the repo root, after evaluation has written summaries):

    python scripts/multitask/make_poster_figures.py \
        --eval-dir outputs/multitask_eval \
        --outputs-dir outputs \
        --out-dir outputs/poster
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the project's loaders so this script stays compatible with whatever the
# eval/training pipeline writes.
from grpo_reasoning.multitask.reporting import (
    load_eval_summaries,
    load_training_runs,
)

# Ordered task families and human-readable names. Index/constraint families are
# where the H2 effect is expected to show.
TASK_TYPE_ORDER = [
    "single_count",
    "multi_count",
    "single_index",
    "multi_index",
    "constraint_generation",
]
TASK_TYPE_LABEL = {
    "single_count": "single\ncount",
    "multi_count": "multi\ncount",
    "single_index": "single\nindex",
    "multi_index": "multi\nindex",
    "constraint_generation": "constraint\ngen",
}
HARD_TASK_TYPES = {"single_index", "multi_index", "constraint_generation"}

# Consistent colors across all figures.
COLOR = {
    "baseline": "#9e9e9e",
    "sparse": "#d1495b",
    "shaped": "#2e7d32",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _by_label(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["_label"]: s for s in summaries}


def _task_type_means(summary: dict[str, Any], field: str = "accuracy") -> dict[str, float]:
    """Average ``field`` per task_type for one eval summary."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in summary.get("tasks", []):
        value = row.get(field)
        if value is None:
            continue
        buckets[row.get("task_type", "unknown")].append(float(value))
    return {tt: float(np.mean(vals)) for tt, vals in buckets.items() if vals}


def _series(log_history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Extract (step, value) pairs for a log key, skipping rows without it."""
    steps, values = [], []
    for row in log_history:
        if key in row and row[key] is not None:
            step = row.get("step", row.get("global_step", len(steps)))
            try:
                values.append(float(row[key]))
                steps.append(float(step))
            except (TypeError, ValueError):
                continue
    return steps, values


def _first_present_key(log_history: list[dict[str, Any]], candidates: list[str]) -> str | None:
    """Return the first candidate log key that appears anywhere in the history."""
    present = set()
    for row in log_history:
        present.update(row.keys())
    for key in candidates:
        if key in present:
            return key
    return None


def _runs_by_label(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map a run to a short arm label based on its output directory name."""
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        name = run["label"]
        if "h2-shaped" in name or "shaped" in name:
            out.setdefault("shaped", run)
        elif "h2-sparse" in name or "sparse" in name:
            out.setdefault("sparse", run)
    return out


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[poster] wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# H1: lift over base
# --------------------------------------------------------------------------- #
def fig_h1_lift(by_label: dict[str, dict[str, Any]], args, out_dir: Path) -> Path | None:
    base = by_label.get(args.baseline_label)
    shaped = by_label.get(args.shaped_label)
    if base is None or shaped is None:
        print("[poster] H1 skipped: need both baseline and shaped summaries")
        return None

    fig, (ax_overall, ax_type) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: headline macro + worst-task accuracy.
    metrics = ["macro_accuracy", "worst_task_accuracy"]
    labels = ["macro\naccuracy", "worst-task\naccuracy"]
    x = np.arange(len(metrics))
    w = 0.38
    base_vals = [float(base.get(m, 0.0)) for m in metrics]
    shaped_vals = [float(shaped.get(m, 0.0)) for m in metrics]
    b1 = ax_overall.bar(x - w / 2, base_vals, w, label="base 0.5B", color=COLOR["baseline"])
    b2 = ax_overall.bar(x + w / 2, shaped_vals, w, label="GRPO (verifier reward)", color=COLOR["shaped"])
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(labels)
    ax_overall.set_ylabel("exact-match accuracy")
    ax_overall.set_ylim(0, 1)
    ax_overall.set_title("H1: GRPO lift over the base model")
    ax_overall.legend(frameon=False, fontsize=9)
    for bars in (b1, b2):
        ax_overall.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    # Right: accuracy by task family.
    base_tt = _task_type_means(base)
    shaped_tt = _task_type_means(shaped)
    types = [t for t in TASK_TYPE_ORDER if t in base_tt or t in shaped_tt]
    x = np.arange(len(types))
    ax_type.bar(x - w / 2, [base_tt.get(t, 0.0) for t in types], w,
                label="base 0.5B", color=COLOR["baseline"])
    ax_type.bar(x + w / 2, [shaped_tt.get(t, 0.0) for t in types], w,
                label="GRPO", color=COLOR["shaped"])
    ax_type.set_xticks(x)
    ax_type.set_xticklabels([TASK_TYPE_LABEL.get(t, t) for t in types], fontsize=8)
    ax_type.set_ylim(0, 1)
    ax_type.set_title("Lift by task family")
    ax_type.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return _save(fig, out_dir / "fig_h1_lift.png")


# --------------------------------------------------------------------------- #
# H2: shaped vs sparse, by task family
# --------------------------------------------------------------------------- #
def fig_h2_by_tasktype(by_label: dict[str, dict[str, Any]], args, out_dir: Path) -> Path | None:
    base = by_label.get(args.baseline_label)
    sparse = by_label.get(args.sparse_label)
    shaped = by_label.get(args.shaped_label)
    if sparse is None or shaped is None:
        print("[poster] H2 by-tasktype skipped: need both sparse and shaped summaries")
        return None

    arms = []
    if base is not None:
        arms.append(("base 0.5B", _task_type_means(base), COLOR["baseline"]))
    arms.append(("sparse (exact-match only)", _task_type_means(sparse), COLOR["sparse"]))
    arms.append(("shaped (dense verifier)", _task_type_means(shaped), COLOR["shaped"]))

    types = [t for t in TASK_TYPE_ORDER if any(t in m for _, m, _ in arms)]
    x = np.arange(len(types))
    n = len(arms)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (label, means, color) in enumerate(arms):
        offset = (i - (n - 1) / 2) * w
        vals = [means.get(t, 0.0) for t in types]
        bars = ax.bar(x + offset, vals, w, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    # Shade the "hard" set-valued families where the stall is expected.
    for i, t in enumerate(types):
        if t in HARD_TASK_TYPES:
            ax.axvspan(i - 0.5, i + 0.5, color="#fff3cd", alpha=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_TYPE_LABEL.get(t, t) for t in types])
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("H2: dense reward is needed on hard set-valued tasks (shaded)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir / "fig_h2_by_tasktype.png")


# --------------------------------------------------------------------------- #
# H2: training dynamics (reward + KL)
# --------------------------------------------------------------------------- #
def fig_h2_training(runs_by_label: dict[str, dict[str, Any]], out_dir: Path) -> Path | None:
    sparse = runs_by_label.get("sparse")
    shaped = runs_by_label.get("shaped")
    if sparse is None and shaped is None:
        print("[poster] H2 training skipped: no trainer_state.json found for either arm")
        return None

    fig, (ax_reward, ax_kl) = plt.subplots(1, 2, figsize=(11, 4.4))
    plotted_reward = False
    plotted_kl = False

    for arm, run in (("sparse", sparse), ("shaped", shaped)):
        if run is None:
            continue
        log = run["log_history"]
        reward_key = _first_present_key(log, ["reward", "rewards/mean", "train/reward"])
        if reward_key:
            steps, vals = _series(log, reward_key)
            if steps:
                ax_reward.plot(steps, _smooth(vals), color=COLOR[arm], label=arm, linewidth=2)
                plotted_reward = True
        kl_key = _first_present_key(log, ["kl", "train/kl", "objective/kl"])
        if kl_key:
            steps, vals = _series(log, kl_key)
            if steps:
                ax_kl.plot(steps, _smooth(vals), color=COLOR[arm], label=arm, linewidth=2)
                plotted_kl = True

    ax_reward.set_xlabel("training step")
    ax_reward.set_ylabel("mean group reward")
    ax_reward.set_title("H2: reward trajectory")
    if plotted_reward:
        ax_reward.legend(frameon=False, fontsize=9)
    ax_kl.set_xlabel("training step")
    ax_kl.set_ylabel("KL to reference")
    ax_kl.set_title("Policy movement (KL)")
    if plotted_kl:
        ax_kl.legend(frameon=False, fontsize=9)

    if not (plotted_reward or plotted_kl):
        plt.close(fig)
        print("[poster] H2 training skipped: no reward/kl keys in logs")
        return None

    fig.tight_layout()
    return _save(fig, out_dir / "fig_h2_training.png")


def _smooth(values: list[float], window: int = 15) -> list[float]:
    """Centered moving-average smoothing for noisy per-step RL curves.

    Uses a shrinking window at the edges (via a cumulative-sum difference) so the
    curve does not dip toward zero at the first/last points the way a fixed-width
    'same'-mode convolution would.
    """
    if len(values) < 3:
        return values
    arr = np.asarray(values, dtype=float)
    half = max(1, min(window, len(arr)) // 2)
    cumsum = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return list(out)


# --------------------------------------------------------------------------- #
# H2: exact-match vs partial credit on the shaped model (index tasks)
# --------------------------------------------------------------------------- #
def fig_h2_partial_vs_exact(by_label: dict[str, dict[str, Any]], args, out_dir: Path) -> Path | None:
    shaped = by_label.get(args.shaped_label)
    if shaped is None:
        print("[poster] partial-vs-exact skipped: no shaped summary")
        return None
    rows = [
        r for r in shaped.get("tasks", [])
        if r.get("task_type") in {"single_index", "multi_index"}
        and "partial_score_mean" in r
    ]
    if not rows:
        print("[poster] partial-vs-exact skipped: no index tasks with partial_score_mean")
        return None

    rows.sort(key=lambda r: r["task_id"])
    labels = [r["task_id"] for r in rows]
    exact = [float(r.get("accuracy", 0.0)) for r in rows]
    partial = [float(r.get("partial_score_mean", 0.0)) for r in rows]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4.6))
    ax.bar(x - w / 2, exact, w, label="exact-match accuracy", color="#1565c0")
    ax.bar(x + w / 2, partial, w, label="verifier partial score (Jaccard)", color="#90caf9")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("H2: on index tasks the shaped model is 'mostly right'\n(exact-match understates the verifier's partial credit)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir / "fig_h2_partial_vs_exact.png")


# --------------------------------------------------------------------------- #
# Headline numbers CSV
# --------------------------------------------------------------------------- #
def write_headline_csv(by_label: dict[str, dict[str, Any]], args, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "poster_headline_numbers.csv"
    fields = ["arm", "macro_accuracy", "worst_task_accuracy", "macro_partial_score"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for label in (args.baseline_label, args.sparse_label, args.shaped_label):
            s = by_label.get(label)
            if s is None:
                continue
            writer.writerow({
                "arm": label,
                "macro_accuracy": round(float(s.get("macro_accuracy", 0.0)), 4),
                "worst_task_accuracy": round(float(s.get("worst_task_accuracy", 0.0)), 4),
                "macro_partial_score": round(float(s.get("macro_partial_score", 0.0)), 4),
            })
    print(f"[poster] wrote {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", default="outputs/multitask_eval")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--out-dir", default="outputs/poster")
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--sparse-label", default="sparse")
    p.add_argument("--shaped-label", default="shaped")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    summaries = load_eval_summaries(args.eval_dir)
    by_label = _by_label(summaries)
    print(f"[poster] eval summaries found: {sorted(by_label)}")

    runs = load_training_runs(args.outputs_dir)
    runs_by_label = _runs_by_label(runs)
    print(f"[poster] training runs found: {sorted(runs_by_label)}")

    made = []
    made.append(fig_h1_lift(by_label, args, out_dir))
    made.append(fig_h2_by_tasktype(by_label, args, out_dir))
    made.append(fig_h2_training(runs_by_label, out_dir))
    made.append(fig_h2_partial_vs_exact(by_label, args, out_dir))
    write_headline_csv(by_label, args, out_dir)

    made = [m for m in made if m]
    print(f"\n[poster] done. {len(made)} figure(s) in {out_dir}/")
    for m in made:
        print(f"  - {m}")


if __name__ == "__main__":
    main()
