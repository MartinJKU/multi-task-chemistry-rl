"""Plot pass@k curves to separate elicitation from expansion.

Reads one or more pass@k summary JSON files (written by
``grpo_reasoning.common.eval.evaluate_pass_at_k``) and draws pass@k-vs-k curves,
one line per model, faceted by task. The figure is the headline artifact for the
"does GRPO teach or just elicit?" question:

* In-distribution tasks: if the base model's curve crosses *above* a fine-tuned
  model at large k, RL sharpened rather than expanded capability.
* Held-out tasks: if a multitask model's curve stays *above* the base even at
  large k, transfer reflects genuine new capability rather than elicitation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _task_label(summary: dict[str, Any]) -> str:
    """Build a stable, human-readable task label from a summary."""
    task_type = summary.get("task_type") or summary.get("task") or "task"
    props = summary.get("properties") or []
    return f"{task_type}:{'+'.join(props)}" if props else str(task_type)


def load_passk_summaries(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load pass@k summary JSON files, skipping unreadable ones."""
    summaries: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[passk-report] skipping {path}: {exc}")
            continue
        if "pass_at_k" not in data:
            print(f"[passk-report] skipping {path}: no pass_at_k field")
            continue
        data.setdefault("_source", str(path))
        summaries.append(data)
    return summaries


def find_crossover(
    base: dict[str, Any],
    other: dict[str, Any],
) -> int | None:
    """Return the smallest k where ``base`` pass@k meets or exceeds ``other``.

    Args:
        base: pass@k summary for the reference (e.g. base model).
        other: pass@k summary for the comparison model (e.g. GRPO model).

    Returns:
        The crossover k, or None if the base never catches up within shared k.
    """
    shared = sorted(
        set(int(k) for k in base["pass_at_k"]) & set(int(k) for k in other["pass_at_k"])
    )
    for k in shared:
        if base["pass_at_k"][str(k)]["mean"] >= other["pass_at_k"][str(k)]["mean"]:
            return k
    return None


def _curve(summary: dict[str, Any]) -> tuple[list[int], list[float], list[float], list[float]]:
    """Extract sorted (k, mean, ci_low, ci_high) arrays from a summary."""
    ks = sorted(int(k) for k in summary["pass_at_k"])
    mean = [summary["pass_at_k"][str(k)]["mean"] for k in ks]
    low = [summary["pass_at_k"][str(k)].get("ci_low", m) for k, m in zip(ks, mean)]
    high = [summary["pass_at_k"][str(k)].get("ci_high", m) for k, m in zip(ks, mean)]
    return ks, mean, low, high


def plot_pass_at_k(
    summaries: list[dict[str, Any]],
    out_dir: str | Path,
    base_label: str = "base",
    stem: str = "pass_at_k",
    title: str | None = None,
) -> list[Path]:
    """Draw faceted pass@k curves (one subplot per task, one line per model).

    Args:
        summaries: Loaded pass@k summaries from ``load_passk_summaries``.
        out_dir: Directory to write the figure(s) and crossover table.
        base_label: ``model_label`` treated as the base/reference model.
        stem: Output filename stem.
        title: Optional overall figure title.

    Returns:
        Paths of the written artifacts (PNG, PDF, crossover CSV).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        print("[passk-report] no summaries to plot")
        return []

    tasks: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        tasks.setdefault(_task_label(summary), []).append(summary)
    task_labels = sorted(tasks)

    ncols = min(3, len(task_labels))
    nrows = (len(task_labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    crossovers: list[dict[str, Any]] = []
    for ax_index, task_label in enumerate(task_labels):
        ax = axes[ax_index // ncols][ax_index % ncols]
        task_summaries = tasks[task_label]
        base = next(
            (s for s in task_summaries if s.get("model_label") == base_label), None
        )
        for summary in sorted(task_summaries, key=lambda s: s.get("model_label", "")):
            ks, mean, low, high = _curve(summary)
            label = summary.get("model_label", "model")
            line = ax.plot(ks, mean, marker="o", label=label)[0]
            ax.fill_between(ks, low, high, alpha=0.15, color=line.get_color())
            if base is not None and summary is not base:
                k_cross = find_crossover(base, summary)
                shared = sorted(
                    set(int(x) for x in base["pass_at_k"])
                    & set(int(x) for x in summary["pass_at_k"])
                )
                kmin = shared[0] if shared else None
                leads_early = (
                    kmin is not None
                    and summary["pass_at_k"][str(kmin)]["mean"]
                    > base["pass_at_k"][str(kmin)]["mean"]
                )
                if k_cross is None:
                    interpretation = "expansion (grpo above base at all k)"
                elif leads_early:
                    interpretation = f"elicitation (grpo leads early, base catches up by k={k_cross})"
                else:
                    interpretation = "regression (base >= grpo across all k)"
                crossovers.append(
                    {
                        "task": task_label,
                        "model": label,
                        "base": base_label,
                        "pass_at_1_model": summary["pass_at_k"].get("1", {}).get("mean"),
                        "pass_at_1_base": base["pass_at_k"].get("1", {}).get("mean"),
                        "crossover_k": k_cross,
                        "interpretation": interpretation,
                    }
                )
                if k_cross is not None:
                    ax.axvline(k_cross, color=line.get_color(), ls=":", alpha=0.5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("k  =  attempts allowed per question  (log₂)")
        ax.set_ylabel("pass@k  =  fraction solved within k tries")
        ax.set_ylim(0, 1)
        ax.set_title(task_label, fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8, title="model", loc="upper left")

    for ax_index in range(len(task_labels), nrows * ncols):
        axes[ax_index // ncols][ax_index % ncols].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    csv_path = out_dir / f"{stem}_crossover.csv"
    _write_crossover_csv(crossovers, csv_path)
    written.append(csv_path)

    for path in written:
        print(f"[passk-report] wrote {path}")
    return written


def _write_crossover_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the per-(task, model) crossover summary table."""
    import csv

    fields = [
        "task",
        "model",
        "base",
        "pass_at_1_base",
        "pass_at_1_model",
        "crossover_k",
        "interpretation",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
