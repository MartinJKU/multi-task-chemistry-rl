"""Automated multi-experiment GRPO training + comparison for MolecularIQ.

Each experiment is a (task_type, properties) combination.  For each one the
script will:
  1. Preprocess training data (skipped if the data directory already exists)
  2. Write a per-experiment YAML config (derived from moleculariq_qwen05b.yaml)
  3. Train a GRPO model          (skipped if a checkpoint already exists)
  4. Evaluate baseline vs trained (skipped if both eval JSONs already exist)
  5. Print a summary table and write a comparison chart

Output files
------------
  outputs/comparison/summary.json          — machine-readable results table
  outputs/comparison/summary.png           — side-by-side accuracy + delta chart
  outputs/comparison/<name>/               — per-experiment eval JSONs + chart

Usage
-----
    # Full run (each experiment ~30-60 min on a consumer GPU):
    python scripts/single_task/auto_train_compare.py

    # Quick smoke-test (50 train steps, 100 eval samples):
    python scripts/single_task/auto_train_compare.py --max-steps 50 --num-eval 100 --num-train 500

    # Regenerate report from already-completed eval JSONs (no training):
    python scripts/single_task/auto_train_compare.py --report-only

    # Run only specific experiments by name:
    python scripts/single_task/auto_train_compare.py --experiments sc_ring_count cg_ring_count

Available experiment names (defined in EXPERIMENTS below):
    sc_ring_count      single_count   x ring_count
    sc_aromatic_ring   single_count   x aromatic_ring_count
    sc_fused_ring      single_count   x fused_ring_count
    sc_carbon_atom     single_count   x carbon_atom_count
    sc_hetero_atom     single_count   x hetero_atom_count
    sc_hba             single_count   x hba_count
    sc_rotatable_bond  single_count   x rotatable_bond_count
    mc_topology        multi_count    x ring/aromatic/fused ring counts
    mc_composition     multi_count    x carbon/hetero/halogen atom counts
    si_ring            single_index   x ring_index
    si_aromatic_ring   single_index   x aromatic_ring_index
    si_carbon_atom     single_index   x carbon_atom_index
    si_hetero_atom     single_index   x hetero_atom_index
    mi_ring_aromatic   multi_index    x ring/aromatic ring indices
    cg_ring_count      constraint_generation x ring_count
    cg_carbon_atom     constraint_generation x carbon_atom_count
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # write PNG without needing a display/GUI
import matplotlib.pyplot as plt
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_SINGLE_TASK_SCRIPTS = _SCRIPTS_DIR / "single_task"
_PYTHON = sys.executable
_BASE_CONFIG = _REPO_ROOT / "configs" / "single_task" / "moleculariq_qwen05b.yaml"
_BASELINE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Experiment definitions — edit this list to add / remove combinations
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """Configuration for one MolecularIQ training/evaluation experiment.

    Args:
        name: Short slug used in directory and file names.
        label: Human-readable label used in charts.
        task_type: MolecularIQ task variant.
        properties: MolecularIQ property names to ask about.
    """

    name: str            # short slug used in dir/file names
    label: str           # human-readable label for charts (may contain \n)
    task_type: str       # single_count | single_index | constraint_generation | ...
    properties: list[str] = field(default_factory=list)

    # Derived paths (set in __post_init__)
    data_dir: Path = field(init=False)
    config_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Derive all filesystem paths from the experiment slug.

        Args:
            None.

        Returns:
            None.
        """
        self.data_dir    = _REPO_ROOT / "data"    / f"miq_{self.name}_train"
        self.config_path = _REPO_ROOT / "configs" / "single_task" / f"miq_{self.name}.yaml"
        self.output_dir  = _REPO_ROOT / "outputs" / f"miq-{self.name}-grpo"
        self.eval_dir    = _REPO_ROOT / "outputs" / "comparison" / self.name


EXPERIMENTS: list[Experiment] = [
    # ---- single_count variants ----
    Experiment(
        name="sc_ring_count",
        label="single_count\n× ring_count",
        task_type="single_count",
        properties=["ring_count"],
    ),
    Experiment(
        name="sc_aromatic_ring",
        label="single_count\n× aromatic_ring_count",
        task_type="single_count",
        properties=["aromatic_ring_count"],
    ),
    Experiment(
        name="sc_fused_ring",
        label="single_count\n× fused_ring_count",
        task_type="single_count",
        properties=["fused_ring_count"],
    ),
    Experiment(
        name="sc_carbon_atom",
        label="single_count\n× carbon_atom_count",
        task_type="single_count",
        properties=["carbon_atom_count"],
    ),
    Experiment(
        name="sc_hetero_atom",
        label="single_count\n× hetero_atom_count",
        task_type="single_count",
        properties=["hetero_atom_count"],
    ),
    Experiment(
        name="sc_hba",
        label="single_count\n× hba_count",
        task_type="single_count",
        properties=["hba_count"],
    ),
    Experiment(
        name="sc_rotatable_bond",
        label="single_count\n× rotatable_bond_count",
        task_type="single_count",
        properties=["rotatable_bond_count"],
    ),
    # ---- multi_count variants ----
    Experiment(
        name="mc_topology",
        label="multi_count\n× topology",
        task_type="multi_count",
        properties=["ring_count", "aromatic_ring_count", "fused_ring_count"],
    ),
    Experiment(
        name="mc_composition",
        label="multi_count\n× composition",
        task_type="multi_count",
        properties=["carbon_atom_count", "hetero_atom_count", "halogen_atom_count"],
    ),
    # ---- single_index variants ----
    # Property names for index tasks come from moleculariq_core.
    Experiment(
        name="si_ring",
        label="single_index\n× ring_index",
        task_type="single_index",
        properties=["ring_index"],
    ),
    Experiment(
        name="si_aromatic_ring",
        label="single_index\n× aromatic_ring_index",
        task_type="single_index",
        properties=["aromatic_ring_index"],
    ),
    Experiment(
        name="si_carbon_atom",
        label="single_index\n× carbon_atom_index",
        task_type="single_index",
        properties=["carbon_atom_index"],
    ),
    Experiment(
        name="si_hetero_atom",
        label="single_index\n× hetero_atom_index",
        task_type="single_index",
        properties=["hetero_atom_index"],
    ),
    # ---- multi_index variants ----
    Experiment(
        name="mi_ring_aromatic",
        label="multi_index\n× ring/aromatic",
        task_type="multi_index",
        properties=["ring_index", "aromatic_ring_index"],
    ),
    # ---- constraint_generation ----
    Experiment(
        name="cg_ring_count",
        label="constraint_generation\n× ring_count",
        task_type="constraint_generation",
        properties=["ring_count"],
    ),
    Experiment(
        name="cg_carbon_atom",
        label="constraint_generation\n× carbon_atom_count",
        task_type="constraint_generation",
        properties=["carbon_atom_count"],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], desc: str) -> None:
    """Run a subprocess command.

    Args:
        cmd: Command arguments to execute.
        desc: Human-readable step description to print.

    Returns:
        None.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
    """
    print(f"\n{'=' * 64}")
    print(f"  {desc}")
    print(f"{'=' * 64}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=_REPO_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed (exit {result.returncode}):\n  {' '.join(str(c) for c in cmd)}"
        )
    print(f"  Completed in {elapsed:.0f}s")


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


def _eval_results_exist(eval_dir: Path) -> bool:
    """Check whether both baseline and trained eval files exist.

    Args:
        eval_dir: Evaluation output directory to inspect.

    Returns:
        True when both expected evaluation JSON files are present.
    """
    return (
        (eval_dir / "baseline_eval.json").exists()
        and (eval_dir / "trained_eval.json").exists()
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_preprocess(exp: Experiment, num_train: int) -> None:
    """Preprocess training data for one experiment.

    Args:
        exp: Experiment definition.
        num_train: Number of training examples to generate.

    Returns:
        None.
    """
    if exp.data_dir.exists():
        print(f"[skip] preprocess {exp.name} — {exp.data_dir} already exists")
        return
    _run(
        [
            _PYTHON, _SINGLE_TASK_SCRIPTS / "preprocess.py",
            "--task",        "moleculariq",
            "--split",       "train",
            "--task-type",   exp.task_type,
            "--properties",  *exp.properties,
            "--num-samples", str(num_train),
            "--out",         exp.data_dir,
        ],
        f"Preprocess [{exp.name}]  task_type={exp.task_type}  properties={exp.properties}",
    )


def step_write_config(
    exp: Experiment, base_cfg: dict, max_steps: Optional[int]
) -> None:
    """Write a per-experiment YAML config.

    Args:
        exp: Experiment definition.
        base_cfg: Base YAML configuration.
        max_steps: Optional training step override.

    Returns:
        None.
    """
    cfg = dict(base_cfg)
    # Always use forward slashes so the YAML is readable on any OS.
    cfg["dataset_path"]          = exp.data_dir.as_posix()
    cfg["output_dir"]            = exp.output_dir.as_posix()
    cfg["moleculariq_task_type"] = exp.task_type
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    exp.config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(exp.config_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)
    print(f"[config] wrote {exp.config_path}")


def step_train(exp: Experiment) -> None:
    """Train one experiment if no checkpoint exists.

    Args:
        exp: Experiment definition.

    Returns:
        None.
    """
    if _checkpoint_exists(exp.output_dir):
        print(f"[skip] train {exp.name} — checkpoint exists at {exp.output_dir}")
        return
    _run(
        [_PYTHON, _SCRIPTS_DIR / "train.py", "--config", exp.config_path],
        f"Train [{exp.name}]  task_type={exp.task_type}  properties={exp.properties}",
    )
    # Note: train.py is the shared entry point at scripts/train.py.


def step_evaluate(exp: Experiment, num_eval: int) -> dict:
    """Evaluate baseline and trained models for one experiment.

    Args:
        exp: Experiment definition.
        num_eval: Number of evaluation examples to generate.

    Returns:
        Dictionary containing baseline and trained metrics.
    """
    exp.eval_dir.mkdir(parents=True, exist_ok=True)

    if _eval_results_exist(exp.eval_dir):
        print(f"[skip] eval {exp.name} — results already in {exp.eval_dir}")
    else:
        _run(
            [
                _PYTHON, _SINGLE_TASK_SCRIPTS / "evaluate.py",
                "--task",        "moleculariq",
                "--task-type",   exp.task_type,
                "--properties",  *exp.properties,
                "--num-samples", str(num_eval),
                "--baseline",    _BASELINE_MODEL,
                "--trained",     exp.output_dir,
                "--out-dir",     exp.eval_dir,
                "--figure",      exp.eval_dir / "baseline_vs_trained.png",
            ],
            f"Evaluate [{exp.name}]  baseline vs trained",
        )

    def _load(path: Path) -> dict:
        """Load metrics from one evaluation JSON file.

        Args:
            path: Evaluation JSON path.

        Returns:
            Metrics dictionary, unwrapped from the `metrics` field when present.
        """
        data = json.loads(path.read_text())
        # eval.py saves {"metrics": {...}, "results": [...]}; unwrap the metrics.
        return data.get("metrics", data)

    return {
        "baseline": _load(exp.eval_dir / "baseline_eval.json"),
        "trained":  _load(exp.eval_dir / "trained_eval.json"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[tuple[Experiment, dict]]) -> None:
    """Print a comparison table for experiment results.

    Args:
        results: Experiment and metrics pairs.

    Returns:
        None.
    """
    col = 46
    header = f"{'Experiment':<{col}} {'Baseline':>9} {'Trained':>9} {'Delta':>9}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for exp, r in results:
        b = r["baseline"]["accuracy"] * 100
        t = r["trained"]["accuracy"] * 100
        label = exp.label.replace("\n", " ")
        print(f"{label:<{col}} {b:>8.1f}% {t:>8.1f}% {t - b:>+8.1f}pp")
    print(sep)


def _plot_summary(results: list[tuple[Experiment, dict]], save_path: Path) -> None:
    """Plot the multi-experiment comparison summary.

    Args:
        results: Experiment and metrics pairs.
        save_path: PNG output path.

    Returns:
        None.
    """
    n = len(results)
    labels      = [exp.label for exp, _ in results]
    baselines   = [r["baseline"]["accuracy"] * 100 for _, r in results]
    trained_acc = [r["trained"]["accuracy"] * 100 for _, r in results]
    deltas      = [t - b for b, t in zip(baselines, trained_acc)]

    x     = list(range(n))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 2.5), 10), dpi=120)

    # Accuracy bar chart
    bars_b = ax1.bar(
        [i - width / 2 for i in x], baselines, width, label="Baseline", color="#888888"
    )
    bars_t = ax1.bar(
        [i + width / 2 for i in x], trained_acc, width, label="GRPO-trained", color="#1f77b4"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("MolecularIQ: Baseline vs GRPO-trained — per task configuration")
    ax1.legend()
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars_b, baselines):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=8,
        )
    for bar, acc in zip(bars_t, trained_acc):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=8,
        )

    # Delta bar chart
    colors      = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    delta_bars  = ax2.bar(x, deltas, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Accuracy delta (pp)")
    ax2.set_title("Improvement over baseline (percentage points)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)
    for bar, d in zip(delta_bars, deltas):
        offset = 0.5 if d >= 0 else -1.5
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
            f"{d:+.1f}pp", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[summary] chart saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the automated training and comparison workflow.

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
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max_steps in every training config (e.g. 50 for a sanity run).",
    )
    p.add_argument(
        "--num-train", type=int, default=5000,
        help="Training samples per experiment (default: 5000).",
    )
    p.add_argument(
        "--num-eval", type=int, default=200,
        help="Evaluation samples per experiment (default: 200).",
    )
    p.add_argument(
        "--experiments", nargs="+", metavar="NAME", default=None,
        help=f"Run only these experiment names. Available: {available}",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip preprocess / train / eval; just regenerate the summary "
             "chart from existing JSON files.",
    )
    args = p.parse_args()

    # Select experiments
    if args.experiments:
        name_set = set(args.experiments)
        exps = [e for e in EXPERIMENTS if e.name in name_set]
        missing = name_set - {e.name for e in exps}
        if missing:
            p.error(f"Unknown experiment(s): {missing}.  Available: {available}")
    else:
        exps = list(EXPERIMENTS)

    print(f"\nExperiments to run ({len(exps)}):")
    for exp in exps:
        print(f"  {exp.name:<30} task_type={exp.task_type}  properties={exp.properties}")

    # Load base YAML config once
    with open(_BASE_CONFIG) as fh:
        base_cfg = yaml.safe_load(fh)

    results: list[tuple[Experiment, dict]] = []

    for exp in exps:
        print(f"\n{'#' * 64}")
        print(f"# EXPERIMENT: {exp.name}  —  {exp.label.replace(chr(10), ' ')}")
        print(f"{'#' * 64}")

        if args.report_only:
            if not _eval_results_exist(exp.eval_dir):
                print(f"[warn] No eval results for {exp.name} at {exp.eval_dir} — skipping")
                continue
        else:
            step_preprocess(exp, args.num_train)
            step_write_config(exp, base_cfg, args.max_steps)
            step_train(exp)

        result = step_evaluate(exp, args.num_eval)
        results.append((exp, result))

        b = result["baseline"]["accuracy"] * 100
        t = result["trained"]["accuracy"] * 100
        print(f"\n[result] {exp.name}: baseline={b:.1f}%  trained={t:.1f}%  delta={t - b:+.1f}pp")

    if not results:
        print("\nNo results to report — nothing to do.")
        return

    _print_table(results)

    summary_dir = _REPO_ROOT / "outputs" / "comparison"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_data = [
        {
            "name":              exp.name,
            "label":             exp.label.replace("\n", " "),
            "task_type":         exp.task_type,
            "properties":        exp.properties,
            "baseline_accuracy": r["baseline"]["accuracy"],
            "trained_accuracy":  r["trained"]["accuracy"],
            "delta_pp":          (r["trained"]["accuracy"] - r["baseline"]["accuracy"]) * 100,
        }
        for exp, r in results
    ]
    summary_json = summary_dir / "summary.json"
    with open(summary_json, "w") as fh:
        json.dump(summary_data, fh, indent=2)
    print(f"[summary] JSON  saved to {summary_json}")

    _plot_summary(results, summary_dir / "summary.png")


if __name__ == "__main__":
    main()
