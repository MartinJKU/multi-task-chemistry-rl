from __future__ import annotations

import argparse
from pathlib import Path


def preprocess_main() -> None:
    """Run the dataset preprocessing command.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True, help="Output directory (save_to_disk format).")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--task-type",
        default=None,
        help=(
            "MolecularIQ task variant: single_count, multi_count, single_index, "
            "multi_index, constraint_generation."
        ),
    )
    p.add_argument(
        "--properties",
        nargs="+",
        default=None,
        help=(
            "MolecularIQ property name(s) to ask about, e.g. ring_count. "
            "For single_* tasks pass exactly one."
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="MolecularIQ generator seed.")
    p.add_argument(
        "--system-prompt-style",
        default=None,
        choices=["with_key_hints", "concise"],
        help="MolecularIQ system prompt style.",
    )
    p.add_argument(
        "--constraint-operator",
        default=None,
        help="Operator for constraint_generation (e.g. '=', '>=', '<=', '>', '<').",
    )
    args = p.parse_args()

    from .data import build_and_save
    from .tasks import list_tasks
    from .utils import filter_moleculariq_kwargs

    if args.task not in list_tasks():
        p.error(f"Unknown task {args.task!r}. Available: {list_tasks()}")

    task_kwargs = filter_moleculariq_kwargs(
        args.task,
        task_type=args.task_type,
        properties=args.properties,
        seed=args.seed,
        system_prompt_style=args.system_prompt_style,
        constraint_operator=args.constraint_operator,
    )

    out = build_and_save(
        task_name=args.task,
        out_dir=args.out,
        split=args.split,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
        task_kwargs=task_kwargs,
    )
    print(f"Saved preprocessed dataset to {out}")


def preprocess_multitask_main() -> None:
    """Run multitask MolecularIQ preprocessing from a YAML config."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="Override out_dir from the YAML.")
    p.add_argument("--strategy", default=None, help="Override strategy from the YAML.")
    p.add_argument("--total-samples", type=int, default=None)
    p.add_argument("--samples-per-task", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from .multitask import MultitaskDatasetConfig, build_and_save_multitask
    from .utils import load_yaml

    cfg_dict = load_yaml(args.config)
    if args.out is not None:
        cfg_dict["out_dir"] = args.out
    if args.strategy is not None:
        cfg_dict["strategy"] = args.strategy
    if args.total_samples is not None:
        cfg_dict["total_samples"] = args.total_samples
    if args.samples_per_task is not None:
        cfg_dict["samples_per_task"] = args.samples_per_task

    out = build_and_save_multitask(
        MultitaskDatasetConfig.from_dict(cfg_dict),
        overwrite=args.overwrite,
    )
    print(f"Saved multitask dataset to {out}")


def audit_moleculariq_main() -> None:
    """Audit MolecularIQ source-pool coverage for configured task specs."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/miq_experiment_suite.yaml",
        help="YAML file with a `tasks` list to audit.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/moleculariq_dataset_audit",
        help="Directory where JSON/CSV audit files are written.",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--repo", default="ml-jku/moleculariq-trainPool")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-raw-rows",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test before scanning the full pool.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print per-task progress every N raw rows; use 0 to disable.",
    )
    args = p.parse_args()

    from .dataset_audit import audit_moleculariq_dataset

    audit_moleculariq_dataset(
        config_path=args.config,
        out_dir=args.out_dir,
        split=args.split,
        repo=args.repo,
        seed=args.seed,
        max_raw_rows=args.max_raw_rows,
        progress_every=args.progress_every,
    )


def train_main() -> None:
    """Run the GRPO training command.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--max-steps", type=int, default=None, help="Override max_steps for a quick run."
    )
    p.add_argument("--output-dir", default=None, help="Override output_dir from the YAML.")
    p.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume from a checkpoint path. If passed without a value, resume from "
            "the latest checkpoint under output_dir."
        ),
    )
    p.add_argument(
        "--no-save-on-interrupt",
        action="store_true",
        help="Disable best-effort checkpoint saving when Ctrl+C interrupts training.",
    )
    args = p.parse_args()

    from .train import TrainArgs, train
    from .utils import load_yaml

    cfg_dict = load_yaml(args.config)
    if args.max_steps is not None:
        cfg_dict["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir
    if args.resume_from_checkpoint is not None:
        cfg_dict["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.no_save_on_interrupt:
        cfg_dict["save_on_interrupt"] = False

    train(TrainArgs(**cfg_dict))


def curriculum_main() -> None:
    """Run staged curriculum training from a YAML config."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--overwrite-datasets",
        action="store_true",
        help="Rebuild stage datasets even if they already exist.",
    )
    p.add_argument(
        "--start-stage",
        default=None,
        help="Start from this stage name instead of the first stage.",
    )
    p.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only build curriculum datasets; do not train.",
    )
    p.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=None,
        help="Override max_steps for every stage, useful for smoke tests.",
    )
    args = p.parse_args()

    from .curriculum import run_curriculum_from_file

    run_curriculum_from_file(
        args.config,
        overwrite_datasets=args.overwrite_datasets,
        start_stage=args.start_stage,
        dataset_only=args.dataset_only,
        max_steps_per_stage=args.max_steps_per_stage,
    )


def evaluate_main() -> None:
    """Run the model evaluation command.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid or no model selection is provided.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", default=None, help="Single model to eval.")
    p.add_argument("--baseline", default=None)
    p.add_argument("--trained", default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--figure", default=None)
    p.add_argument("--task-type", default=None)
    p.add_argument("--properties", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--system-prompt-style", default=None, choices=["with_key_hints", "concise"]
    )
    p.add_argument("--constraint-operator", default=None)
    args = p.parse_args()

    import gc

    import torch

    from .eval import evaluate
    from .plotting import plot_baseline_vs_trained
    from .tasks import list_tasks
    from .utils import filter_moleculariq_kwargs

    if args.task not in list_tasks():
        p.error(f"Unknown task {args.task!r}. Available: {list_tasks()}")

    task_kwargs = filter_moleculariq_kwargs(
        args.task,
        task_type=args.task_type,
        properties=args.properties,
        seed=args.seed,
        system_prompt_style=args.system_prompt_style,
        constraint_operator=args.constraint_operator,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model is not None:
        evaluate(
            model_path=args.model,
            task_name=args.task,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            save_path=out_dir / f"{Path(args.model).name}_eval.json",
            task_kwargs=task_kwargs,
        )
        return

    if not (args.baseline and args.trained):
        raise SystemExit("Provide --model OR both --baseline and --trained.")

    print(f"\n=== Baseline: {args.baseline} ===")
    baseline = evaluate(
        model_path=args.baseline,
        task_name=args.task,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        save_path=out_dir / "baseline_eval.json",
        task_kwargs=task_kwargs,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n=== Trained: {args.trained} ===")
    trained = evaluate(
        model_path=args.trained,
        task_name=args.task,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        save_path=out_dir / "trained_eval.json",
        task_kwargs=task_kwargs,
    )

    print("\n================ COMPARISON ================")
    b = baseline["accuracy"] * 100
    t = trained["accuracy"] * 100
    print(f"Baseline accuracy: {b:.2f}%")
    print(f"Trained  accuracy: {t:.2f}%")
    print(f"Delta:             {t - b:+.2f} pp")

    fig_path = Path(args.figure) if args.figure else out_dir / "baseline_vs_trained.png"
    plot_baseline_vs_trained(
        baseline, trained, fig_path, title=f"{args.task}: baseline vs GRPO-trained"
    )


def evaluate_multitask_main() -> None:
    """Evaluate one model on every task in a multitask config."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-label", default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out-dir", default="outputs/multitask_eval")
    args = p.parse_args()

    import gc
    import json
    from datetime import datetime

    import torch

    from .eval import evaluate
    from .multitask import MultitaskDatasetConfig
    from .utils import load_yaml

    cfg = MultitaskDatasetConfig.from_dict(load_yaml(args.config))
    label = args.model_label or Path(args.model).name.replace("/", "_")
    out_dir = Path(args.out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    for spec in cfg.tasks:
        print(f"\n=== Eval {label}: {spec.task_id} ===")
        metrics = evaluate(
            model_path=args.model,
            task_name="moleculariq",
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            save_path=out_dir / f"{spec.task_id}_eval.json",
            task_kwargs=spec.task_kwargs(default_seed=cfg.seed),
        )
        task_row = {
            "task_id": spec.task_id,
            "task_type": spec.task_type,
            "properties": list(spec.properties),
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
        }
        for key in (
            "partial_score_mean",
            "answer_present_rate",
            "json_valid_rate",
            "valid_smiles_rate",
        ):
            if key in metrics:
                task_row[key] = metrics[key]
        task_results.append(task_row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracies = [row["accuracy"] for row in task_results]
    summary = {
        "model_label": label,
        "model_path": args.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "worst_task_accuracy": min(accuracies) if accuracies else 0.0,
        "tasks": task_results,
    }
    partial_scores = [
        row["partial_score_mean"]
        for row in task_results
        if "partial_score_mean" in row
    ]
    if partial_scores:
        summary["macro_partial_score"] = sum(partial_scores) / len(partial_scores)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[multitask-eval] wrote {summary_path}")
    print(f"[multitask-eval] macro accuracy = {summary['macro_accuracy']:.2%}")
    print(f"[multitask-eval] worst task     = {summary['worst_task_accuracy']:.2%}")


def plot_training_main() -> None:
    """Run the training-curve plotting command.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, help="The trainer output directory.")
    p.add_argument(
        "--save-dir",
        default=None,
        help="Where to write the PNG (default: <output-dir>/figures).",
    )
    args = p.parse_args()

    from .plotting import plot_training_curves

    plot_training_curves(args.output_dir, save_dir=args.save_dir)


def report_main() -> None:
    """Create experiment comparison plots from outputs and eval summaries."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root directory containing trainer outputs and multitask_eval.",
    )
    p.add_argument(
        "--eval-dir",
        default=None,
        help="Directory containing multitask eval subfolders (default: <outputs-dir>/multitask_eval).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write report artifacts (default: <outputs-dir>/report).",
    )
    args = p.parse_args()

    try:
        from .reporting import create_experiment_report
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "matplotlib is required for report plotting. Install project "
                "dependencies with `pip install -r requirements.txt`."
            ) from exc
        raise

    artifacts = create_experiment_report(
        outputs_dir=args.outputs_dir,
        eval_dir=args.eval_dir,
        out_dir=args.out_dir,
    )
    print("\n[report] generated:")
    for group, paths in artifacts.items():
        if not paths:
            continue
        print(f"  {group}:")
        for path in paths:
            print(f"    {path}")
