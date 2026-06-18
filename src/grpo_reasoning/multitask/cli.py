from __future__ import annotations

import argparse
from pathlib import Path


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

    from ..common.utils import load_yaml
    from .dataset import MultitaskDatasetConfig, build_and_save_multitask

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
    p.add_argument(
        "--soft-match-threshold",
        type=float,
        default=0.9,
        help="Partial-score cutoff for the lenient soft_accuracy metric.",
    )
    p.add_argument(
        "--no-filtered-eval",
        action="store_true",
        help=(
            "Evaluate on the raw (unfiltered) test distribution. By default the "
            "same dataset filters used for training (e.g. shorter index lists) are "
            "applied to the test split so exact-match accuracy is achievable."
        ),
    )
    args = p.parse_args()

    import gc
    import json
    from dataclasses import replace
    from datetime import datetime

    import torch

    from ..common.eval import evaluate
    from ..common.utils import load_yaml
    from .dataset import MultitaskDatasetConfig, build_task_dataset

    cfg = MultitaskDatasetConfig.from_dict(load_yaml(args.config))
    label = args.model_label or Path(args.model).name.replace("/", "_")
    out_dir = Path(args.out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    for spec in cfg.tasks:
        print(f"\n=== Eval {label}: {spec.task_id} ===")
        eval_dataset = None
        if spec.filters and not args.no_filtered_eval:
            # Mirror the training distribution: build a filtered *test* split so the
            # exact-match target is achievable instead of arbitrarily long lists.
            built = build_task_dataset(
                replace(spec, num_samples=args.num_samples),
                split="test",
                default_seed=cfg.seed,
            )
            if len(built) > args.num_samples:
                built = built.select(range(args.num_samples))
            eval_dataset = built
        metrics = evaluate(
            model_path=args.model,
            task_name="moleculariq",
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            save_path=out_dir / f"{spec.task_id}_eval.json",
            task_kwargs=spec.task_kwargs(default_seed=cfg.seed),
            dataset=eval_dataset,
            soft_match_threshold=args.soft_match_threshold,
        )
        task_row = {
            "task_id": spec.task_id,
            "task_type": spec.task_type,
            "properties": list(spec.properties),
            "accuracy": metrics["accuracy"],
            "soft_accuracy": metrics["soft_accuracy"],
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
    soft_accuracies = [row["soft_accuracy"] for row in task_results]
    summary = {
        "model_label": label,
        "model_path": args.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "soft_match_threshold": args.soft_match_threshold,
        "filtered_eval": not args.no_filtered_eval,
        "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "worst_task_accuracy": min(accuracies) if accuracies else 0.0,
        "macro_soft_accuracy": (
            sum(soft_accuracies) / len(soft_accuracies) if soft_accuracies else 0.0
        ),
        "worst_task_soft_accuracy": min(soft_accuracies) if soft_accuracies else 0.0,
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
    print(f"[multitask-eval] macro accuracy      = {summary['macro_accuracy']:.2%}")
    print(f"[multitask-eval] macro soft accuracy = {summary['macro_soft_accuracy']:.2%}")
    print(f"[multitask-eval] worst task          = {summary['worst_task_accuracy']:.2%}")


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
    p.add_argument(
        "--base-train-config",
        default=None,
        help="Override the curriculum's base_train_config (e.g. an A100-tuned config).",
    )
    args = p.parse_args()

    from .curriculum import run_curriculum_from_file

    run_curriculum_from_file(
        args.config,
        overwrite_datasets=args.overwrite_datasets,
        start_stage=args.start_stage,
        dataset_only=args.dataset_only,
        max_steps_per_stage=args.max_steps_per_stage,
        base_train_config=args.base_train_config,
    )


def audit_moleculariq_main() -> None:
    """Audit MolecularIQ source-pool coverage for configured task specs."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/multitask/miq_experiment_suite.yaml",
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

    from .audit import audit_moleculariq_dataset

    audit_moleculariq_dataset(
        config_path=args.config,
        out_dir=args.out_dir,
        split=args.split,
        repo=args.repo,
        seed=args.seed,
        max_raw_rows=args.max_raw_rows,
        progress_every=args.progress_every,
    )


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
