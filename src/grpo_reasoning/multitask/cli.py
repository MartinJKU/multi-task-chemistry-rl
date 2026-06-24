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
    """Evaluate one model on every task in a multitask config.

    Always also evaluates the untrained base model as a ``baseline`` reference
    (once, cached) so every report shows the lift over the base model.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-label", default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out-dir", default="outputs/multitask_eval")
    p.add_argument(
        "--baseline-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Untrained reference model evaluated alongside the target model.",
    )
    p.add_argument(
        "--baseline-label",
        default="baseline",
        help="Label/subfolder used for the baseline eval.",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline reference eval.",
    )
    p.add_argument(
        "--overwrite-baseline",
        action="store_true",
        help="Re-run the baseline eval even if its summary.json already exists.",
    )
    args = p.parse_args()

    import gc
    import json
    from datetime import datetime

    import torch

    from ..common.eval import evaluate
    from ..common.utils import load_yaml
    from .dataset import MultitaskDatasetConfig

    cfg = MultitaskDatasetConfig.from_dict(load_yaml(args.config))
    eval_root = Path(args.out_dir)

    def _evaluate_model(model_path: str, label: str) -> dict:
        """Evaluate one model on every config task and write its summary."""
        out_dir = eval_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        task_results = []
        for spec in cfg.tasks:
            print(f"\n=== Eval {label}: {spec.task_id} ===")
            metrics = evaluate(
                model_path=model_path,
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
                "distinct_answer_rate",
                "most_common_answer_rate",
                "answer_present_rate",
                "json_valid_rate",
                "valid_smiles_rate",
                "index_precision_mean",
                "index_recall_mean",
                "index_gold_empty_rate",
                "index_empty_gold_total",
                "index_nonempty_total",
                "index_empty_gold_accuracy",
                "index_nonempty_accuracy",
                "index_nonempty_partial_score_mean",
                "index_nonempty_precision_mean",
                "index_nonempty_recall_mean",
                "avg_index_pred_len",
                "avg_index_gold_len",
                "avg_index_false_positives",
                "avg_index_false_negatives",
                "empty_gold_nonempty_rate",
                "superset_rate",
                "subset_rate",
                "constraint_satisfied_fraction_mean",
                "constraint_target_count",
                "constraint_target_macro_accuracy",
                "worst_constraint_target_accuracy",
                "ringless_when_ring_requested_rate",
                "canonical_smiles_distinct_rate",
                "canonical_smiles_most_common_rate",
                "successful_canonical_smiles_distinct_rate",
                "successful_canonical_smiles_most_common_rate",
                "trivial_alkane_rate",
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
            "model_path": model_path,
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
        print(f"[multitask-eval] {label}: macro accuracy = {summary['macro_accuracy']:.2%}")
        print(f"[multitask-eval] {label}: worst task     = {summary['worst_task_accuracy']:.2%}")
        return summary

    # Baseline first so the reference exists even if the target eval is interrupted.
    # Skip when it is the same model, already cached, or explicitly disabled.
    if (
        not args.no_baseline
        and args.baseline_label != (args.model_label or "")
        and Path(args.baseline_model) != Path(args.model)
    ):
        baseline_summary = eval_root / args.baseline_label / "summary.json"
        if baseline_summary.exists() and not args.overwrite_baseline:
            print(
                f"[multitask-eval] baseline already evaluated at {baseline_summary} "
                "(use --overwrite-baseline to redo)"
            )
        else:
            print(f"\n[multitask-eval] === baseline reference: {args.baseline_model} ===")
            _evaluate_model(args.baseline_model, args.baseline_label)

    label = args.model_label or Path(args.model).name.replace("/", "_")
    _evaluate_model(args.model, label)



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
        "--end-stage",
        default=None,
        help="Stop after this stage name instead of running all remaining stages.",
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
    p.add_argument(
        "--base-model",
        default=None,
        help=(
            "Override the model used for the first executed stage. Useful when "
            "continuing from an SFT warm-start checkpoint with --start-stage."
        ),
    )
    args = p.parse_args()

    from .curriculum import run_curriculum_from_file

    run_curriculum_from_file(
        args.config,
        overwrite_datasets=args.overwrite_datasets,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        dataset_only=args.dataset_only,
        max_steps_per_stage=args.max_steps_per_stage,
        base_train_config=args.base_train_config,
        base_model=args.base_model,
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
