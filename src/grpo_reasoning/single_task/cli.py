from __future__ import annotations

import argparse
from pathlib import Path


def preprocess_main() -> None:
    """Run the single-task dataset preprocessing command.

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

    from ..common.tasks import list_tasks
    from ..common.utils import filter_moleculariq_kwargs
    from .data import build_and_save

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


def evaluate_main() -> None:
    """Run the single-task evaluation command.

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

    from ..common.eval import evaluate
    from ..common.tasks import list_tasks
    from ..common.utils import filter_moleculariq_kwargs
    from .plotting import plot_baseline_vs_trained

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
