from __future__ import annotations

import argparse


def train_main() -> None:
    """Run the GRPO training command.

    The same command trains both single-task and multitask datasets; the
    trainer auto-detects mixed datasets from their `task_type` column.

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


def sft_main() -> None:
    """Run supervised warm-start fine-tuning from a YAML config."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--max-steps", type=int, default=None, help="Override max_steps for a quick run."
    )
    p.add_argument("--model-name", default=None, help="Override model_name from YAML.")
    p.add_argument("--output-dir", default=None, help="Override output_dir from YAML.")
    p.add_argument(
        "--dataset-path", default=None, help="Override dataset_path from YAML."
    )
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

    from .sft import SFTArgs, sft_train
    from .utils import load_yaml

    cfg_dict = load_yaml(args.config)
    if args.max_steps is not None:
        cfg_dict["max_steps"] = args.max_steps
    if args.model_name is not None:
        cfg_dict["model_name"] = args.model_name
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir
    if args.dataset_path is not None:
        cfg_dict["dataset_path"] = args.dataset_path
    if args.resume_from_checkpoint is not None:
        cfg_dict["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.no_save_on_interrupt:
        cfg_dict["save_on_interrupt"] = False

    sft_train(SFTArgs(**cfg_dict))
