from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_log_history(output_dir: Path | str) -> list[dict]:
    """Load trainer log history from an output directory.

    Args:
        output_dir: Trainer output directory or checkpoint parent directory.

    Returns:
        Trainer `log_history` rows.

    Raises:
        FileNotFoundError: If no trainer state file is found.
    """
    output_dir = Path(output_dir)
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        ckpts = sorted(
            (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No trainer_state.json under {output_dir}")
        state_path = ckpts[-1] / "trainer_state.json"

    with open(state_path) as f:
        return json.load(f)["log_history"]


def plot_training_curves(
    output_dir: Path | str, save_dir: Path | str | None = None
) -> Path:
    """Plot GRPO training curves from trainer state.

    Args:
        output_dir: Trainer output directory or checkpoint parent directory.
        save_dir: Optional directory where the PNG should be written.

    Returns:
        Path to the generated training-curve PNG.

    Raises:
        FileNotFoundError: If no trainer state file is found.
    """
    output_dir = Path(output_dir)
    save_dir = Path(save_dir) if save_dir else output_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    log = _load_log_history(output_dir)
    steps, loss, kl, reward = [], [], [], []
    component_keys: set[str] = set()
    for row in log:
        if "loss" in row:
            steps.append(row["step"])
            loss.append(row["loss"])
            kl.append(row.get("kl"))
            reward.append(row.get("reward"))
        for k in row:
            if k.startswith("rewards/"):
                component_keys.add(k)
    component_keys = sorted(component_keys)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(steps, loss)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    if any(r is not None for r in reward):
        axes[1].plot(steps, [r if r is not None else float("nan") for r in reward])
        axes[1].set_title("Total reward (mean per step)")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("reward")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    if any(k is not None for k in kl):
        axes[2].plot(steps, [k if k is not None else float("nan") for k in kl])
        axes[2].set_title("KL to reference")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("KL")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")

    if component_keys:
        for key in component_keys:
            ys = [row.get(key) for row in log if "loss" in row]
            axes[3].plot(
                steps,
                [y if y is not None else float("nan") for y in ys],
                label=key.replace("rewards/", ""),
            )
        axes[3].set_title("Reward components")
        axes[3].set_xlabel("step")
        axes[3].set_ylabel("reward")
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3)
    else:
        axes[3].axis("off")

    fig.suptitle(f"GRPO training: {output_dir.name}")
    fig.tight_layout()
    out_path = save_dir / "training_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


def plot_baseline_vs_trained(
    baseline_metrics: dict,
    trained_metrics: dict,
    save_path: Path | str,
    title: str = "Baseline vs GRPO-trained",
) -> Path:
    """Plot baseline and trained evaluation accuracy.

    Args:
        baseline_metrics: Metrics dictionary for the baseline model.
        trained_metrics: Metrics dictionary for the trained model.
        save_path: Path where the comparison PNG should be written.
        title: Figure title.

    Returns:
        Path to the generated comparison PNG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["baseline", "trained"]
    accs = [baseline_metrics["accuracy"] * 100, trained_metrics["accuracy"] * 100]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, accs, color=["#888", "#1f77b4"])
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(100, max(accs) * 1.1))
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {save_path}")
    return save_path
