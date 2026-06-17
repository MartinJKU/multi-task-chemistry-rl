#!/bin/bash
# =============================================================================
# Path A: reproduce the adaptive/balanced/pooled comparison (your heatmap),
# now with the counts warm-start + index difficulty filters.
#   1. counts pretrain  -> outputs/miq-curriculum-01-counts (the warm-start base)
#   2. train + eval each strategy
# Datasets must already be built by slurm/setup_leonardo.sh.
#
#   sbatch slurm/strategies.slurm
# =============================================================================
#SBATCH --job-name=miq-strategies
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi

# 1) Counts pretrain -> warm-start checkpoint the strategy configs point at.
grpo-train --config configs/multitask/miq_curriculum_01_counts_train.yaml

# 2) Each strategy: train (warm-started + index-filtered) then evaluate.
#    NOTE: the 'adaptive' DATASET was prebuilt before any eval existed, so its
#    sampling is uniform on this pass. For true error-weighted adaptive sampling,
#    after the balanced eval rebuild it on a LOGIN node (needs internet):
#      grpo-preprocess-multitask --config configs/multitask/miq_multitask_adaptive.yaml
#    (it reads outputs/multitask_eval/balanced/summary.json) then retrain adaptive.
for s in balanced pooled adaptive; do
    grpo-train --config configs/multitask/miq_multitask_${s}_train.yaml
    grpo-evaluate-multitask \
        --config configs/multitask/miq_multitask_${s}.yaml \
        --model outputs/miq-multitask-${s}-grpo \
        --model-label ${s}
done

# 3) Build the comparison report/heatmap from the three eval summaries.
#    Defaults: eval-dir=outputs/multitask_eval, out-dir=outputs/report.
grpo-report --outputs-dir outputs || true
