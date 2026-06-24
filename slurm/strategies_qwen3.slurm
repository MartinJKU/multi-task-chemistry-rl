#!/bin/bash
# =============================================================================
# Qwen3.5-0.8B counterpart of strategies.slurm: the adaptive/balanced/pooled
# comparison, warm-started from a Qwen3 counts checkpoint.
#   1. counts pretrain  -> outputs/miq-curriculum-01-counts-qwen3 (warm-start base)
#   2. train + eval each strategy -> outputs/miq-multitask-<s>-grpo-qwen3
# Reuses the SAME datasets and venv as the 0.5B run; writes to *-qwen3 outputs so
# it runs next to the 0.5B results without clobbering them.
#
# Prereqs (one-time, on a LOGIN node):
#   - datasets already built by slurm/setup_leonardo.sh
#   - Qwen3 model cached: bash slurm/cache_model_leonardo.sh
#
#   sbatch slurm/strategies_qwen3.slurm
# =============================================================================
#SBATCH --job-name=miq-strategies-qwen3
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

# Uses the MODERN stack venv built by slurm/setup_qwen3.sh (transformers/trl new
# enough to load the qwen3_5 architecture), NOT the legacy 0.5B venv.
export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo-qwen3}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

export PYTHONUNBUFFERED=1          # stream logs live instead of block-buffering
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
# Fail fast if CUDA is not visible (otherwise training would silently run on CPU)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

# 1) Counts pretrain -> warm-start checkpoint the Qwen3 strategy configs point at.
grpo-train --config configs/multitask/miq_curriculum_01_counts_qwen3_train.yaml

# 2) Each strategy: train (warm-started + index-filtered) then evaluate.
#    The baseline reference is the Qwen3 base model (a separate subfolder from the
#    0.5B baseline), so the report shows lift over the right base model.
#    NOTE: the 'adaptive' DATASET is shared with the 0.5B run, so its sampling is
#    uniform on this pass. For true error-weighted adaptive sampling on Qwen3, see
#    the header note in configs/multitask/miq_multitask_adaptive_qwen3_train.yaml.
for s in balanced pooled adaptive; do
    grpo-train --config configs/multitask/miq_multitask_${s}_qwen3_train.yaml
    grpo-evaluate-multitask \
        --config configs/multitask/miq_multitask_${s}.yaml \
        --model outputs/miq-multitask-${s}-grpo-qwen3 \
        --model-label ${s}-qwen3 \
        --baseline-model Qwen/Qwen3.5-0.8B \
        --baseline-label baseline-qwen3
done

# 3) Build the comparison report/heatmap. It discovers ALL eval summaries under
#    outputs/multitask_eval, so the 0.5B and Qwen3 runs appear side by side.
grpo-report --outputs-dir outputs || true
