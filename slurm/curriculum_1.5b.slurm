#!/bin/bash
# =============================================================================
# Path B (1.5B): run the full 4-stage curriculum (counts -> simple index ->
# full index -> generation) as ONE job for Qwen/Qwen2.5-1.5B-Instruct.
# Produces outputs/miq-curriculum-15b-04-generation. Writes to its own
# outputs/miq-curriculum-15b-* dirs, so it can run in parallel with the 0.5B
# curriculum job (each uses a single GPU) without clobbering it.
#
# Prereqs (one-time, on a LOGIN node): slurm/setup_leonardo.sh must have cached
# the 1.5B model and built the datasets (the datasets are shared with the 0.5B
# run; this config reuses them read-only).
#
#   sbatch slurm/curriculum_1.5b.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum-15b
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg for a <=30 min smoke test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 1.5B model still fits on a single A100 64GB
#SBATCH --cpus-per-task=8               # 32 cores / 4 GPUs per node
#SBATCH --mem=64G
#SBATCH --time=12:00:00                  # 1.5B is slower per step than 0.5B
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

# Compute nodes have no internet -> force HuggingFace offline (cache is prefilled)
export PYTHONUNBUFFERED=1          # stream logs live instead of block-buffering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs
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

# Datasets prebuilt -> this only trains. The 1.5B base config uses a smaller
# training batch than the 0.5B one (see miq_multitask_a100_1.5b_train.yaml); if
# you OOM, drop per_device_train_batch_size there to 8.
# For a smoke test add: --max-steps-per-stage 20
grpo-curriculum \
    --config configs/multitask/miq_curriculum_1.5b.yaml \
    --base-train-config configs/multitask/miq_multitask_a100_1.5b_train.yaml \
    --max-steps-per-stage 1500
