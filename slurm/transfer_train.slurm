#!/bin/bash
# =============================================================================
# TRAIN ONLY (decoupled from eval). Capped at MAX_STEPS so a usable checkpoint
# lands in hours, and saves every 100 steps so eval can run on a partial model.
# Run slurm/transfer_eval.slurm separately once a checkpoint exists.
#
#   sbatch slurm/transfer_train.slurm                 # default 400 steps
#   MAX_STEPS=250 sbatch slurm/transfer_train.slurm   # even faster
# =============================================================================
#SBATCH --job-name=miq-transfer-train
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"
MAX_STEPS="${MAX_STEPS:-400}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] {torch.cuda.get_device_name(0)}')"

echo "[train] MAX_STEPS=$MAX_STEPS (checkpoints every 100 steps in outputs/miq-transfer-grpo)"
grpo-train \
    --config configs/multitask/miq_transfer_train.yaml \
    --max-steps "$MAX_STEPS"

echo "[done] checkpoints under outputs/miq-transfer-grpo (final = the output dir itself)"
ls -d outputs/miq-transfer-grpo/checkpoint-* 2>/dev/null || true
