#!/bin/bash
# =============================================================================
# EXPERIMENTAL: curriculum with vLLM generation (5-20x faster generation).
# Requires the SEPARATE, newer-stack vLLM venv built by slurm/setup_vllm.sh.
# vLLM colocates with training on a single A100 (0.5B model leaves plenty room).
#
#   bash slurm/setup_vllm.sh          # one-time, on a login node
#   sbatch slurm/curriculum_vllm.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum-vllm
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo-vllm}"   # the SEPARATE vLLM venv
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

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
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

grpo-curriculum \
    --config configs/multitask/miq_curriculum.yaml \
    --base-train-config configs/multitask/miq_multitask_a100_vllm_train.yaml \
    --max-steps-per-stage 1500
