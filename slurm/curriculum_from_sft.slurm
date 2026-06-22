#!/bin/bash
# =============================================================================
# Continue the curriculum from Stage 2 using the SFT index warm-start checkpoint.
#
# Run after outputs/miq-sft-index-warmstart exists:
#   sbatch slurm/curriculum_from_sft.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum-sft
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

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

MAX_STEPS_PER_STAGE="${MAX_STEPS_PER_STAGE:-1500}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

if [[ ! -d outputs/miq-sft-index-warmstart ]]; then
    echo "[curriculum] missing outputs/miq-sft-index-warmstart; run SFT first" >&2
    exit 2
fi

grpo-curriculum \
    --config configs/multitask/miq_curriculum.yaml \
    --base-train-config configs/multitask/miq_multitask_a100_train.yaml \
    --start-stage 02_simple_index \
    --base-model outputs/miq-sft-index-warmstart \
    --max-steps-per-stage "$MAX_STEPS_PER_STAGE"
