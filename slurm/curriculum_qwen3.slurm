#!/bin/bash
# =============================================================================
# Qwen3.5-0.8B counterpart of curriculum.slurm: the full 4-stage curriculum
# (counts -> simple index -> full index -> generation) as ONE job.
# Produces outputs/miq-curriculum-04-generation-qwen3.
# Reuses the SAME stage datasets and venv as the 0.5B run; writes to *-qwen3
# outputs so it runs next to the 0.5B results without clobbering them.
#
# Prereqs (one-time, on a LOGIN node):
#   - datasets already built by slurm/setup_leonardo.sh
#   - Qwen3 model cached: bash slurm/cache_model_leonardo.sh
#
#   sbatch slurm/curriculum_qwen3.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum-qwen3
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg for a <=30 min smoke test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 0.8B model -> a single A100 is plenty
#SBATCH --cpus-per-task=8               # 32 cores / 4 GPUs per node
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Uses the MODERN stack venv built by slurm/setup_qwen3.sh (transformers/trl new
# enough to load the qwen3_5 architecture), NOT the legacy 0.5B venv.
export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo-qwen3}"
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

# Datasets prebuilt -> this only trains. miq_curriculum_qwen3.yaml sets
# base_model: Qwen/Qwen3.5-0.8B (stage 1) and chains later stages from the
# previous output; the A100-tuned base config supplies the throughput-tuned
# hyperparameters (its model_name is overridden by base_model, so it is reused
# as-is). For a smoke test add: --max-steps-per-stage 20
grpo-curriculum \
    --config configs/multitask/miq_curriculum_qwen3.yaml \
    --base-train-config configs/multitask/miq_multitask_a100_train.yaml \
    --max-steps-per-stage 1500
