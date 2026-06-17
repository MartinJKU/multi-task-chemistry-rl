#!/bin/bash
# =============================================================================
# Path B: run the full 4-stage curriculum (counts -> simple index -> full index
# -> generation) as ONE job. Produces outputs/miq-curriculum-04-generation.
# Datasets must already be built by slurm/setup_leonardo.sh.
#
#   sbatch slurm/curriculum.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum
#SBATCH --account=YOUR_ACCOUNT          # <-- set this (list budgets with: saldo -b)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg for a <=30 min smoke test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 0.5B model -> a single A100 is plenty
#SBATCH --cpus-per-task=8               # 32 cores / 4 GPUs per node
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

# Compute nodes have no internet -> force HuggingFace offline (cache is prefilled)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
# Datasets prebuilt -> this only trains. For a smoke test add: --max-steps-per-stage 20
grpo-curriculum --config configs/multitask/miq_curriculum.yaml
