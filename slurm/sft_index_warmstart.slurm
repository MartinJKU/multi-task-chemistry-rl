#!/bin/bash
# =============================================================================
# Optional SFT warm-start for index tasks.
#
# Run after outputs/miq-curriculum-01-counts exists:
#   sbatch slurm/sft_index_warmstart.slurm
# =============================================================================
#SBATCH --job-name=miq-sft-index
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
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

SFT_MAX_STEPS="${SFT_MAX_STEPS:-}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

if [[ ! -d outputs/miq-curriculum-01-counts ]]; then
    echo "[sft] missing outputs/miq-curriculum-01-counts; train stage 01_counts first" >&2
    exit 2
fi

cmd=(grpo-sft --config configs/multitask/miq_sft_index_warmstart.yaml)
if [[ -n "$SFT_MAX_STEPS" ]]; then
    cmd+=(--max-steps "$SFT_MAX_STEPS")
fi

printf '[sft] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

grpo-evaluate-multitask \
    --config configs/multitask/miq_index_eval.yaml \
    --model outputs/miq-sft-index-warmstart \
    --model-label sft-index-warmstart \
    --out-dir outputs/multitask_eval-sft-gate \
    --no-baseline

grpo-check-index-gate \
    --summary outputs/multitask_eval-sft-gate/sft-index-warmstart/summary.json
