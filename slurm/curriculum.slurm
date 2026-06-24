#!/bin/bash
# =============================================================================
# Path B: run the full 4-stage curriculum (counts -> simple index -> full index
# -> generation) as ONE job. Produces outputs/miq-curriculum-04-generation.
# Datasets must already be built by slurm/setup_leonardo.sh.
#
#   sbatch slurm/curriculum.slurm
# =============================================================================
#SBATCH --job-name=miq-curriculum
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
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

export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
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

CURRICULUM_START_STAGE="${CURRICULUM_START_STAGE:-}"
CURRICULUM_END_STAGE="${CURRICULUM_END_STAGE:-}"
CURRICULUM_BASE_MODEL="${CURRICULUM_BASE_MODEL:-}"
MAX_STEPS_PER_STAGE="${MAX_STEPS_PER_STAGE:-1500}"

nvidia-smi
# Fail fast if CUDA is not visible (otherwise training would silently run on CPU)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

# Datasets prebuilt -> this only trains. The A100-tuned base config decodes far
# more completions in parallel to actually use the GPU (see miq_multitask_a100_train.yaml).
cmd=(
    grpo-curriculum
    --config configs/multitask/miq_curriculum.yaml
    --base-train-config configs/multitask/miq_multitask_a100_train.yaml
    --max-steps-per-stage "$MAX_STEPS_PER_STAGE"
)
if [[ -n "$CURRICULUM_START_STAGE" ]]; then
    cmd+=(--start-stage "$CURRICULUM_START_STAGE")
fi
if [[ -n "$CURRICULUM_END_STAGE" ]]; then
    cmd+=(--end-stage "$CURRICULUM_END_STAGE")
fi
if [[ -n "$CURRICULUM_BASE_MODEL" ]]; then
    cmd+=(--base-model "$CURRICULUM_BASE_MODEL")
fi

printf '[curriculum] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
