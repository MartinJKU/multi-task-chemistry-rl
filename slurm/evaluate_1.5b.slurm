#!/bin/bash
# =============================================================================
# Evaluate the 1.5B curriculum model against ONE filtered suite, then build the
# comparison report/heatmap. Mirrors evaluate.slurm but for the 1.5B run:
#   - target model:   outputs/miq-curriculum-15b-04-generation
#   - baseline model: Qwen/Qwen2.5-1.5B-Instruct (NOT the 0.5B default), so the
#     "lift over base model" comparison is apples-to-apples.
# Eval regenerates the test set live from moleculariq_core + the cached SMILES
# pool, so it runs offline on a GPU node.
#
#   PROJECT_ROOT=$WORK/grpo-reasoning-15b sbatch slurm/evaluate_1.5b.slurm
#   EVAL_ALL_STAGES=1 PROJECT_ROOT=$WORK/grpo-reasoning-15b sbatch slurm/evaluate_1.5b.slurm
#
# Set PROJECT_ROOT to wherever the 1.5B run wrote its outputs/ (the separate
# worktree/clone you trained it in). Overridable via env:
#   EVAL_CONFIG       eval suite (default: balanced; pooled/adaptive are equivalent)
#   EVAL_ALL_STAGES   1 to also eval each curriculum stage (default: 0)
#   NUM_SAMPLES       test samples per task (default: 200)
# =============================================================================
#SBATCH --job-name=miq-eval-15b
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg (+ lower --time) for final-only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 1.5B model still fits on a single A100 64GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00                  # 1.5B is slower per sample than 0.5B
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

# Compute nodes have no internet -> force HuggingFace offline (cache is prefilled)
export PYTHONUNBUFFERED=1          # stream logs live instead of block-buffering
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_CONFIG="${EVAL_CONFIG:-configs/multitask/miq_multitask_balanced.yaml}"
EVAL_ALL_STAGES="${EVAL_ALL_STAGES:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
# Fail fast if CUDA is not visible (otherwise eval would silently run on CPU)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

echo "[eval] suite=$EVAL_CONFIG num_samples=$NUM_SAMPLES all_stages=$EVAL_ALL_STAGES baseline=$BASELINE_MODEL"

# 1) Final 1.5B curriculum model -> outputs/multitask_eval/curriculum_15b/summary.json
#    The 1.5B base model is evaluated once as baseline_15b for the lift comparison.
grpo-evaluate-multitask \
    --config "$EVAL_CONFIG" \
    --model outputs/miq-curriculum-15b-04-generation \
    --model-label curriculum_15b \
    --baseline-model "$BASELINE_MODEL" \
    --baseline-label baseline_15b \
    --num-samples "$NUM_SAMPLES"

# 2) Optional: evaluate every 1.5B stage against the same suite (forgetting check).
if [[ "$EVAL_ALL_STAGES" == "1" ]]; then
    for s in 01-counts 02-simple-index 03-full-index 04-generation; do
        model_dir="outputs/miq-curriculum-15b-${s}"
        if [[ ! -d "$model_dir" ]]; then
            echo "[eval] skip missing $model_dir"
            continue
        fi
        grpo-evaluate-multitask \
            --config "$EVAL_CONFIG" \
            --model "$model_dir" \
            --model-label "curriculum_15b_${s}" \
            --baseline-model "$BASELINE_MODEL" \
            --baseline-label baseline_15b \
            --num-samples "$NUM_SAMPLES"
    done
fi

# 3) Build comparison plots/heatmap + CSV tables from every eval summary found.
grpo-report --outputs-dir outputs || true

echo "[eval] DONE. See outputs/multitask_eval/*/summary.json and outputs/report/."
