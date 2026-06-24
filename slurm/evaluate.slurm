#!/bin/bash
# =============================================================================
# Evaluate trained curriculum models against ONE filtered suite, then build the
# comparison report/heatmap. Eval regenerates the test set live from
# moleculariq_core + the cached SMILES pool, so it runs offline on a GPU node.
#
# The untrained base model is always evaluated once as a 'baseline' reference
# (cached in outputs/multitask_eval/baseline), so the report shows lift over it.
#
#   sbatch slurm/evaluate.slurm                       # final curriculum model only
#   EVAL_ALL_STAGES=1 sbatch slurm/evaluate.slurm     # + every stage (forgetting check)
#
# Overridable via env:
#   EVAL_CONFIG       eval suite (default: balanced; pooled/adaptive are equivalent)
#   EVAL_ALL_STAGES   1 to also eval each curriculum stage (default: 0)
#   NUM_SAMPLES       test samples per task (default: 200)
# =============================================================================
#SBATCH --job-name=miq-eval
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg (+ lower --time) for final-only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 0.5B model -> a single A100 is plenty
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_CONFIG="${EVAL_CONFIG:-configs/multitask/miq_multitask_balanced.yaml}"
EVAL_ALL_STAGES="${EVAL_ALL_STAGES:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
# Fail fast if CUDA is not visible (otherwise eval would silently run on CPU)
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

echo "[eval] suite=$EVAL_CONFIG num_samples=$NUM_SAMPLES all_stages=$EVAL_ALL_STAGES"

# 1) Final curriculum model -> outputs/multitask_eval/curriculum/summary.json
grpo-evaluate-multitask \
    --config "$EVAL_CONFIG" \
    --model outputs/miq-curriculum-04-generation \
    --model-label curriculum \
    --num-samples "$NUM_SAMPLES"

# 2) Optional: evaluate every stage against the same suite to check that the
#    broadened replay stopped count tasks collapsing between stages.
if [[ "$EVAL_ALL_STAGES" == "1" ]]; then
    for s in 01-counts 02-simple-index 03-full-index 04-generation; do
        model_dir="outputs/miq-curriculum-${s}"
        if [[ ! -d "$model_dir" ]]; then
            echo "[eval] skip missing $model_dir"
            continue
        fi
        grpo-evaluate-multitask \
            --config "$EVAL_CONFIG" \
            --model "$model_dir" \
            --model-label "curriculum_${s}" \
            --num-samples "$NUM_SAMPLES"
    done
fi

# 3) Build comparison plots/heatmap + CSV tables from every eval summary found.
#    Defaults: eval-dir=outputs/multitask_eval, out-dir=outputs/report.
grpo-report --outputs-dir outputs || true

echo "[eval] DONE. See outputs/multitask_eval/*/summary.json and outputs/report/."
