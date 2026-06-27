#!/bin/bash
# =============================================================================
# EVAL ONLY (decoupled from training): pass@k for base vs. a checkpoint, + plot.
# Runs on WHATEVER checkpoint you point it at, so you are never hostage to
# training finishing. The base model is evaluated automatically as the reference.
#
#   # latest/full model:
#   sbatch slurm/transfer_eval.slurm
#   # a specific partial checkpoint (eval while training continues elsewhere):
#   MODEL=outputs/miq-transfer-grpo/checkpoint-200 sbatch slurm/transfer_eval.slurm
#   # quick base-only curve right now (no training needed):
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct MODEL_LABEL=base NO_BASELINE=1 sbatch slurm/transfer_eval.slurm
# =============================================================================
#SBATCH --job-name=miq-transfer-eval
#SBATCH --account=EUHPC_D27_069
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

# Tunables (override on the sbatch command line).
MODEL="${MODEL:-outputs/miq-transfer-grpo}"
MODEL_LABEL="${MODEL_LABEL:-grpo}"
NUM_SAMPLES="${NUM_SAMPLES:-80}"
N_COMPLETIONS="${N_COMPLETIONS:-128}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64,128}"
NO_BASELINE="${NO_BASELINE:-0}"

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

BASELINE_FLAG=""
if [[ "$NO_BASELINE" == "1" ]]; then BASELINE_FLAG="--no-baseline"; fi

echo "[eval] MODEL=$MODEL LABEL=$MODEL_LABEL items=$NUM_SAMPLES n=$N_COMPLETIONS"
grpo-eval-passk \
    --config configs/multitask/miq_transfer_probe.yaml \
    --model "$MODEL" \
    --model-label "$MODEL_LABEL" \
    --baseline-model Qwen/Qwen2.5-0.5B-Instruct \
    --baseline-label base \
    --num-samples "$NUM_SAMPLES" \
    --n-completions "$N_COMPLETIONS" \
    --k-values "$K_VALUES" \
    --temperature 0.8 \
    --max-new-tokens 256 \
    --out-dir outputs/passk_eval \
    $BASELINE_FLAG

grpo-plot-passk \
    --passk-dir outputs/passk_eval \
    --out-dir outputs/passk_report \
    --base-label base \
    --title "MolecularIQ pass@k: elicitation vs. expansion (in-dist vs. held-out)"

echo "[done] outputs/passk_report/pass_at_k.{png,pdf} + pass_at_k_crossover.csv"
