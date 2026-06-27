#!/bin/bash
# =============================================================================
# H1 x H4 transfer / pass@k study.
#
#   1. Train ONE pooled multitask GRPO model on single-property count+index
#      tasks (configs/multitask/miq_transfer_train.yaml), warm-started from the
#      instruct base. Held out: multi-* compositions, constraint generation, and
#      the hba_count property.
#   2. pass@k (base vs. trained) on in-distribution AND held-out tasks
#      (configs/multitask/miq_transfer_probe.yaml).
#   3. Plot the crossover/transfer figure.
#
# The training dataset must be built FIRST on a login node (needs internet for
# the SMILES pool):
#   grpo-preprocess-multitask --config configs/multitask/miq_transfer.yaml --overwrite
# The pass@k probe generates its eval items on the fly from the cached pool.
#
#   sbatch slurm/transfer_passk.slurm
# =============================================================================
#SBATCH --job-name=miq-transfer-passk
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
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
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] {torch.cuda.get_device_name(0)}')"

# 1) Train the pooled transfer model (dataset prebuilt on a login node).
grpo-train --config configs/multitask/miq_transfer_train.yaml

# 2) pass@k: base reference is evaluated automatically alongside the trained model.
#    Lower --n-completions / --num-samples if you are short on wall time; raise
#    them for tighter CIs and a clearer large-k crossover.
grpo-eval-passk \
    --config configs/multitask/miq_transfer_probe.yaml \
    --model outputs/miq-transfer-grpo \
    --model-label grpo \
    --baseline-model Qwen/Qwen2.5-0.5B-Instruct \
    --baseline-label base \
    --num-samples 100 \
    --n-completions 256 \
    --temperature 0.8 \
    --out-dir outputs/passk_eval

# 3) Build the crossover / transfer figure (PNG + PDF + crossover CSV).
grpo-plot-passk \
    --passk-dir outputs/passk_eval \
    --out-dir outputs/passk_report \
    --base-label base \
    --title "MolecularIQ pass@k: elicitation vs. expansion (in-dist vs. held-out)"

echo "[done] see outputs/passk_report/pass_at_k.{png,pdf} and *_crossover.csv"
