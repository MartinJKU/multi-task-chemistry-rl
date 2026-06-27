#!/bin/bash
# =============================================================================
# Evaluate the H1/H2 runs and build the poster figures.
#
# Evaluates three models on the balanced suite (the base 0.5B reference is
# auto-evaluated and cached), then runs grpo-report and the poster-figure
# script. Submit it with a dependency so it starts only after BOTH training
# arms finish -- slurm/run_h2.sh wires this up automatically. To run by hand:
#
#   sbatch --dependency=afterok:<shaped_jobid>:<sparse_jobid> slurm/h2_eval_report.slurm
#
# Outputs:
#   outputs/multitask_eval/{baseline,shaped,sparse}/summary.json
#   outputs/report/...            (full comparison report)
#   outputs/poster/fig_h1_*.png, fig_h2_*.png, poster_headline_numbers.csv
# =============================================================================
#SBATCH --job-name=miq-h2-eval
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

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_CONFIG="${EVAL_CONFIG:-configs/multitask/miq_multitask_balanced.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

echo "[h2-eval] suite=$EVAL_CONFIG num_samples=$NUM_SAMPLES"

# Dense arm first: this also triggers the cached base-model baseline eval.
grpo-evaluate-multitask \
    --config "$EVAL_CONFIG" \
    --model outputs/miq-h2-shaped-grpo \
    --model-label shaped \
    --num-samples "$NUM_SAMPLES"

grpo-evaluate-multitask \
    --config "$EVAL_CONFIG" \
    --model outputs/miq-h2-sparse-grpo \
    --model-label sparse \
    --num-samples "$NUM_SAMPLES"

# Full comparison report (heatmaps, training curves, CSVs).
grpo-report --outputs-dir outputs || true

# Poster-specific figures keyed to H1 and H2.
python scripts/multitask/make_poster_figures.py \
    --eval-dir outputs/multitask_eval \
    --outputs-dir outputs \
    --out-dir outputs/poster

echo "[h2-eval] DONE. Poster figures in outputs/poster/, full report in outputs/report/."
