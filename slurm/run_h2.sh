#!/bin/bash
# =============================================================================
# One-shot launcher for the H1 + H2 poster experiment. Run on a Leonardo LOGIN
# node (it needs internet for the preprocess step, then submits GPU jobs).
#
#   cd $WORK/grpo-reasoning
#   bash slurm/run_h2.sh
#
# It will:
#   1. build the balanced multitask dataset if it is missing (login node only),
#   2. submit the shaped and sparse GRPO arms in PARALLEL (two A100s),
#   3. submit the eval+report+poster job to start after BOTH arms finish.
#
# Optional overrides:
#   STEPS=1000 bash slurm/run_h2.sh        # shorter training if the queue is tight
#   NUM_SAMPLES=300 bash slurm/run_h2.sh   # more eval samples per task
# =============================================================================
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"
STEPS="${STEPS:-}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

cd "$PROJECT_ROOT"
mkdir -p logs

# 1) Balanced dataset (login node has internet for moleculariq_core + trainPool).
if [[ ! -d data/miq_multitask_balanced_train ]]; then
    echo "[run_h2] balanced dataset missing -> preprocessing on the login node"
    module purge
    module load "$PYTHON_MODULE"
    source "$VENV_DIR/bin/activate"
    grpo-preprocess-multitask --config configs/multitask/miq_multitask_balanced.yaml
    deactivate || true
else
    echo "[run_h2] balanced dataset already present"
fi

# 2) Two reward arms in parallel.
shaped_id=$(sbatch --parsable --export="ALL,VARIANT=shaped,STEPS=${STEPS}" slurm/h2_reward_ablation.slurm)
echo "[run_h2] submitted shaped arm: job $shaped_id"
sparse_id=$(sbatch --parsable --export="ALL,VARIANT=sparse,STEPS=${STEPS}" slurm/h2_reward_ablation.slurm)
echo "[run_h2] submitted sparse arm: job $sparse_id"

# 3) Eval + report + poster figures, gated on both arms succeeding.
eval_id=$(sbatch --parsable \
    --dependency="afterok:${shaped_id}:${sparse_id}" \
    --export="ALL,NUM_SAMPLES=${NUM_SAMPLES}" \
    slurm/h2_eval_report.slurm)
echo "[run_h2] submitted eval/report/poster: job $eval_id (after $shaped_id,$sparse_id)"

echo
echo "[run_h2] queue:"
squeue --me || true
echo
echo "[run_h2] when job $eval_id finishes, collect:"
echo "         outputs/poster/fig_h1_lift.png"
echo "         outputs/poster/fig_h2_by_tasktype.png"
echo "         outputs/poster/fig_h2_training.png"
echo "         outputs/poster/fig_h2_partial_vs_exact.png"
echo "         outputs/poster/poster_headline_numbers.csv"
