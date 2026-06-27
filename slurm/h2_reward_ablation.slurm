#!/bin/bash
# =============================================================================
# H2 (also produces the H1 generalist): dense-vs-sparse reward GRPO ablation.
#
# Trains ONE reward arm, selected by the VARIANT env var (shaped|sparse). Submit
# it TWICE so the two arms train in parallel on two A100s -> both finish in one
# overnight window:
#
#   VARIANT=shaped sbatch slurm/h2_reward_ablation.slurm   # dense reward (= H1 model)
#   VARIANT=sparse sbatch slurm/h2_reward_ablation.slurm   # exact-match only
#
# The two configs are identical except use_shaped_moleculariq_reward, and both
# start from the SAME base init, so any difference is attributable to the reward.
# The balanced dataset must already be built (slurm/run_h2.sh does this for you).
#
# Override the step budget if the queue is tight: STEPS=1000 VARIANT=... sbatch ...
# =============================================================================
#SBATCH --job-name=miq-h2
#SBATCH --account=EUHPC_D27_069        # list budgets with: saldo -b
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal                    # boost_qos_dbg (+ lower --time) for a smoke test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 0.5B model -> a single A100 is plenty
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

VARIANT="${VARIANT:?set VARIANT=shaped or VARIANT=sparse}"
case "$VARIANT" in
    shaped|sparse) ;;
    *) echo "VARIANT must be 'shaped' or 'sparse' (got '$VARIANT')"; exit 2 ;;
esac
CONFIG="configs/multitask/miq_h2_${VARIANT}_train.yaml"
STEPS="${STEPS:-}"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
mkdir -p logs

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT available!'; print(f'[gpu] cuda=True device={torch.cuda.get_device_name(0)}')"

echo "[h2] variant=$VARIANT config=$CONFIG steps=${STEPS:-<config default>}"
if [[ -n "$STEPS" ]]; then
    grpo-train --config "$CONFIG" --max-steps "$STEPS"
else
    grpo-train --config "$CONFIG"
fi
echo "[h2] DONE training $VARIANT -> outputs/miq-h2-${VARIANT}-grpo"
