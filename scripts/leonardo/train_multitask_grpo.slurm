#!/bin/bash
# GRPO multitask training on CINECA Leonardo (Booster partition, 4x A100 64GB/node).
# Generation is vLLM-accelerated in colocate mode (one engine per GPU), which is
# what makes the larger num_generations in the multitask configs affordable.
#
# Submit:
#   sbatch scripts/leonardo/train_multitask_grpo.slurm \
#       configs/multitask/miq_multitask_pooled_train.yaml
#
# Before the first run, fill in the <PLACEHOLDERS> below and follow the offline
# pre-caching steps in scripts/leonardo/README.md (Booster compute nodes have no
# internet, so the model and dataset must already be on disk).
#
#SBATCH --job-name=miq-grpo
#SBATCH --account=<YOUR_PROJECT_ACCOUNT>     # REQUIRED budget account; check with `saldo -b`
#SBATCH --partition=boost_usr_prod           # Leonardo Booster GPU partition
###SBATCH --qos=boost_qos_dbg                # uncomment for a <=30 min debug slot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                  # one launcher; accelerate spawns one proc per GPU
#SBATCH --gres=gpu:4                         # GPUs per node (1..4); batch math stays valid for any
#SBATCH --cpus-per-task=32                   # 32 physical cores on a Booster node
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Config to train (default: pooled). Pass another as the first argument. ---
CONFIG="${1:-configs/multitask/miq_multitask_pooled_train.yaml}"

# --- Project + Python environment (EDIT THESE) ------------------------------
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$PROJECT_DIR/.venv}"           # venv with torch (CUDA), trl, and vllm installed
cd "$PROJECT_DIR"
mkdir -p logs

# Leonardo modules: match the CUDA your venv's torch/vllm were built against.
module purge
module load cuda/12.2                          # <-- adjust to your build (e.g. 12.1 / 12.3)

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- One process per GPU; vLLM is colocated on each GPU. --------------------
NGPU="${SLURM_GPUS_ON_NODE:-4}"
PORT=$((20000 + SLURM_JOB_ID % 10000))

# --- Offline mode: Booster compute nodes have NO outbound internet. ----------
# Pre-download the model + dataset on a login node first (see README).
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-32} / NGPU ))"

MULTI=""
if [ "$NGPU" -gt 1 ]; then MULTI="--multi_gpu"; fi

echo "[leonardo] config=$CONFIG  gpus=$NGPU  node=$(hostname)  port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

accelerate launch \
  --num_processes "$NGPU" \
  --num_machines 1 \
  $MULTI \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  scripts/train.py --config "$CONFIG"
