#!/bin/bash
# =============================================================================
# Pre-cache an extra base model into the shared HuggingFace cache.
# Run this ONCE on a Leonardo LOGIN node (login nodes have internet); compute
# nodes are offline and read the model from $HF_HOME.
#
# Use this to add a second base model (e.g. Qwen3.5-0.8B) next to an existing
# checkout WITHOUT re-running the full setup_leonardo.sh: it reuses the same
# venv, the same HF cache, and the datasets you already built.
#
#   ssh mneumeis@login.leonardo.cineca.it
#   cd $WORK/grpo-reasoning
#   bash slurm/cache_model_leonardo.sh                 # defaults to Qwen3.5-0.8B
#   bash slurm/cache_model_leonardo.sh Qwen/Qwen3-0.6B # or any HF repo id
# =============================================================================
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-0.8B}"

export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

echo "[cache] MODEL=$MODEL"
echo "[cache] VENV_DIR=$VENV_DIR"
echo "[cache] HF_HOME=$HF_HOME"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
mkdir -p "$HF_HOME"

MODEL="$MODEL" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

m = os.environ["MODEL"]
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print(f"[cache] {m} cached into HF_HOME")
PY

echo "[cache] DONE. Compute-node jobs can now load $MODEL offline."
