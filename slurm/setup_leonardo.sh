#!/bin/bash
# =============================================================================
# Run this ONCE on a Leonardo LOGIN node (login nodes have internet access).
# It builds the Python environment and pre-downloads EVERYTHING the training
# jobs need, because Leonardo COMPUTE nodes have NO internet access.
#
#   ssh mneumeis@login.leonardo.cineca.it
#   cd $WORK && git clone -b claude/amazing-ritchie-tbn0o3 <repo-url> grpo-reasoning
#   cd grpo-reasoning
#   bash slurm/setup_leonardo.sh
# =============================================================================
set -euo pipefail

# ---- adjust if you want different locations -------------------------------
export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"   # where you cloned the repo
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"               # venv lives in $WORK (NOT $HOME: quota)
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"                   # model + dataset cache (persistent)
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"   # check with: module avail python
# ---------------------------------------------------------------------------

echo "[setup] PROJECT_ROOT=$PROJECT_ROOT"
echo "[setup] VENV_DIR=$VENV_DIR"
echo "[setup] HF_HOME=$HF_HOME"

module purge
module load "$PYTHON_MODULE"

mkdir -p "$HF_HOME"

# 1) Virtual environment ----------------------------------------------------
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# 2) PyTorch for A100 (cu121 wheels bundle their own CUDA runtime) ----------
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) Project + chemistry (moleculariq-core, git) + 8-bit optimizer ----------
cd "$PROJECT_ROOT"
pip install -e ".[chem,bnb]"

# 4) Pre-download the base model into the HF cache --------------------------
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "Qwen/Qwen2.5-0.5B-Instruct"
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("[setup] base model cached")
PY

# 5) Pre-build ALL datasets (downloads the MolecularIQ SMILES pool once) -----
#    After this, training jobs read datasets from disk and need no internet.
grpo-curriculum --config configs/multitask/miq_curriculum.yaml --dataset-only
grpo-preprocess-multitask --config configs/multitask/miq_curriculum_01_counts.yaml
grpo-preprocess-multitask --config configs/multitask/miq_multitask_balanced.yaml
grpo-preprocess-multitask --config configs/multitask/miq_multitask_pooled.yaml
grpo-preprocess-multitask --config configs/multitask/miq_multitask_adaptive.yaml

echo "[setup] DONE. Now edit the #SBATCH --account line in the .slurm files and submit."
