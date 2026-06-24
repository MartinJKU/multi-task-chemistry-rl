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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo}"               # venv lives in $WORK (NOT $HOME: quota)
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"                   # model + dataset cache (persistent)
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK/pip_cache}"       # keep pip cache out of $HOME
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"   # check with: module avail python
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.0}"                   # newest stable PyTorch as of 2026-06-17
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
CLEAN_VENV="${CLEAN_VENV:-1}"                                 # 1 = remove old venv first
# ---------------------------------------------------------------------------

echo "[setup] PROJECT_ROOT=$PROJECT_ROOT"
echo "[setup] VENV_DIR=$VENV_DIR"
echo "[setup] HF_HOME=$HF_HOME"
echo "[setup] PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "[setup] PYTORCH_VERSION=$PYTORCH_VERSION"
echo "[setup] PYTORCH_INDEX_URL=$PYTORCH_INDEX_URL"

module purge
module load "$PYTHON_MODULE"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$(dirname "$VENV_DIR")"

# 1) Virtual environment ----------------------------------------------------
if [[ "$CLEAN_VENV" == "1" && -d "$VENV_DIR" ]]; then
    case "$VENV_DIR" in
        "$WORK"/venvs/*)
            echo "[setup] removing existing venv for a clean install: $VENV_DIR"
            rm -rf "$VENV_DIR"
            ;;
        *)
            echo "[setup] refusing to remove VENV_DIR outside \$WORK/venvs: $VENV_DIR" >&2
            exit 2
            ;;
    esac
fi
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# 2) PyTorch for A100. Wheels bundle the CUDA runtime; the NVIDIA driver on
#    the compute node must still be new enough for the selected CUDA wheel.
#    If the cluster driver is too old, rerun with for example:
#      PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 PYTORCH_VERSION=2.5.1 bash slurm/setup_leonardo.sh
pip install "torch==${PYTORCH_VERSION}" --index-url "$PYTORCH_INDEX_URL"

# 3) Project + chemistry (moleculariq-core, git) ---------------------------
#    NOTE: we deliberately do NOT install the [bnb] extra here. bitsandbytes
#    0.45.x ships a prebuilt CUDA library linked against GLIBC_2.34, but the
#    RHEL 8 compute nodes only provide glibc 2.28, so the native library
#    cannot load and the 8-bit optimizer crashes at optimizer.step()
#    (NameError: str2optimizer8bit_blockwise). The configs therefore use
#    optim: adamw_torch instead of adamw_8bit.
cd "$PROJECT_ROOT"
pip install -c slurm/constraints-leonardo.txt -e ".[chem]"

# 4) Dependency sanity checks -----------------------------------------------
pip check
python - <<'PY'
import importlib.metadata as md
import torch

packages = [
    "torch",
    "transformers",
    "trl",
    "datasets",
    "pyarrow",
    "accelerate",
    "peft",
    "numpy",
]
for name in packages:
    print(f"[setup] {name}=={md.version(name)}")
print(f"[setup] torch cuda build={torch.version.cuda}")
print(f"[setup] torch cuda available on this node={torch.cuda.is_available()}")
PY

# 5) Pre-download the base model into the HF cache --------------------------
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "Qwen/Qwen2.5-0.5B-Instruct"
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("[setup] base model cached")
PY

# 6) Pre-build ALL datasets (downloads the MolecularIQ SMILES pool once) -----
#    After this, training jobs read datasets from disk and need no internet.
#    `--dataset-only` builds all 4 curriculum stage datasets, INCLUDING
#    data/miq_curriculum_01_counts_train (stage 1) which the strategy runs reuse
#    as their counts warm-start dataset -- so we do NOT rebuild it separately.
grpo-curriculum \
    --config configs/multitask/miq_curriculum.yaml \
    --dataset-only \
    --overwrite-datasets
grpo-preprocess-multitask \
    --config configs/multitask/miq_multitask_balanced.yaml \
    --overwrite
grpo-preprocess-multitask \
    --config configs/multitask/miq_multitask_pooled.yaml \
    --overwrite
grpo-preprocess-multitask \
    --config configs/multitask/miq_multitask_adaptive.yaml \
    --overwrite

echo "[setup] DONE. Now edit the #SBATCH --account line in the .slurm files and submit."
