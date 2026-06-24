#!/bin/bash
# =============================================================================
# Build a SEPARATE venv with a modern transformers/TRL stack so we can train on
# newer model architectures (Qwen3.5-0.8B, model_type `qwen3_5`) that the pinned
# legacy stack (transformers 4.49 / trl 0.16) cannot even load.
#
# Run this ONCE on a Leonardo LOGIN node (login nodes have internet). It does NOT
# touch your main $WORK/venvs/grpo venv or its 0.5B results: it creates its own
# venv, reuses the shared $HF_HOME cache, and reuses the datasets you already
# built with setup_leonardo.sh (datasets are model-independent).
#
#   ssh mneumeis@login.leonardo.cineca.it
#   cd $WORK/grpo-reasoning
#   git fetch origin claude/stoic-tesla-1hfhyu && git checkout claude/stoic-tesla-1hfhyu
#   bash slurm/setup_qwen3.sh
#   sbatch slurm/curriculum_qwen3.slurm   # or strategies_qwen3.slurm
#
# Expect to iterate on exact versions: transformers<->trl compatibility moves
# fast. If a version conflict or an unknown-arch error appears, paste it and pin.
# =============================================================================
set -euo pipefail

# ---- adjust if you want different locations / model -------------------------
export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo-qwen3}"   # SEPARATE from the 0.5B venv
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"             # shared model/dataset cache
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK/pip_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
QWEN3_MODEL="${QWEN3_MODEL:-Qwen/Qwen3.5-0.8B}"
# Pin the modern stack here if you need reproducibility; empty = latest.
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers}"
TRL_SPEC="${TRL_SPEC:-trl}"
CLEAN_VENV="${CLEAN_VENV:-1}"
# ---------------------------------------------------------------------------

echo "[qwen3-setup] PROJECT_ROOT=$PROJECT_ROOT"
echo "[qwen3-setup] VENV_DIR=$VENV_DIR"
echo "[qwen3-setup] HF_HOME=$HF_HOME"
echo "[qwen3-setup] QWEN3_MODEL=$QWEN3_MODEL"
echo "[qwen3-setup] PYTORCH_VERSION=$PYTORCH_VERSION  PYTORCH_INDEX_URL=$PYTORCH_INDEX_URL"

module purge
module load "$PYTHON_MODULE"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$(dirname "$VENV_DIR")"

# 1) Fresh, isolated venv -----------------------------------------------------
if [[ "$CLEAN_VENV" == "1" && -d "$VENV_DIR" ]]; then
    case "$VENV_DIR" in
        "$WORK"/venvs/*)
            echo "[qwen3-setup] removing existing venv for a clean install: $VENV_DIR"
            rm -rf "$VENV_DIR"
            ;;
        *)
            echo "[qwen3-setup] refusing to remove VENV_DIR outside \$WORK/venvs: $VENV_DIR" >&2
            exit 2
            ;;
    esac
fi
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# 2) PyTorch (same wheel selection knobs as setup_leonardo.sh) ---------------
pip install "torch==${PYTORCH_VERSION}" --index-url "$PYTORCH_INDEX_URL"

# 3) Modern transformers + TRL that understand the new architecture ----------
#    Unpinned by default so pip resolves a mutually compatible pair; override
#    with TRANSFORMERS_SPEC / TRL_SPEC to lock versions.
pip install "$TRANSFORMERS_SPEC" "$TRL_SPEC"

# 4) Project CODE only (--no-deps so the legacy pins in pyproject.toml do NOT
#    downgrade the modern transformers/trl just installed) ---------------------
cd "$PROJECT_ROOT"
pip install -e . --no-deps

# 5) Remaining runtime deps + chemistry verifier ------------------------------
pip install datasets accelerate peft matplotlib pyyaml tqdm
pip install "moleculariq-core @ git+https://github.com/ml-jku/moleculariq-core.git"

# 5b) Align NumPy with the MODERN stack. The legacy 0.5B venv pins numpy<2 for
#     old RDKit wheels, but torch 2.7 / transformers 5.x are built against the
#     NumPy 2.x ABI: with numpy<2 here, numpy cannot even import itself against
#     these wheels and `import torch` dies with
#       ImportError: cannot import name '_dtype_ctypes' from partially
#       initialized module 'numpy.core' (circular import).
#     Force a clean numpy 2.x LAST so nothing installed above can downgrade it.
#     Modern RDKit (pulled by moleculariq-core) supports NumPy 2.x.
pip install --force-reinstall --no-cache-dir "numpy>=2.1,<3"

# 6) Sanity check: versions + that THIS stack can build the new architecture --
pip check || echo "[qwen3-setup] pip check reported issues (often harmless across the modern stack)"
QWEN3_MODEL="$QWEN3_MODEL" python - <<'PY'
import os
import numpy, torch, transformers, trl
print(f"[qwen3-setup] numpy={numpy.__version__} (must be 2.x for the modern stack)")
print(f"[qwen3-setup] torch={torch.__version__} cuda_build={torch.version.cuda}")
print(f"[qwen3-setup] transformers={transformers.__version__} trl={trl.__version__}")
assert int(numpy.__version__.split(".")[0]) >= 2, "numpy<2 breaks import torch on this stack"

# Fail loudly here (on the login node, with internet) rather than inside a job if
# the installed transformers still does not know the new architecture.
from transformers import AutoConfig
m = os.environ["QWEN3_MODEL"]
cfg = AutoConfig.from_pretrained(m)
print(f"[qwen3-setup] {m} arch recognized: model_type={getattr(cfg, 'model_type', '?')}")
PY

# 7) Pre-cache the model into HF_HOME so offline compute nodes can load it ----
QWEN3_MODEL="$QWEN3_MODEL" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
m = os.environ["QWEN3_MODEL"]
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print(f"[qwen3-setup] {m} weights cached into HF_HOME")
PY

echo "[qwen3-setup] DONE."
echo "[qwen3-setup] Submit jobs with VENV_DIR pointed at this venv, e.g.:"
echo "  sbatch slurm/curriculum_qwen3.slurm"
echo "  sbatch slurm/strategies_qwen3.slurm"
