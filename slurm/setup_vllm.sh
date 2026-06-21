#!/bin/bash
# =============================================================================
# EXPERIMENTAL: build a SEPARATE venv with a newer, vLLM-compatible stack.
# Run ONCE on a Leonardo LOGIN node. This does NOT touch your main venv.
#
# Why separate: your main venv is pinned to the legacy GRPO stack
# (trl 0.16 / transformers <4.50 / older torch) for compatibility. vLLM needs
# newer torch + transformers, which in turn want trl >=0.17. Those newer
# versions are mutually coherent (new torch has FSDPModule; new trl matches the
# new transformers sampler API), so the conflicts we hit earlier do not apply
# here -- but it must live in its own environment.
#
#   bash slurm/setup_vllm.sh
#   sbatch slurm/curriculum_vllm.slurm
#
# Expect to iterate on exact versions: vLLM<->trl<->transformers compatibility
# moves fast. If a version conflict appears, paste it and pin accordingly.
# =============================================================================
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$WORK/grpo-reasoning}"
export VENV_DIR="${VENV_DIR:-$WORK/venvs/grpo-vllm}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.6--gcc--8.5.0}"

module purge
module load "$PYTHON_MODULE"

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# 1) vLLM pulls a compatible torch + transformers automatically.
pip install vllm

# 2) trl with vLLM support (colocate mode); compatible with vLLM's transformers.
pip install "trl>=0.18"

# 3) Project CODE only -- --no-deps so the legacy pins in pyproject do NOT
#    downgrade the vLLM-compatible torch/transformers/trl just installed.
cd "$PROJECT_ROOT"
pip install -e . --no-deps

# 4) The remaining runtime deps that don't conflict with vLLM.
pip install datasets accelerate peft "numpy<2.0" matplotlib pyyaml tqdm bitsandbytes
pip install "moleculariq-core @ git+https://github.com/ml-jku/moleculariq-core.git"

# 5) Sanity check: imports + CUDA.
python - <<'PY'
import torch, trl, transformers, vllm
print(f"[vllm-setup] torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[vllm-setup] trl={trl.__version__} transformers={transformers.__version__} vllm={vllm.__version__}")
PY

echo "[vllm-setup] DONE. Submit with: sbatch slurm/curriculum_vllm.slurm"
