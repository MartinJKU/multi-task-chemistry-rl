# Running on Leonardo (CINECA)

Leonardo Booster nodes have 4× A100 64GB. This 0.5B GRPO project needs **one
GPU**. The key constraint: **compute nodes have no internet**, so all downloads
happen on the **login node** during setup, and training jobs run offline.

## 1. One-time setup (on a LOGIN node — has internet)

```bash
ssh mneumeis@login.leonardo.cineca.it
cd $WORK
git clone -b claude/amazing-ritchie-tbn0o3 <repo-url> grpo-reasoning
cd grpo-reasoning
bash slurm/setup_leonardo.sh        # venv + deps + pre-download model & datasets
```

`setup_leonardo.sh` puts the venv in `$WORK/venvs/grpo` and the HuggingFace
cache in `$WORK/hf_cache` (not `$HOME` — quota). Override via `VENV_DIR`,
`HF_HOME`, `PIP_CACHE_DIR`, `PROJECT_ROOT`, `PYTHON_MODULE`,
`PYTORCH_VERSION`, and `PYTORCH_INDEX_URL` env vars. Check the Python module name
first with `module avail python`.

By default the script removes an existing `$WORK/venvs/grpo` first
(`CLEAN_VENV=1`) so the install is genuinely fresh, then installs:

- PyTorch `2.7.0` from `https://download.pytorch.org/whl/cu126`
- the pinned TRL/Transformers stack in `slurm/constraints-leonardo.txt`
- `moleculariq-core`, `bitsandbytes`, and the project package

If the compute-node NVIDIA driver is too old for the CUDA 12.6 wheel, rebuild
with an older PyTorch wheel index, for example:

```bash
PYTORCH_VERSION=2.5.1 \
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
bash slurm/setup_leonardo.sh
```

The setup ends with `pip check` and prints the installed versions. On login
nodes `torch.cuda.is_available()` may be false; the SLURM scripts assert CUDA
availability on the actual GPU compute node before training starts.

## 2. Set your account

Find your project budget:

```bash
saldo -b
```

Put it in the `#SBATCH --account=` line of the `.slurm` files (replace
`YOUR_ACCOUNT`).

## 3. Submit a job

Two independent options:

```bash
# Path B — the staged curriculum (one model: counts -> index -> generation)
sbatch slurm/curriculum.slurm

# Path A — the adaptive/balanced/pooled comparison (your heatmap),
#          now warm-started from counts + index difficulty filters
sbatch slurm/strategies.slurm
```

`curriculum.slurm` uses the **A100-tuned** base config
(`configs/multitask/miq_multitask_a100_train.yaml`): bigger batch, more
generations, and no gradient checkpointing, so generation runs many completions
in parallel and actually uses the A100 (a small-batch 0.5B GRPO run is
generation-latency-bound and otherwise no faster than a desktop GPU). Drop
`per_device_train_batch_size` to 32 in that config if you hit OOM.

### Optional: vLLM generation (experimental, biggest speedup)

vLLM makes generation 5-20x faster but needs a newer torch/transformers/trl than
the pinned base stack, so it lives in a **separate venv**:

```bash
bash slurm/setup_vllm.sh           # one-time, builds $WORK/venvs/grpo-vllm
sbatch slurm/curriculum_vllm.slurm # single GPU, vLLM colocates with training
```

This is best-effort scaffolding — vLLM<->trl<->transformers versions move fast,
so expect to pin a version or two on first run (paste any conflict and adjust).
The non-vLLM A100 path above is the reliable default.

Smoke test first (cheap, <=30 min, debug QoS): edit the job to
`--qos=boost_qos_dbg`, `--time=00:30:00`, and append
`--max-steps-per-stage 20` to the `grpo-curriculum` line.

## 4. Monitor

```bash
squeue -u $USER
tail -f logs/miq-curriculum-<jobid>.out
```

Outputs land in `outputs/` (checkpoints) and `outputs/multitask_eval/<label>/summary.json`
(per-task metrics). For the weak tasks, watch `partial_score_mean` on `si_*`/`mi_*`
rise above ~0 — that's the real signal at 0.5B, since the heatmap only shows
exact-match accuracy.

## Running Qwen3.5-0.8B next to the 0.5B checkout (separate modern venv)

You can compare `Qwen/Qwen3.5-0.8B` against the existing 0.5B runs. The Qwen3
jobs reuse the same `$WORK/hf_cache` and the **same datasets** (datasets are
model-independent) and write to `*-qwen3` output directories, so nothing the
0.5B runs produced is overwritten.

The catch: Qwen3.5 reports `model_type: qwen3_5`, which the pinned legacy stack
(`transformers==4.49`, `trl==0.16`) cannot load — and that pin is load-bearing
for the 0.5B GRPO trainer, so we must **not** upgrade the main venv. Instead the
Qwen3 runs use a **separate venv** with a modern transformers/trl, exactly the
pattern `setup_vllm.sh` uses. The shared training code is version-robust (it
filters `GRPOConfig` kwargs to what the installed trl accepts and picks the right
`dtype`/`torch_dtype` kwarg), so the same `grpo-train`/`grpo-curriculum` commands
run under either venv.

One-time, on an internet-connected **login node**:

```bash
cd $WORK/grpo-reasoning
git fetch origin claude/stoic-tesla-1hfhyu
git checkout claude/stoic-tesla-1hfhyu      # or merge/cherry-pick the configs

# Build $WORK/venvs/grpo-qwen3 (modern transformers+trl), verify the qwen3_5
# arch is recognized, and pre-cache the model into the shared HF cache.
bash slurm/setup_qwen3.sh
```

This does not touch your main `$WORK/venvs/grpo` venv or its results. Then submit
the Qwen3 jobs — they default `VENV_DIR` to `$WORK/venvs/grpo-qwen3` and can queue
alongside the 0.5B jobs:

```bash
sbatch slurm/strategies_qwen3.slurm   # pooled/balanced/adaptive on Qwen3
sbatch slurm/curriculum_qwen3.slurm   # full staged curriculum on Qwen3
```

Set the `#SBATCH --account=` line in those two files the same way as the others.
`grpo-report` discovers every eval summary under `outputs/multitask_eval`, so the
final report shows the 0.5B and Qwen3 runs side by side.

Notes specific to this larger / newer model:

- **Expect to pin a version or two on the first `setup_qwen3.sh` run.**
  transformers<->trl compatibility moves fast; the script installs the latest of
  each by default. If `grpo-train` errors on a `GRPOConfig`/`GRPOTrainer` API
  change, pin via `TRANSFORMERS_SPEC=... TRL_SPEC=... bash slurm/setup_qwen3.sh`.
  The version-robust shims in `common/train.py` cover kwarg/dtype renames, not a
  wholesale trainer-API rewrite.
- If the **exact HF repo id differs** from `Qwen/Qwen3.5-0.8B`, set it once:
  `QWEN3_MODEL=Qwen/<id> bash slurm/setup_qwen3.sh`, and update the `model_name` /
  `base_model` line in the six `*qwen3*` configs.
- `curriculum_qwen3.slurm` reuses the A100-tuned base config
  (`per_device_train_batch_size: 32`). The 0.8B model uses more memory than the
  0.5B one, so if you hit OOM drop that to 16 in
  `configs/multitask/miq_multitask_a100_train.yaml` (or point
  `--base-train-config` at a copy with the smaller batch).
- Smoke test first: edit the job to `--qos=boost_qos_dbg`, `--time=00:30:00`,
  and append `--max-steps-per-stage 20` to the `grpo-curriculum` line.

> `slurm/cache_model_leonardo.sh` only works for models the *legacy* stack can
> load (e.g. another Qwen2.5 checkpoint). It cannot cache `qwen3_5` — use
> `setup_qwen3.sh` for that.

## Notes / gotchas

- **Offline mode**: jobs export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`. If you see a download error, the model/dataset wasn't
  cached on the login node — re-run `setup_leonardo.sh` (or `setup_qwen3.sh` for
  the Qwen3 venv/model).
- **bitsandbytes / 8-bit optimizer**: NOT used on Leonardo. bitsandbytes 0.45.x
  ships a CUDA library linked against `GLIBC_2.34`, but the RHEL 8 nodes only
  provide glibc 2.28, so it cannot load and the 8-bit optimizer crashes at
  `optimizer.step()`. The configs therefore use `optim: adamw_torch` (full
  AdamW; for the 0.5B model the extra optimizer memory is negligible on an
  A100). If you have a node with glibc >= 2.34 and want 8-bit, install
  `pip install -e ".[chem,bnb]"` and set `optim: adamw_8bit`.
- **True adaptive sampling** needs the balanced eval `summary.json` to exist,
  then the adaptive dataset must be rebuilt on a login node (internet) before
  retraining — see the comment in `strategies.slurm`.
