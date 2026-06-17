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

## Notes / gotchas

- **Offline mode**: jobs export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`. If you see a download error, the model/dataset wasn't
  cached on the login node — re-run `setup_leonardo.sh`.
- **bitsandbytes**: the configs use `optim: adamw_8bit`. If bnb misbehaves,
  install without it (`pip install -e ".[chem]"`) and change `optim` to
  `adamw_torch` in the `*_train.yaml` configs.
- **True adaptive sampling** needs the balanced eval `summary.json` to exist,
  then the adaptive dataset must be rebuilt on a login node (internet) before
  retraining — see the comment in `strategies.slurm`.
