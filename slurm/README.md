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
`HF_HOME`, `PROJECT_ROOT`, `PYTHON_MODULE` env vars. Check the Python module name
first with `module avail python`.

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
