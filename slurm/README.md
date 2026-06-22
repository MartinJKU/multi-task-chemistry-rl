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

### Running 0.5B and 1.5B in parallel

Each job only uses **one GPU**, and a Booster node has 4× A100, so you can run a
1.5B curriculum alongside the 0.5B one — they are independent SLURM jobs that the
scheduler runs concurrently (subject to your QoS GPU/job limits and budget):

```bash
sbatch slurm/curriculum.slurm        # 0.5B -> outputs/miq-curriculum-*
sbatch slurm/curriculum_1.5b.slurm   # 1.5B -> outputs/miq-curriculum-15b-*
```

The 1.5B job (`curriculum_1.5b.slurm`) uses `miq_curriculum_1.5b.yaml`
(`base_model: Qwen/Qwen2.5-1.5B-Instruct`) and the
`miq_multitask_a100_1.5b_train.yaml` base config, which keeps the same recipe but
a smaller `per_device_train_batch_size` (16) since 1.5B is ~3× the parameters.
Its checkpoints go to `outputs/miq-curriculum-15b-*`, so nothing collides with
the 0.5B run, and it **reuses the same prebuilt datasets** (datasets are
model-agnostic). The 1.5B model is cached by `setup_leonardo.sh` alongside the
0.5B one — if you set it up before this was added, re-run `setup_leonardo.sh`
(or just `MODELS="Qwen/Qwen2.5-1.5B-Instruct" bash slurm/setup_leonardo.sh` is
not enough since it rebuilds the venv; on a login node simply
`python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; m='Qwen/Qwen2.5-1.5B-Instruct'; T.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"`
with `HF_HOME=$WORK/hf_cache` exported). If you OOM on the 1.5B run, drop
`per_device_train_batch_size` to 8 in `miq_multitask_a100_1.5b_train.yaml`.

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

## 5. Evaluate

`grpo-evaluate-multitask` scores a trained model on a suite **and** evaluates a
base model once as a cached `baseline` reference, so the report shows lift over
the untrained model. Eval regenerates the test set live from the cached SMILES
pool, so it runs offline on a GPU node.

```bash
# 0.5B run: target = curriculum, baseline = Qwen2.5-0.5B (the default)
sbatch slurm/evaluate.slurm

# 1.5B run: target = curriculum_15b, baseline = Qwen2.5-1.5B
PROJECT_ROOT=$WORK/grpo-reasoning-15b sbatch slurm/evaluate_1.5b.slurm
```

For the 1.5B model use `evaluate_1.5b.slurm`: it points at
`outputs/miq-curriculum-15b-04-generation` and sets the baseline to
**Qwen/Qwen2.5-1.5B-Instruct** with a distinct `baseline_15b` label — so its
"lift over base" is apples-to-apples and nothing overwrites the 0.5B
`baseline`/`curriculum` summaries. Point `PROJECT_ROOT` at the folder where the
1.5B run wrote its `outputs/`. Add `EVAL_ALL_STAGES=1` to also score every stage
(forgetting check).

The underlying command, if you'd rather run it directly:

```bash
grpo-evaluate-multitask \
    --config configs/multitask/miq_multitask_balanced.yaml \
    --model outputs/miq-curriculum-15b-04-generation \
    --model-label curriculum_15b \
    --baseline-model Qwen/Qwen2.5-1.5B-Instruct \
    --baseline-label baseline_15b \
    --num-samples 200
```

Each eval writes `outputs/multitask_eval/<label>/summary.json`, and `grpo-report
--outputs-dir outputs` aggregates every summary in that folder into the
heatmap/CSV. To get a single 0.5B-vs-1.5B comparison, run both evals so their
summaries (`curriculum`, `baseline`, `curriculum_15b`, `baseline_15b`) live under
the same `outputs/multitask_eval/`, then run `grpo-report` once.

## Notes / gotchas

- **Offline mode**: jobs export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`. If you see a download error, the model/dataset wasn't
  cached on the login node — re-run `setup_leonardo.sh`.
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
