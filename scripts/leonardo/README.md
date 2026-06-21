# Running on CINECA Leonardo (Booster)

`train_multitask_grpo.slurm` runs the multitask GRPO configs on a Leonardo
Booster node (4x A100 64GB) with **vLLM colocate** generation. This is the
setup the project README's "Scaling up later" section refers to.

## Why vLLM here

The multitask train configs now use `num_generations: 8` and
`per_device_train_batch_size: 8`. A larger GRPO group gives lower-variance
advantage estimates and fewer all-same-reward (zero-gradient) groups, but it
multiplies generation cost. `model.generate` would make that painful, so the
configs enable vLLM:

```yaml
grpo_overrides:
  use_vllm: true
  vllm_mode: colocate
  scale_rewards: false
```

`vllm_mode: colocate` runs a vLLM engine on each training GPU, so the launcher
just starts one process per GPU with `accelerate`.

## Batch / group divisibility

TRL requires `num_generations` to evenly divide the effective generation batch.
With `per_device_train_batch_size = 8` and `gradient_accumulation_steps = 4`:

| GPUs (`--gres=gpu:N`) | per_device x N | global (x grad_accum) | divisible by 8 |
|-----------------------|----------------|-----------------------|----------------|
| 1                     | 8              | 32                    | yes            |
| 2                     | 16             | 64                    | yes            |
| 4                     | 32             | 128                   | yes            |

So you can request 1, 2, or 4 GPUs without changing the config. If you raise
`num_generations` to 16, request >=2 GPUs (or set `per_device_train_batch_size:
16`) so it still divides evenly.

## One-time setup

1. **Build a venv** with a CUDA torch matching the `cuda/*` module you load,
   then install the project plus vLLM:

   ```bash
   module load cuda/12.2
   python -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt vllm
   ```

2. **Pre-cache the model + dataset on a login node** (Booster compute nodes are
   offline). Set `HF_HOME` to the same path the job uses:

   ```bash
   export HF_HOME=$PWD/.hf_cache
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
   PY
   ```

3. **Preprocess datasets on the login node** (writes to `data/`, needs network):

   ```bash
   python scripts/multitask/preprocess_multitask.py \
       --config configs/multitask/miq_multitask_pooled.yaml
   ```

4. **Edit the `<PLACEHOLDERS>`** in `train_multitask_grpo.slurm`: `--account`,
   the `VENV` path, and the `cuda/*` module version.

## Submit

```bash
# Default (pooled):
sbatch scripts/leonardo/train_multitask_grpo.slurm

# A specific config:
sbatch scripts/leonardo/train_multitask_grpo.slurm \
    configs/multitask/miq_multitask_balanced_train.yaml
```

For a quick smoke test, grab a debug slot (uncomment `#SBATCH --qos=boost_qos_dbg`)
and cap steps from the CLI, e.g. add `--max-steps 50` to the `scripts/train.py`
line, or run it directly inside an `srun --gres=gpu:1 --pty bash` session.
