# grpo-reasoning

Multitask GRPO fine-tuning of **Qwen2.5-0.5B-Instruct** using **TRL** for
MolecularIQ chemistry reasoning
([`ml-jku/moleculariq-trainPool`](https://huggingface.co/datasets/ml-jku/moleculariq-trainPool)).

The project compares ways to train one chemistry reasoning model across several
MolecularIQ subtasks, including pooled training, balanced task sampling, and
adaptive task sampling. Single-task MolecularIQ runs are kept only as specialist
baselines for comparison and later distillation experiments.

## Repository layout

The code is split into two pipelines that share a common core. Pick the folder
that matches what you want to do; the shared building blocks live in `common`.

```text
src/grpo_reasoning/
  common/                  Shared by both pipelines
    prompts.py             R1-style system prompt + <answer> extraction
    rewards.py             Format reward + correctness/shaped MolecularIQ rewards
    tasks/                 Task registry: base ABC + moleculariq
    train.py               GRPO training (auto-detects single vs multitask data)
    eval.py                Greedy eval + accuracy + per-sample JSON
    utils.py               Seeding, YAML loader, tokenizer
    cli.py                 grpo-train entry point (shared)
  single_task/             Specialist (one task) pipeline
    data.py                Preprocess one task -> save_to_disk
    plotting.py            Training curves + baseline-vs-trained bar
    cli.py                 preprocess / evaluate / plot-training entry points
  multitask/               Multitask pipeline
    dataset.py             Pooled/balanced/adaptive dataset builders
    curriculum.py          Staged curriculum training
    audit.py               MolecularIQ source-pool coverage audit
    reporting.py           Cross-run comparison plots + CSV tables
    cli.py                 preprocess-multitask / evaluate-multitask /
                           curriculum / audit / report entry points

configs/
  single_task/             Specialist train configs + the base model config
  multitask/               Multitask, curriculum, and audit-suite configs

scripts/
  train.py                 Shared training wrapper
  single_task/             Wrappers + automated specialist comparison scripts
  multitask/               Wrappers + dataset audit / report scripts

tests/                     Reward and multitask sanity tests
```

## Why this layout

- **No vLLM** - Windows-friendly; uses `model.generate`. Set `use_vllm=true` later when
  moving to a Linux machine with more VRAM.
- **No wandb** - Trainer writes `trainer_state.json`; we parse it with matplotlib.
- **Preprocess once, train on cached data** - as your supervisor recommended.
  The preprocess step builds a complete HF dataset (with `prompt` and `answer`
  columns) and saves it to disk; the trainer just loads it.
- **Task abstraction** - `common/tasks/base.py` defines a tiny `Task` ABC.
  MolecularIQ task variants generate `question` + `answer` rows, then shared
  prompt, reward, training, eval, and plotting code handles the rest.
- **Multitask metadata** - mixed datasets carry `task_id`, `task_type`, and
  `properties` columns so rewards and evaluation can dispatch per example.

## Setup

```powershell
# 1) Install PyTorch with CUDA matching your driver. Example: CUDA 12.6
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 2) Install the project runtime, chemistry, optimizer, and test extras
pip install -r requirements.txt
```

The configs use `optim: adamw_torch` by default for broad compatibility.
`bitsandbytes` (the `[bnb]` extra) provides an optional 8-bit AdamW optimizer;
where it loads cleanly you can set `optim: adamw_8bit` in the YAML to save
optimizer memory. It will not load on platforms with glibc < 2.34 (e.g. RHEL 8)
or on Windows.

After installation the workflows are available either through the script
wrappers shown below or through console commands:

| Console command | Module |
|-----------------|--------|
| `grpo-train` | shared |
| `grpo-preprocess`, `grpo-evaluate`, `grpo-plot-training` | single-task |
| `grpo-preprocess-multitask`, `grpo-evaluate-multitask`, `grpo-curriculum`, `grpo-audit-moleculariq`, `grpo-report` | multitask |

If you OOM, reduce `max_completion_length` first, then `num_generations`, then
`per_device_train_batch_size`.

For Leonardo, prefer the dedicated setup script instead of the generic commands
above:

```bash
bash slurm/setup_leonardo.sh
```

It creates a clean venv in `$WORK/venvs/grpo`, installs the constrained
TRL/Transformers stack from `slurm/constraints-leonardo.txt`, installs PyTorch
from the configured CUDA wheel index, runs `pip check`, and pre-caches the model
and datasets for offline compute nodes.

## Pause and resume training

Training writes regular `checkpoint-*` directories according to `save_steps`.
You can pause a run with `Ctrl+C`; the trainer will try to save an interrupt
checkpoint before exiting. Training is shared, so `scripts/train.py` is used for
both pipelines.

Resume from the latest checkpoint in the configured output directory:

```powershell
python scripts/train.py --config configs/multitask/miq_multitask_pooled_train.yaml `
    --resume-from-checkpoint
```

Resume from a specific checkpoint:

```powershell
python scripts/train.py --config configs/multitask/miq_multitask_pooled_train.yaml `
    --resume-from-checkpoint outputs/miq-multitask-pooled-grpo/checkpoint-3200
```

If you want `Ctrl+C` to stop immediately without saving an extra checkpoint:

```powershell
python scripts/train.py --config configs/multitask/miq_multitask_pooled_train.yaml `
    --no-save-on-interrupt
```

## Specialist (single-task) chemistry runs

`ml-jku/moleculariq-trainPool` ships SMILES + metadata. Questions and ground-truth
answers are generated via
[`moleculariq_core.MolecularIQD`](https://github.com/ml-jku/moleculariq-core) and
cached to disk during preprocessing. The training loop never calls MolecularIQD.

Pick one task variant per specialist run via `--task-type` and the property via
`--properties`.

The 16-task comparison suite is listed in
`configs/multitask/miq_experiment_suite.yaml`. There is also a static specialist
training YAML for every task, named `configs/single_task/miq_<task_id>.yaml`. The
automated specialist comparison script can regenerate those YAMLs from
`configs/single_task/moleculariq_qwen05b.yaml`, but the static files make manual
runs and future distillation work easier to inspect.

To inspect how many full-pool examples each configured task can generate:

```powershell
# Quick smoke test on only the first 1000 raw rows
python scripts/multitask/inspect_moleculariq_dataset.py --max-raw-rows 1000

# Full MolecularIQ trainPool audit for the current 16-task suite
python scripts/multitask/inspect_moleculariq_dataset.py
```

The audit writes `summary.json`, `task_counts.csv`, `task_type_counts.csv`,
`property_counts.csv`, and `property_stats.csv` to
`outputs/moleculariq_dataset_audit`. This helps check whether pooled and
balanced training are genuinely different, or whether per-task sample caps make
them effectively equal.

| `--task-type`            | What the model must return inside `<answer>` |
|--------------------------|----------------------------------------------|
| `single_count`           | JSON like `{"ring_count": 3}`                |
| `multi_count`            | JSON dict with multiple count keys           |
| `single_index`           | JSON like `{"ring_index": [0, 1, 2]}`        |
| `multi_index`            | JSON dict with multiple index lists          |
| `constraint_generation`  | JSON like `{"smiles": "CCO"}`                |

```powershell
# 1) Build the cached HF dataset for one specialist
python scripts/single_task/preprocess.py --task moleculariq --split train `
    --task-type single_count --properties ring_count `
    --num-samples 5000 --out data/miq_sc_ring_count_train

# 2) Make sure moleculariq_task_type in the YAML matches --task-type above
python scripts/train.py --config configs/single_task/miq_sc_ring_count.yaml

# 3) Plot training curves
python scripts/single_task/plot_training.py --output-dir outputs/miq-sc_ring_count-grpo

# 4) Evaluate with matching task parameters
python scripts/single_task/evaluate.py --task moleculariq `
    --task-type single_count --properties ring_count `
    --baseline Qwen/Qwen2.5-0.5B-Instruct `
    --trained  outputs/miq-sc_ring_count-grpo `
    --num-samples 200
```

Switching tasks is a config and preprocessing change, for example
`--task-type single_count --properties aromatic_ring_count` or
`--task-type single_index --properties carbon_atom_index`. Valid property names
live in `moleculariq_core.properties` (`COUNT_MAP`, `INDEX_MAP`,
`CONSTRAINT_MAP`).

Scoring uses `moleculariq_core.evaluate_answer`, which tolerates property-name
aliases and JSON formatting variations; the format reward still enforces the
R1 `<reasoning>...</reasoning>\n<answer>...</answer>` scaffold.

Training also includes a shaped MolecularIQ reward by default. Exact correctness
is still rewarded separately, but count tasks get numeric-closeness partial
credit, index tasks get atom-set overlap credit, and constraint-generation tasks
get valid-SMILES credit plus property-closeness credit for supported RDKit
properties.

### Automated specialist comparison

`scripts/single_task/auto_train_compare.py` trains one model per task
configuration and evaluates each against its own task, and
`scripts/single_task/cross_generalization.py` builds the full (model × eval-task)
transfer matrix. See each script's module docstring for usage.

## Multitask chemistry runs

The multitask path trains one model on several MolecularIQ subtasks at once.
Datasets are still preprocessed first, but each row now carries `task_id`,
`task_type`, and `properties` metadata so the reward function can dispatch to
the right MolecularIQ scorer per example.

The default suite contains 16 subtasks across single-count, multi-count,
single-index, multi-index, and constraint-generation variants. It covers ring
topology, aromaticity, composition, H-bond acceptors, rotatable bonds, and
atom-index attribution.

Three dataset strategies are included:

| Strategy | Config | Meaning |
|----------|--------|---------|
| `pooled` | `configs/multitask/miq_multitask_pooled.yaml` | Concatenate all selected task datasets and shuffle. |
| `balanced` | `configs/multitask/miq_multitask_balanced.yaml` | Sample the same number of examples from each task. |
| `adaptive` | `configs/multitask/miq_multitask_adaptive.yaml` | Sample by task weights, optionally computed from a previous multitask eval summary so weak tasks get more data. |

Example pooled run:

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_multitask_pooled.yaml
python scripts/train.py --config configs/multitask/miq_multitask_pooled_train.yaml
python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_pooled.yaml `
    --model outputs/miq-multitask-pooled-grpo --model-label pooled
```

Balanced run:

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_multitask_balanced.yaml
python scripts/train.py --config configs/multitask/miq_multitask_balanced_train.yaml
python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_balanced.yaml `
    --model outputs/miq-multitask-balanced-grpo --model-label balanced
```

Adaptive run:

```powershell
# First evaluate a previous model, for example the balanced one.
python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_balanced.yaml `
    --model outputs/miq-multitask-balanced-grpo --model-label balanced

# Then rebuild the adaptive dataset. The default adaptive config reads
# outputs/multitask_eval/balanced/summary.json and oversamples lower-accuracy tasks.
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_multitask_adaptive.yaml --overwrite
python scripts/train.py --config configs/multitask/miq_multitask_adaptive_train.yaml
python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_adaptive.yaml `
    --model outputs/miq-multitask-adaptive-grpo --model-label adaptive
```

`scripts/multitask/evaluate_multitask.py` writes per-task JSON files plus a
`summary.json` with macro accuracy and worst-task accuracy. That summary is the
handoff point for adaptive sampling and for later comparison scripts.

Create a visual report across all discovered train/eval outputs:

```powershell
python scripts/multitask/plot_experiment_report.py --outputs-dir outputs
```

By default this reads trainer logs from `outputs/*/trainer_state.json` or the
latest checkpoint under each output folder, reads model summaries from
`outputs/multitask_eval/*/summary.json`, and writes plots plus CSV tables to
`outputs/report`. The useful files are:

- `overall_metrics.png`: macro, worst-task, and partial-score comparison.
- `per_task_accuracy_heatmap.png`: which model wins or fails on each subtask.
- `task_type_accuracy.png`: average accuracy by task family.
- `training_loss.png`, `training_reward.png`, `training_kl.png`: overlaid train
  curves across runs.
- `eval_overall.csv`, `eval_by_task.csv`, `training_final_metrics.csv`: tables
  for thesis notes or spreadsheet analysis.

New evaluation files also include diagnostic partial metrics:

- `partial_score_mean`: shaped partial score before exact-match thresholding.
- `answer_present_rate`: fraction of completions with `<answer>...</answer>`.
- `json_valid_rate`: fraction of extracted answers that parse as JSON.
- `valid_smiles_rate`: fraction with parseable generated SMILES, mainly useful
  for `constraint_generation`.

## Curriculum training

Curriculum follows the same preprocess/train/eval pattern as the other
approaches, but each stage has its own dataset and train config. Later train
configs set `model_name` to the previous stage output directory.

Stage 1: count foundations

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_curriculum_01_counts.yaml
python scripts/train.py --config configs/multitask/miq_curriculum_01_counts_train.yaml
```

Stage 2: simple index tasks with count replay

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_curriculum_02_simple_index.yaml
python scripts/train.py --config configs/multitask/miq_curriculum_02_simple_index_train.yaml
```

Stage 3: full index tasks with replay

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_curriculum_03_full_index.yaml
python scripts/train.py --config configs/multitask/miq_curriculum_03_full_index_train.yaml
```

Stage 4: constraint generation with replay

```powershell
python scripts/multitask/preprocess_multitask.py --config configs/multitask/miq_curriculum_04_generation.yaml
python scripts/train.py --config configs/multitask/miq_curriculum_04_generation_train.yaml
```

Evaluate the final curriculum model against the full suite:

```powershell
python scripts/multitask/evaluate_multitask.py --config configs/multitask/miq_multitask_balanced.yaml `
    --model outputs/miq-curriculum-04-generation --model-label curriculum
```

For a quick smoke test, run each stage with a temporary output directory and a
small step cap, for example:

```powershell
python scripts/train.py --config configs/multitask/miq_curriculum_01_counts_train.yaml `
    --max-steps 50 --output-dir outputs/miq-curriculum-01-smoke
```

The default curriculum stages are:

| Stage | Focus |
|-------|-------|
| `01_counts` | Single-count and multi-count foundations. |
| `02_simple_index` | Shorter SMILES and shorter index lists, plus count replay. |
| `03_full_index` | Full index tasks and multi-index with count replay. |
| `04_generation` | Constraint generation with count/index replay. |

`scripts/multitask/run_curriculum.py --config configs/multitask/miq_curriculum.yaml`
is still available as a convenience wrapper that runs all stages in order, but
the explicit configs above are the recommended experiment interface.

## Scaling up later

When you move to a larger Linux box with enough VRAM:

- Set `use_vllm: true` in `grpo_overrides` in the YAML, plus `vllm_mode: colocate`.
- Bump `num_generations` and `per_device_train_batch_size`.
- Optionally add `attn_implementation: flash_attention_2` in `model_init_kwargs`.

## Tests

```powershell
pytest -q
```
