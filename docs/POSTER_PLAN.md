# Thesis poster plan — verifier-reward GRPO for chemistry reasoning

A focused, 48-hour-feasible plan for the poster. It uses the existing
`multi-task-chemistry-rl` GRPO pipeline and Qwen2.5-0.5B-Instruct on the
MolecularIQ symbolic tasks. Everything below runs on Leonardo with single-A100
jobs that can be submitted in parallel.

## The one-sentence story

> A symbolic **verifier** turns MolecularIQ into a reinforcement-learning
> environment, and **GRPO** teaches a small (0.5B) LLM to reason over molecules
> — but only when the verifier hands back a **dense** reward, because GRPO learns
> from *within-group reward variance* and exact-match rewards collapse that
> variance to zero on hard tasks.

## Hypotheses

### H1 (headline / sanity) — GRPO with a verifier reward beats the base model
Verifier-reward GRPO lifts Qwen2.5-0.5B-Instruct above its untrained self on
MolecularIQ counting, atom-indexing, and constraint-generation tasks.
*Why it matters:* establishes that the symbolic-verifier RL setup works at all
on a small model, and quantifies the lift.

### H2 (the differentiator) — Dense rewards are necessary; sparse rewards stall
Holding everything else fixed, GRPO with a **dense, verifier-shaped** reward
learns the hard *set-valued* tasks (atom indices, constraint generation),
whereas GRPO with a **sparse exact-match** reward stalls near the base model on
those tasks while still improving on the easy low-cardinality count tasks.

*Mechanism (this is the GRPO-specific insight):* GRPO replaces a value critic
with a **group-relative advantage** — for each prompt it samples G completions
and standardizes their rewards, `A_i = (r_i − mean(r)) / std(r)`. On a hard
exact-match task almost every sampled completion scores 0, so within a group
`std(r) ≈ 0` and every advantage `≈ 0`: **no gradient flows**. A dense reward
(numeric closeness for counts, **Jaccard overlap** for index sets, per-constraint
satisfaction + valid-SMILES credit for generation) restores intra-group reward
variance, producing informative advantages and letting the policy improve. This
is exactly what the verifier's `return_details=True` partial credit is for.

## Why this design is clean

- **One-flag ablation.** `configs/multitask/miq_h2_shaped_train.yaml` and
  `miq_h2_sparse_train.yaml` are byte-for-byte identical except
  `use_shaped_moleculariq_reward` (true vs false). Same data, same init (base
  model, *not* a warm-started checkpoint), same steps, same seed, same decoding.
  Any difference is attributable to the reward signal alone.
- **Shared runs.** The shaped arm *is* the H1 generalist, so H1 and H2 cost two
  training runs total, not three.
- **Honest evaluation.** Both arms are scored by the **same official verifier**
  used in training; exact-match `accuracy` is the headline, and the verifier's
  `partial_score_mean` (Jaccard for index tasks) is reported alongside so the
  poster can show the shaped model is "mostly right" even where exact-match is
  harsh. The base model is auto-evaluated as the reference in every figure.

## Run it on Leonardo (the 48-hour runbook)

All from `$WORK/grpo-reasoning` after the one-time `bash slurm/setup_leonardo.sh`.

```bash
# Launch everything: preprocess (login node) + 2 parallel training arms +
# a dependent eval/report/poster job. ~one overnight window end-to-end.
bash slurm/run_h2.sh

# If the queue is tight, shorten training (still shows the divergence):
STEPS=1000 bash slurm/run_h2.sh
```

`run_h2.sh` prints the three job IDs and the queue. When the eval job finishes,
the poster artifacts are in `outputs/poster/`.

**Faster generation (optional but recommended if it works):** generation, not
backprop, dominates GRPO wall-clock with `model.generate`. If your vLLM venv from
`slurm/setup_vllm.sh` is healthy, enabling vLLM cuts training to ~1–2 h/arm
(`use_vllm: true`, `vllm_mode: colocate` in the train YAMLs). Treat this as a
nice-to-have; the plain path already fits 48 h.

**Manual equivalent** (if you prefer to drive jobs yourself):

```bash
grpo-preprocess-multitask --config configs/multitask/miq_multitask_balanced.yaml   # login node
VARIANT=shaped sbatch slurm/h2_reward_ablation.slurm
VARIANT=sparse sbatch slurm/h2_reward_ablation.slurm
sbatch --dependency=afterok:<shaped_id>:<sparse_id> slurm/h2_eval_report.slurm
```

## Figures (auto-generated into `outputs/poster/`)

| File | Hypothesis | What it shows |
|------|-----------|---------------|
| `fig_h1_lift.png` | H1 | base vs GRPO macro & worst-task accuracy, plus lift by task family |
| `fig_h2_by_tasktype.png` | H2 | base vs sparse vs shaped accuracy per task family; hard set-valued families shaded — sparse stays flat there, shaped climbs |
| `fig_h2_training.png` | H2 | mean group reward and KL-to-reference over steps for both arms — sparse reward plateaus, shaped rises |
| `fig_h2_partial_vs_exact.png` | H2 | shaped model on index tasks: exact-match vs verifier Jaccard partial credit (the model is mostly right) |
| `poster_headline_numbers.csv` | — | macro / worst-task / partial-score per arm for the abstract |

Regenerate figures any time (e.g. after a longer run) without re-evaluating:

```bash
python scripts/multitask/make_poster_figures.py \
    --eval-dir outputs/multitask_eval --outputs-dir outputs --out-dir outputs/poster
```

## What you can claim (fill the brackets from the CSV)

- **H1:** "Verifier-reward GRPO raises macro exact-match accuracy from [base]%
  to [shaped]% on a 0.5B model, with the largest gains on [task family]."
- **H2:** "With an identical setup, sparse exact-match GRPO reaches only
  [sparse-index]% on set-valued index tasks vs [shaped-index]% for the dense
  reward — consistent with vanishing group-relative advantages when exact-match
  variance collapses. Counts, which retain reward variance, improve under both."

## Honest caveats (good to pre-empt on the poster)

- Single seed (42) and one model size (0.5B); results are an existence proof of
  the mechanism, not a scaling study. (If a second GPU window is free, a second
  seed for each arm turns the bars into mean±range — cheap robustness.)
- Index tasks are restricted to tractable molecules (length / list-length
  filters) applied identically to train and eval, so all arms are scored on the
  same distribution.
- `distinct_answer_rate` is a collapse signal for *index* tasks only; counts and
  constraint generation have tiny discrete target spaces, so read exact-match
  accuracy there.

## Optional extension if a run finishes early (H4 — curriculum)

A counts→index→generation curriculum vs a flat multitask run at equal compute
is one extra chained job (`sbatch slurm/curriculum.slurm`, then evaluate with
`--model-label curriculum`). The poster figures and report pick it up
automatically once its `summary.json` exists. Only add this if H1+H2 are solid
first.

## Poster skeleton

- **Title:** *Dense verifiers, not just rewards: what GRPO needs to teach a small
  LLM molecular reasoning.*
- **Background:** MolecularIQ symbolic tasks; GRPO = critic-free, group-relative
  RL; the verifier gives both a binary verdict and dense partial credit.
- **Method:** Qwen2.5-0.5B-Instruct, balanced multitask GRPO, one-flag reward
  ablation, official-verifier scoring.
- **Results:** `fig_h1_lift` (it works) → `fig_h2_by_tasktype` (dense needed on
  hard tasks) → `fig_h2_training` (reward/KL trajectories) → `fig_h2_partial_vs_exact`
  (mostly-right).
- **Takeaway:** For verifier-based RL on small models, *reward density* — not just
  reward correctness — determines which skills are learnable, because GRPO's
  group-relative advantage needs within-group reward variance.
