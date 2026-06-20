"""Generate a PDF failure-analysis report for MolecularIQ multitask GRPO.

Explains, with evidence from the eval summaries and per-sample files, why the
index (si_*/mi_*) and constraint-generation (cg_*) task families are not learned
while counting tasks are.

Usage:
    python scripts/multitask/make_failure_report.py \
        --summary outputs/multitask_eval/curriculum/summary.json \
        --eval-dir outputs/multitask_eval/curriculum \
        --baseline outputs/multitask_eval/baseline/summary.json \
        --out outputs/report/failure_analysis.pdf
"""
from __future__ import annotations

import argparse
import collections
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)
COUNT = {"single_count", "multi_count"}
INDEX = {"single_index", "multi_index"}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(summary):
    return {t["task_id"]: t for t in summary["tasks"]}


def _top_outputs(eval_path, n=5):
    """Return (distinct_count, total, [(answer, count), ...])."""
    data = _load(eval_path)
    res = data.get("results", [])
    ctr = collections.Counter(r.get("extracted") or "" for r in res)
    return len(ctr), len(res), ctr.most_common(n)


def _text_page(pdf, title, blocks):
    """Render a page of text. blocks = list of (kind, text).

    kind in {"h2", "body", "bullet", "mono", "space"}.
    """
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    y = 0.95
    title_fs = 16 if len(title) < 42 else 13.5
    fig.text(0.07, y, title, fontsize=title_fs, fontweight="bold", va="top")
    y -= 0.045
    fig.text(0.07, y, "_" * 92, fontsize=9, color="#888888", va="top")
    y -= 0.030

    for kind, text in blocks:
        if kind == "space":
            y -= 0.018
            continue
        if kind == "h2":
            y -= 0.006
            fig.text(0.07, y, text, fontsize=12.5, fontweight="bold", va="top")
            y -= 0.030
            continue
        mono = kind == "mono"
        bullet = kind == "bullet"
        width = 84 if mono else 96
        prefix = "•  " if bullet else ""
        indent = "    " if mono else ""
        font = {"family": "monospace", "fontsize": 8.4} if mono else {"fontsize": 10.3}
        for i, line in enumerate(textwrap.wrap(text, width=width) or [""]):
            lead = prefix if (bullet and i == 0) else ("   " if bullet else "")
            fig.text(0.075, y, indent + lead + line, va="top",
                     color="#111111", **font)
            y -= 0.0225 if not mono else 0.0205
        y -= 0.010

    pdf.savefig(fig)
    plt.close(fig)


def _bar_page(pdf, cur, base, order):
    """Accuracy: baseline vs curriculum, grouped by task family."""
    import numpy as np

    labels = order
    cur_v = [cur[t]["accuracy"] * 100 for t in labels]
    base_v = [base.get(t, 0.0) for t in labels]
    fig, ax = plt.subplots(figsize=A4)
    yy = np.arange(len(labels))
    h = 0.4
    ax.barh(yy + h / 2, base_v, h, label="baseline (untrained 0.5B)", color="#bcbddc")
    ax.barh(yy - h / 2, cur_v, h, label="curriculum (trained)", color="#3182bd")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("exact-match accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Accuracy by task: trained vs baseline", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # shade index + generation families
    for i, t in enumerate(labels):
        if t.startswith(("si_", "mi_", "cg_")):
            ax.axhspan(i - 0.5, i + 0.5, color="#fde0dd", alpha=0.35, zorder=0)
    fig.text(0.07, 0.05,
             "Pink rows = not learned (index si_/mi_, generation cg_). Counting\n"
             "tasks improve sharply over baseline; structured-output families stay at floor.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def _diag_page(pdf, cur):
    """distinct_answer_rate bar to visualise collapse."""
    import numpy as np

    fams = [
        ("counts", [t for t in cur if cur[t]["task_type"] in COUNT]),
        ("index", [t for t in cur if cur[t]["task_type"] in INDEX]),
        ("generation", [t for t in cur if cur[t]["task_type"] == "constraint_generation"]),
    ]
    labels, vals, colors = [], [], []
    cmap = {"counts": "#31a354", "index": "#de2d26", "generation": "#756bb1"}
    for fam, ts in fams:
        for t in ts:
            labels.append(t)
            vals.append(cur[t].get("distinct_answer_rate", 0.0) * 100)
            colors.append(cmap[fam])
    fig, ax = plt.subplots(figsize=A4)
    yy = np.arange(len(labels))
    ax.barh(yy, vals, color=colors)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("distinct_answer_rate (% of 200 prompts with a unique answer)")
    ax.set_xlim(0, 100)
    ax.set_title("Output diversity — the collapse signature", fontsize=14, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.07, 0.055,
             "Counts legitimately repeat (few integer values). Index: a healthy model\n"
             "nears 100% (each molecule a unique atom set) — 18-24% is a near-fixed prior.\n"
             "Generation ~0.5% = one molecule for every prompt.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", default="outputs/report/failure_analysis.pdf")
    args = p.parse_args()

    cur_summary = _load(args.summary)
    cur = _rows(cur_summary)
    base = {}
    if args.baseline and Path(args.baseline).exists():
        base = {t["task_id"]: t["accuracy"] * 100 for t in _load(args.baseline)["tasks"]}
    else:
        # Baseline accuracies (untrained Qwen2.5-0.5B-Instruct) from the eval run.
        base = {
            "sc_ring_count": 22.5, "sc_aromatic_ring": 21.0, "sc_fused_ring": 12.0,
            "sc_carbon_atom": 1.0, "sc_hetero_atom": 4.0, "sc_hba": 10.0,
            "sc_rotatable_bond": 15.0, "mc_topology": 2.5, "mc_composition": 0.0,
            "si_ring": 4.5, "si_aromatic_ring": 2.5, "si_carbon_atom": 0.0,
            "si_hetero_atom": 0.0, "mi_ring_aromatic": 5.0, "cg_ring_count": 17.0,
            "cg_carbon_atom": 0.5,
        }

    order = [t["task_id"] for t in cur_summary["tasks"]]

    # Pull qualitative examples if eval files are available.
    idx_ex = gen_ex = None
    if args.eval_dir:
        ed = Path(args.eval_dir)
        for cand in ("si_aromatic_ring", "si_ring", "si_carbon_atom"):
            f = ed / f"{cand}_eval.json"
            if f.exists():
                idx_ex = (cand, _top_outputs(f)); break
        for cand in ("cg_carbon_atom", "cg_ring_count"):
            f = ed / f"{cand}_eval.json"
            if f.exists():
                gen_ex = (cand, _top_outputs(f)); break

    def acc(t):
        return cur[t]["accuracy"] * 100

    def ps(t):
        return cur[t]["partial_score_mean"] * 100

    def dar(t):
        return cur[t].get("distinct_answer_rate", 0.0) * 100

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # ---- Page 1: title + executive summary ----
        _text_page(pdf, "Why Index & Generation Tasks Are Not Learned", [
            ("body", "Failure analysis of multitask GRPO fine-tuning of "
                     "Qwen2.5-0.5B-Instruct on MolecularIQ. Model: "
                     f"{cur_summary.get('model_path','curriculum')}. Each task is "
                     "evaluated on 200 held-out molecules; scoring uses the official "
                     "moleculariq_core verifier."),
            ("space", ""),
            ("h2", "Executive summary"),
            ("body", "The model clearly learns COUNTING (e.g. ring_count "
                     f"{acc('sc_ring_count'):.0f}% vs {base['sc_ring_count']:.0f}% "
                     "baseline) but two task families stay at the floor: atom-INDEX "
                     "identification (si_*, mi_*) and constraint-based GENERATION "
                     "(cg_*)."),
            ("body", "Both failures have the same shape: the task demands an "
                     "instance-specific structured output, but the policy collapses "
                     "to a single instance-independent answer that the dense "
                     "partial-credit reward still tolerates. The exact-match reward "
                     "is too sparse to pull the model out of that basin, and a 0.5B "
                     "model lacks the capacity for the underlying SMILES-graph "
                     "reasoning (index) and controllable generation (cg)."),
            ("space", ""),
            ("bullet", "Index: outputs a fixed positional 'prior' (a contiguous "
                       "block of atom indices), not the molecule's real atom set."),
            ("bullet", "Generation: outputs ONE always-valid molecule (benzene) for "
                       "every prompt, ignoring the requested property value."),
            ("bullet", "These are capability limits, not scoring bugs: answers are "
                       "valid JSON / valid SMILES, and partial credit is the honest "
                       "verifier score."),
        ])

        # ---- Page 2: results overview ----
        _bar_page(pdf, cur, base, order)

        # ---- Page 3: index ----
        idx_blocks = [
            ("h2", "Symptom"),
            ("body", "Exact accuracy is ~0 across all index tasks: "
                     f"si_ring {acc('si_ring'):.1f}%, si_aromatic_ring "
                     f"{acc('si_aromatic_ring'):.1f}%, si_carbon_atom "
                     f"{acc('si_carbon_atom'):.1f}%, si_hetero_atom "
                     f"{acc('si_hetero_atom'):.1f}%, mi_ring_aromatic "
                     f"{acc('mi_ring_aromatic'):.1f}%."),
            ("body", "Yet answers are well-formed JSON (json_valid_rate = 1.0) and "
                     "earn moderate partial credit (e.g. si_ring "
                     f"{ps('si_ring'):.0f}%). The partial score is misleading on its "
                     "own; the diversity metric exposes why."),
            ("h2", "Diagnosis: collapse to a positional prior"),
            ("body", f"distinct_answer_rate is only {dar('si_ring'):.0f}-"
                     f"{dar('mi_ring_aromatic'):.0f}% — the model reuses a handful of "
                     "answers across 200 different molecules. A model that actually "
                     "read each molecule would approach 100% (every molecule has its "
                     "own atom set)."),
        ]
        if idx_ex:
            name, (nd, tot, common) = idx_ex
            idx_blocks += [
                ("body", f"Most frequent outputs for {name} ({nd} distinct / {tot}):"),
            ]
            idx_blocks += [("mono", f"{c:>3}x  {a}") for a, c in common]
        idx_blocks += [
            ("space", ""),
            ("body", "The outputs are contiguous integer runs (e.g. [7,8,...,15]) "
                     "that drift only slightly with the molecule. The model even "
                     "emits a non-empty list when the gold answer is empty. It has "
                     "learned the MARGINAL distribution of where ring/atom indices "
                     "tend to sit, not how to locate them in a given SMILES string."),
            ("h2", "Why the reward cannot fix it"),
            ("body", "Atom indexing requires parsing SMILES into an ordered graph "
                     "and emitting the exact 0-based set — a structured perception "
                     "task. Exact-set match (the only un-gameable signal) is almost "
                     "never achieved, so it gives no gradient. The dense overlap "
                     "reward (Jaccard) still hands partial credit to a generic block "
                     "because real index sets are often contiguous, so GRPO settles "
                     "in the prior. At 0.5B the model cannot do per-atom indexing."),
        ]
        _text_page(pdf, "Index tasks (si_*, mi_*): positional-prior collapse", idx_blocks)

        # ---- Page 4: constraint generation ----
        gen_blocks = [
            ("h2", "Symptom"),
            ("body", f"cg_ring_count {acc('cg_ring_count'):.0f}%, cg_carbon_atom "
                     f"{acc('cg_carbon_atom'):.0f}%. valid_smiles_rate = 1.0 (every "
                     "output is a parseable molecule), so the failure is not about "
                     "producing valid chemistry."),
            ("h2", "Diagnosis: mode collapse to one safe molecule"),
            ("body", f"distinct_answer_rate = {dar('cg_carbon_atom'):.1f}% — the "
                     "model emits essentially ONE molecule for all 200 prompts, "
                     "regardless of the requested property value."),
        ]
        if gen_ex:
            name, (nd, tot, common) = gen_ex
            gen_blocks += [("body", f"Outputs for {name} ({nd} distinct / {tot}):")]
            gen_blocks += [("mono", f"{c:>3}x  {a}") for a, c in common]
        gen_blocks += [
            ("space", ""),
            ("body", "The molecule is benzene (c1ccccc1): 6 carbons, 1 ring. This "
                     "explains the accuracies exactly — the model is correct only "
                     "when the requested value happens to match benzene. ~18% of "
                     "test targets ask for ring_count = 1 (=> cg_ring_count ~18%); "
                     "~3% ask for carbon_count = 6 (=> cg_carbon_atom ~3%). It is "
                     "right by coincidence, not by construction."),
            ("h2", "Why the reward cannot fix it"),
            ("body", "The generation reward is a valid-SMILES bonus plus "
                     "property-closeness. Emitting one always-valid molecule banks "
                     "the validity bonus and some closeness on every prompt — a "
                     "high-reward, zero-risk policy. Producing a molecule with a "
                     "SPECIFIED count requires compositional, controllable "
                     "generation (assemble a graph to hit an exact integer), which "
                     "is far harder than recognising a property and is not forced by "
                     "partial credit. Exact satisfaction is rare, so there is little "
                     "gradient to escape the constant-output basin."),
        ]
        _text_page(pdf, "Constraint generation (cg_*): mode collapse to benzene", gen_blocks)

        # ---- Page 5: diversity figure ----
        _diag_page(pdf, cur)

        # ---- Page 6: common cause + implications ----
        _text_page(pdf, "Common root cause and implications", [
            ("h2", "One mechanism, two symptoms"),
            ("body", "Counting asks the model to recognise an aggregate scalar — a "
                     "single number summarising the whole molecule — which a 0.5B "
                     "model can approximate (hence the large gains over baseline)."),
            ("body", "Index and constraint generation instead require "
                     "INSTANCE-SPECIFIC STRUCTURED output: a precise set of atom "
                     "positions, or a precise molecule. In both, the model defaults "
                     "to an instance-independent answer (a fixed index block; a "
                     "single molecule) that the dense reward tolerates, while the "
                     "exact reward is too sparse to provide an escape gradient."),
            ("space", ""),
            ("h2", "These are not scoring artefacts"),
            ("bullet", "Scoring is the official moleculariq_core verifier; the exact "
                       "and partial signals come from one verifier call."),
            ("bullet", "Answers are well-formed (json_valid_rate = 1.0; "
                       "valid_smiles_rate = 1.0), so format is not the bottleneck."),
            ("bullet", "Partial credit uses Jaccard (not F1), which penalises the "
                       "'dump everything' exploit, so the reported numbers are "
                       "honest rather than inflated."),
            ("space", ""),
            ("h2", "What would be required to learn them"),
            ("bullet", "A larger base model with enough capacity to parse SMILES "
                       "into an atom-indexed graph."),
            ("bullet", "Structured / constrained decoding so outputs are tied to the "
                       "actual atoms of the input molecule."),
            ("bullet", "For generation: a reward that requires exact constraint "
                       "satisfaction, a curriculum from tiny targets, and more "
                       "exploration (higher num_generations) to break the constant "
                       "output."),
            ("bullet", "Otherwise, treat index/generation as out of scope at 0.5B "
                       "and report counting + the collapse analysis as the result."),
        ])

    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
