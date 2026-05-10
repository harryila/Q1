#!/usr/bin/env python3
"""Page-1 teaser: 'Different heads, same fragile tasks.'

Two panels at single-column width.

Panel A. Cross-model retrieval-head pools barely overlap.
   Three bars (Llama-Qwen, Llama-Mistral, Qwen-Mistral) of top-16 union
   Jaccard, with empirical random expected shown as a horizontal dashed
   marker over each bar. Random-expected source: 1000 random head-subset
   pairs sampled at the same sizes from each model's full head population.

Panel B. The same fact-extraction tasks are fragile across all three models.
   8 x 3 heatmap of target sensitivity at K=16, where
       sensitivity[t] = mean over sources s of drop[s -> t][K=16].
   Rows are tasks, ordered by cross-model mean fragility (most fragile at
   top, most robust at bottom). The visual story: top rows are dark across
   all three columns; bottom rows are pale across all three.

Inputs:
    results/cross_model/cross_model_union_overlap.csv
    results/<model>/transfer/cross_task_transfer_matrix.json   (3 models)

Writes:
    paper/figures/main/fig_teaser_different_heads_same_tasks.{pdf,png}

Run from paper/:
    python scripts/figures/make_fig_teaser.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS: List[Tuple[str, str, str]] = [
    ("Llama",   "Llama-3.1-8B",   "results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json"),
    ("Qwen",    "Qwen2.5-7B",     "results/qwen_2_5_7B_instruct/transfer/cross_task_transfer_matrix.json"),
    ("Mistral", "Mistral-7B",     "results/mistral_7B_instruct/transfer/cross_task_transfer_matrix.json"),
]
K_TEASER = "16"

# Display labels for the eight SEC tasks (snake_case -> compact reader-friendly)
TASK_LABEL = {
    "registrant_name":        "registrant_name",
    "headquarters_city":      "HQ_city",
    "headquarters_state":     "HQ_state",
    "incorporation_state":    "incorp_state",
    "incorporation_year":     "incorp_year",
    "employees_count_total":  "employees",
    "ceo_lastname":           "ceo_lastname",
    "holder_record_amount":   "holder_amount",
}


def load_pair_overlaps_at_k16() -> List[Dict]:
    """Return the three Llama-Qwen / Llama-Mistral / Qwen-Mistral rows at K=16."""
    rows = list(csv.DictReader(open("results/cross_model/cross_model_union_overlap.csv")))
    keep_models = {"Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct"}
    short = {
        "Llama-3.1-8B-Instruct":   "Llama",
        "Qwen2.5-7B-Instruct":     "Qwen",
        "Mistral-7B-Instruct":     "Mistral",
    }
    out = []
    for r in rows:
        if int(r["K"]) != 16:
            continue
        if r["model_a"] not in keep_models or r["model_b"] not in keep_models:
            continue
        out.append({
            "pair":         f"{short[r['model_a']]}\u2013{short[r['model_b']]}",
            "obs":          float(r["jaccard_obs"]),
            "rand_mean":    float(r["jaccard_rand_mean"]),
            "rand_p97_5":   float(r["jaccard_rand_p97.5"]),
        })
    return out


def load_target_sensitivity_at_k16() -> Tuple[List[str], np.ndarray]:
    """Return (tasks-sorted-by-fragility-desc, 8x3 sensitivity matrix)."""
    sens: Dict[str, Dict[str, float]] = {}
    for short, _disp, path in MODELS:
        d = json.load(open(path))
        sources, targets = d["sources"], d["targets"]
        sens[short] = {
            t: float(np.mean([d["results"][s][t]["by_k"][K_TEASER]["drop_from_k0"]
                              for s in sources]))
            for t in targets
        }
    # task order: descending cross-model mean
    tasks_in_data = list(sens["Llama"].keys())
    cm_mean = {t: np.mean([sens[m][t] for m, _, _ in MODELS]) for t in tasks_in_data}
    tasks_sorted = sorted(tasks_in_data, key=cm_mean.get, reverse=True)

    M = np.zeros((len(tasks_sorted), len(MODELS)))
    for i, t in enumerate(tasks_sorted):
        for j, (short, _, _) in enumerate(MODELS):
            M[i, j] = sens[short][t]
    return tasks_sorted, M


def main() -> None:
    pair_rows  = load_pair_overlaps_at_k16()
    tasks, S   = load_target_sensitivity_at_k16()

    fig = plt.figure(figsize=(3.45, 2.7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.7], wspace=0.85,
                           left=0.16, right=0.92, top=0.86, bottom=0.18)

    # Compact pair labels for Panel A so they don't crash into Panel B.
    # Cover both orderings since the CSV may emit either.
    PAIR_SHORT = {
        "Llama\u2013Qwen":     "L\u2013Q",
        "Qwen\u2013Llama":     "L\u2013Q",
        "Llama\u2013Mistral":  "L\u2013M",
        "Mistral\u2013Llama":  "L\u2013M",
        "Qwen\u2013Mistral":   "Q\u2013M",
        "Mistral\u2013Qwen":   "Q\u2013M",
    }

    # --- Panel A: bar chart of pair-wise Jaccard at K=16 ---
    axA = fig.add_subplot(gs[0, 0])
    pairs   = [PAIR_SHORT.get(r["pair"], r["pair"]) for r in pair_rows]
    obs     = [r["obs"] for r in pair_rows]
    rmeans  = [r["rand_mean"] for r in pair_rows]
    rhi     = [r["rand_p97_5"] for r in pair_rows]
    x = np.arange(len(pair_rows))
    axA.bar(x, obs, width=0.60, color="#4a7ab8",
            edgecolor="#1f3f6b", linewidth=0.6)
    # Random-expected mean as a short horizontal dash above each bar.
    for xi, m in zip(x, rmeans):
        axA.hlines(m, xi - 0.34, xi + 0.34,
                   colors="#888888", linestyles="--", linewidth=1.1, zorder=3)
    # Light vertical line out to the random p97.5 to give a sense of the null
    # spread without dominating the visual.
    for xi, lo, hi in zip(x, rmeans, rhi):
        axA.vlines(xi, lo, hi, colors="#bbbbbb", linewidth=0.8, zorder=2)

    axA.set_xticks(x)
    axA.set_xticklabels(pairs, fontsize=8.0)
    axA.set_ylim(0, max(0.06, max(obs + rhi) * 1.15))
    axA.set_ylabel("Top-16 Jaccard", fontsize=8.0)
    axA.set_title("A. Heads barely overlap", fontsize=8.5, pad=4)
    axA.tick_params(axis="y", labelsize=7.0)
    axA.tick_params(axis="x", length=0, pad=2)
    for spine in ("top", "right"):
        axA.spines[spine].set_visible(False)
    axA.grid(axis="y", alpha=0.20, linestyle=":")
    axA.set_axisbelow(True)
    # Inline legend
    axA.plot([], [], color="#888888", linestyle="--", linewidth=1.1, label="random")
    axA.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.6,
               borderpad=0.1, labelspacing=0.2)

    # --- Panel B: 8 x 3 task-sensitivity heatmap ---
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(S, aspect="auto", cmap="Reds", vmin=0.0, vmax=0.85)
    # Compact column labels (just the model family); full identifiers are in Methods.
    axB.set_xticks(np.arange(len(MODELS)))
    axB.set_xticklabels([short for short, _, _ in MODELS], fontsize=7.5)
    axB.set_yticks(np.arange(len(tasks)))
    axB.set_yticklabels([TASK_LABEL[t] for t in tasks], fontsize=6.6)
    axB.set_title("B. Same tasks fragile", fontsize=8.5, pad=4)
    # Annotate each cell
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = S[i, j]
            txt_color = "white" if v > 0.50 else "#222222"
            axB.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color=txt_color, fontsize=6.3)
    # Hide spines but keep ticks readable
    for spine in axB.spines.values():
        spine.set_visible(False)
    axB.tick_params(length=0)

    # Colorbar attached to axB --- thin vertical strip on the right
    cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.04)
    cb.set_label("sensitivity\n(drop)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.5)

    out_dir = Path("figures/main")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig_teaser_different_heads_same_tasks.pdf"
    out_png = out_dir / "fig_teaser_different_heads_same_tasks.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Echo headline numbers for paper writing
    print()
    print("Panel A --- pair-wise top-16 union Jaccard at K=16:")
    for r in pair_rows:
        lift = r["obs"] / r["rand_mean"] if r["rand_mean"] else float("inf")
        print(f"  {r['pair']:<14} obs={r['obs']:.3f}  rand_mean={r['rand_mean']:.3f}  lift={lift:.2f}x")
    print()
    print("Panel B --- target sensitivity at K=16, sorted by cross-model mean:")
    print(f"  {'task':<22}" + "".join(f"{m:>8}" for m, _, _ in MODELS) + "    cross-mean")
    for i, t in enumerate(tasks):
        cm = np.mean(S[i])
        print(f"  {TASK_LABEL[t]:<22}" + "".join(f"{S[i,j]:>8.3f}" for j in range(S.shape[1])) + f"      {cm:.3f}")


if __name__ == "__main__":
    main()
