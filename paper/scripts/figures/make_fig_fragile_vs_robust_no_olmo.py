#!/usr/bin/env python3
"""Recreate fragile_vs_robust_overlay.png without OLMo.

The full 5-model version is still produced by ``layer_distribution.py``.
This focused script regenerates the same plot for the three models with
end-to-end ablation pipelines (Llama, Qwen, Mistral) only, so the appendix
figure matches the body's "three open-weights models" framing.

Inputs read directly from on-disk artifacts (no analysis re-run):
    detection/<model>/topk/long_context_<task>_top16.json
        (or  detection/<model>/long_context_<task>_heads.json [:16])
    results/fragility_predictors/correlations.csv          # fragility ranking
    results/layer_analysis/permutation_tests.csv           # perm + MW p-values

Output (overwrites both):
    paper/results/layer_analysis/fragile_vs_robust_overlay.png
    icml2026__1_ (1)/figures/fragile_vs_robust_overlay.png   (if dir exists)

Run from paper/:
    python scripts/figures/make_fig_fragile_vs_robust_no_olmo.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (display label, detection dir, n_layers)
MODELS: List[Tuple[str, str, int]] = [
    ("Llama-3.1-8B-Instruct", "detection/llama_3_1_8B_instruct",  32),
    ("Mistral-7B-Instruct",   "detection/mistral_7B_instruct",    32),
    ("Qwen2.5-7B-Instruct",   "detection/qwen_2_5_7B_instruct",   28),
]

K = 16
NUM_RELATIVE_BINS = 14

TASKS = [
    "registrant_name", "headquarters_city", "headquarters_state",
    "incorporation_state", "incorporation_year", "employees_count_total",
    "ceo_lastname", "holder_record_amount",
]


def load_top16_layers(model_dir: str, task: str) -> List[int]:
    base = Path(model_dir)
    topk_path = base / "topk" / f"long_context_{task}_top{K}.json"
    if topk_path.exists():
        ranked = json.load(open(topk_path))
    else:
        full = json.load(open(base / f"long_context_{task}_heads.json"))
        ranked = full[:K]
    return [int(key.split("-", 1)[0]) for key, _ in ranked]


def load_fragility_ranking() -> Dict[str, float]:
    """Read the cross-model fragility per task from correlations.csv."""
    out: Dict[str, float] = {}
    with open("results/fragility_predictors/correlations.csv") as fh:
        for row in csv.DictReader(fh):
            t = row["task"]
            if t.startswith("_") or t.startswith("corr:"):
                continue
            try:
                out[t] = float(row["fragility"])
            except (ValueError, TypeError):
                continue
    if not out:
        raise RuntimeError("Could not read fragility ranking from correlations.csv")
    return out


def load_perm_pvalues() -> Dict[str, Tuple[float, float]]:
    """Return {model_label: (permutation_p, mannwhitney_p)}."""
    out: Dict[str, Tuple[float, float]] = {}
    with open("results/layer_analysis/permutation_tests.csv") as fh:
        for row in csv.DictReader(fh):
            m = row["model"]
            if m.startswith("_") or m.startswith("pair:"):
                continue
            try:
                out[m] = (
                    float(row["permutation_p_two_sided"]),
                    float(row["mannwhitney_p_two_sided"]),
                )
            except (ValueError, TypeError):
                continue
    return out


def relative_depth_hist(layers: List[int], n_layers: int) -> np.ndarray:
    """Convert layer indices to a NUM_RELATIVE_BINS-bin histogram on [0, 1]."""
    if n_layers <= 1:
        rel = np.zeros(len(layers))
    else:
        rel = np.array(layers, dtype=float) / (n_layers - 1)
    counts, _ = np.histogram(rel, bins=NUM_RELATIVE_BINS, range=(0.0, 1.0))
    return counts.astype(float)


def main() -> None:
    fragility = load_fragility_ranking()
    perm_p = load_perm_pvalues()

    # Top-4 fragile / bottom-4 robust by cross-model fragility
    ranked = sorted(fragility.items(), key=lambda kv: -kv[1])
    fragile = [t for t, _ in ranked[:4]]
    robust  = [t for t, _ in ranked[-4:]]
    print("Fragility ranking (cross-model mean):")
    for i, (t, v) in enumerate(ranked, 1):
        tag = " (fragile)" if t in fragile else (" (robust)" if t in robust else "")
        print(f"  {i}. {t:<24} {v:.3f}{tag}")
    print()

    # Build per-(model, task) relative-depth histograms
    rel_hists: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for m_label, m_dir, n_layers in MODELS:
        for t in TASKS:
            layers = load_top16_layers(m_dir, t)
            rel_hists[m_label][t] = relative_depth_hist(layers, n_layers)

    # Plot 1 x N grid
    bin_centers = (np.arange(NUM_RELATIVE_BINS) + 0.5) / NUM_RELATIVE_BINS
    width = 1.0 / NUM_RELATIVE_BINS / 2.5
    n_models = len(MODELS)

    fig, axes = plt.subplots(1, n_models, figsize=(3.0 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, (m_label, _m_dir, _n_layers) in enumerate(MODELS):
        ax = axes[i]
        f_mass = np.zeros(NUM_RELATIVE_BINS)
        r_mass = np.zeros(NUM_RELATIVE_BINS)
        for t in fragile:
            f_mass += rel_hists[m_label][t]
        for t in robust:
            r_mass += rel_hists[m_label][t]
        if f_mass.sum() > 0:
            f_mass = f_mass / f_mass.sum()
        if r_mass.sum() > 0:
            r_mass = r_mass / r_mass.sum()
        ax.bar(bin_centers - width / 2, f_mass, width=width,
               label="fragile (top 4)", alpha=0.8)
        ax.bar(bin_centers + width / 2, r_mass, width=width,
               label="robust (bottom 4)", alpha=0.8)
        p_perm, p_mw = perm_p.get(m_label, (float("nan"), float("nan")))
        ax.set_title(f"{m_label}\nperm p={p_perm:.3g}, MW-U p={p_mw:.3g}", fontsize=8)
        ax.set_xlabel("Relative depth (layer / (n_layers - 1))", fontsize=8)
        if i == 0:
            ax.set_ylabel("Fraction of top-16 heads", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Layer distribution: fragile (top-4) vs robust (bottom-4) tasks",
                 fontsize=10, y=1.02)
    fig.tight_layout()

    # Overwrite the source-of-truth file under paper/results/layer_analysis/
    out_main = Path("results/layer_analysis/fragile_vs_robust_overlay.png")
    out_main.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_main, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_main}")

    # Mirror into the LaTeX submission's figures dir if present.
    latex_copy = Path("../icml2026__1_ (1)/figures/fragile_vs_robust_overlay.png")
    if latex_copy.parent.exists():
        fig.savefig(latex_copy, dpi=150, bbox_inches="tight")
        print(f"Wrote {latex_copy}")
    else:
        print(f"(skipped LaTeX copy: {latex_copy.parent} does not exist)")

    plt.close(fig)


if __name__ == "__main__":
    main()
