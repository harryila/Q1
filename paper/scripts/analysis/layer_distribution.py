#!/usr/bin/env python3
"""Layer-position analysis of detected retrieval heads (Task 1).

Goal: graduate the paper's fragility claim from a behavioural observation
to a mechanistic one. If fragile tasks share a layer-localised circuit,
that's a structural finding the paper can lead with.

For each model M and SEC task t we load the top-16 detected heads (as
``(layer, head)`` pairs from
``detection/<model>/topk/long_context_<task>_top16.json``, falling back to
slicing the full ranking when the top-K file isn't present). Per-task we
compute:

    layer_histogram[t]    counts of heads per layer, on the model's
                          original layer axis (32 layers for Llama / Mistral
                          / OLMo, 28 for Qwen).
    layer_entropy[t]      Shannon entropy of the layer histogram (one
                          number per task, used as the unit of analysis for
                          the fragile-vs-robust test).

Cross-task tests (per model):

    permutation_test  primary; 1000 random fragile/robust assignments
                      of the 8 tasks; observed mean-entropy-difference
                      compared against the null distribution. Robust to
                      the cross-task dependency structure (tasks share
                      heads — e.g. headquarters_city ↔ headquarters_state
                      overlap at 78% per FINDINGS.md §3 — so a
                      pooled-heads KS test would violate independence).
    mann_whitney_u   sanity check; on the 4 fragile vs 4 robust per-task
                     entropies. n is tiny (4 vs 4 → minimum achievable
                     p ≈ 0.029) but the test is honest.

Cross-model tests:

    layer_pearson  for each task, Pearson r between two models' layer-mass
                   vectors on a shared 14-bin relative-depth grid (model
                   layer ÷ (n_layers - 1) → 14 equal-width bins). Avoids
                   the information loss of interpolating 32-layer models
                   down to 28 layers.

Outputs:

    paper/results/layer_analysis/per_task_layer_histograms.png
    paper/results/layer_analysis/fragile_vs_robust_overlay.png
    paper/results/layer_analysis/layer_concentration_table.csv
    paper/results/layer_analysis/permutation_tests.csv
    paper/results/layer_analysis/REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = [
    "ceo_lastname",
    "employees_count_total",
    "headquarters_city",
    "headquarters_state",
    "holder_record_amount",
    "incorporation_state",
    "incorporation_year",
    "registrant_name",
]

# (label, det_dir, n_layers, n_heads_per_layer)
MODELS: List[Tuple[str, str, int, int]] = [
    ("Llama-3.1-8B-Instruct", "detection/llama_3_1_8B_instruct",  32, 32),
    ("Mistral-7B-Instruct",   "detection/mistral_7B_instruct",    32, 32),
    ("OLMo-7B-Instruct",      "detection/olmo_7B/instruct",       32, 32),
    ("OLMo-7B (base)",        "detection/olmo_7B/base",           32, 32),
    ("Qwen2.5-7B-Instruct",   "detection/qwen_2_5_7B_instruct",   28, 28),
]

K = 16
NUM_PERMUTATIONS = 1000
NUM_RELATIVE_BINS = 14


# ─── Loading ────────────────────────────────────────────────────────────────
def load_top_k_heads(repo_root: str, det_dir: str, task: str, k: int) -> List[Tuple[int, int]]:
    """Return the top-K (layer, head) pairs for one task."""
    topk_path = os.path.join(repo_root, det_dir, "topk",
                             f"long_context_{task}_top{k}.json")
    if not os.path.exists(topk_path):
        full_path = os.path.join(repo_root, det_dir, f"long_context_{task}_heads.json")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Neither {topk_path} nor {full_path} exists")
        topk_path = full_path
    with open(topk_path) as fh:
        d = json.load(fh)
    out: List[Tuple[int, int]] = []
    for entry in d[:k]:
        key = entry[0]
        # Format is "L-H" where L and H are non-negative; first '-' is the separator
        l, h = key.split("-", 1)
        out.append((int(l), int(h)))
    return out


def load_fragility(repo_root: str) -> Dict[str, float]:
    """Read fragility per task from Task 2's output if present; else recompute."""
    csv_path = os.path.join(repo_root, "results", "fragility_predictors", "correlations.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"[layer_distribution] Task 2 output missing: {csv_path}\n"
                 f"Run scripts/analysis/predictive_fragility.py first.")
    out: Dict[str, float] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            t = row["task"]
            if t in TASKS:
                out[t] = float(row["fragility"])
    if len(out) != len(TASKS):
        sys.exit(f"[layer_distribution] Task 2 output is missing some tasks: {set(TASKS) - set(out.keys())}")
    return out


# ─── Stats ──────────────────────────────────────────────────────────────────
def layer_histogram(heads: List[Tuple[int, int]], n_layers: int) -> np.ndarray:
    h = np.zeros(n_layers)
    for layer, _head in heads:
        if 0 <= layer < n_layers:
            h[layer] += 1
    return h


def layer_entropy(hist: np.ndarray) -> float:
    p = hist / hist.sum() if hist.sum() > 0 else hist
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def relative_depth_histogram(heads: List[Tuple[int, int]],
                             n_layers: int, n_bins: int) -> np.ndarray:
    """Bin head layers into n_bins equal-width bins on the [0, 1] relative-depth axis."""
    if n_layers <= 1:
        return np.zeros(n_bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(n_bins)
    for layer, _head in heads:
        rel = layer / (n_layers - 1)
        # rightmost edge inclusive
        b = min(int(rel * n_bins), n_bins - 1)
        out[b] += 1
    return out


def permutation_p_value(values: List[float], group_a_mask: List[bool],
                        n_perm: int = NUM_PERMUTATIONS, seed: int = 0
                        ) -> Tuple[float, float, float, float]:
    """Two-sided permutation test on mean-difference between two groups.

    Returns (observed_diff, null_mean, null_std, two_sided_p_value).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    mask = np.array(group_a_mask, dtype=bool)
    if mask.sum() == 0 or mask.sum() == len(mask):
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = arr[mask].mean() - arr[~mask].mean()
    n_a = mask.sum()
    nulls = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(arr))
        a_indices = perm[:n_a]
        b_indices = perm[n_a:]
        nulls[i] = arr[a_indices].mean() - arr[b_indices].mean()
    p_two_sided = float(np.mean(np.abs(nulls) >= abs(obs)))
    return float(obs), float(nulls.mean()), float(nulls.std()), p_two_sided


# ─── Reporting ──────────────────────────────────────────────────────────────
def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_per_task_histograms(path: str,
                             histograms_per_model: Dict[str, Dict[str, np.ndarray]],
                             n_layers_per_model: Dict[str, int]) -> None:
    n_models = len(MODELS)
    n_tasks = len(TASKS)
    fig, axes = plt.subplots(n_models, n_tasks,
                             figsize=(2.0 * n_tasks, 1.6 * n_models),
                             sharey=True)
    for i, (m_label, _, _, _) in enumerate(MODELS):
        n_layers = n_layers_per_model[m_label]
        for j, t in enumerate(TASKS):
            ax = axes[i, j] if n_models > 1 else axes[j]
            hist = histograms_per_model[m_label][t]
            ax.bar(range(n_layers), hist, width=1.0)
            if i == 0:
                ax.set_title(t.replace("_", "\n"), fontsize=7)
            if j == 0:
                ax.set_ylabel(m_label.split("-")[0][:7], fontsize=7)
            ax.set_xlim(-0.5, n_layers - 0.5)
            ax.set_xticks([])
            ax.set_yticks([0, 4, 8])
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Top-{K} head layer histograms (rows = models, cols = tasks)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fragile_vs_robust_overlay(path: str,
                                   relative_hists_per_model: Dict[str, Dict[str, np.ndarray]],
                                   fragile: List[str], robust: List[str],
                                   perm_results: Dict[str, Dict[str, float]]) -> None:
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(3.0 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]
    bin_centers = (np.arange(NUM_RELATIVE_BINS) + 0.5) / NUM_RELATIVE_BINS
    for i, (m_label, _, _, _) in enumerate(MODELS):
        ax = axes[i]
        f_mass = np.zeros(NUM_RELATIVE_BINS)
        r_mass = np.zeros(NUM_RELATIVE_BINS)
        for t in fragile:
            f_mass += relative_hists_per_model[m_label][t]
        for t in robust:
            r_mass += relative_hists_per_model[m_label][t]
        # Normalise so each shows fraction-per-bin
        if f_mass.sum() > 0:
            f_mass = f_mass / f_mass.sum()
        if r_mass.sum() > 0:
            r_mass = r_mass / r_mass.sum()
        width = 1.0 / NUM_RELATIVE_BINS / 2.5
        ax.bar(bin_centers - width / 2, f_mass, width=width,
               label="fragile (top 4)", alpha=0.8)
        ax.bar(bin_centers + width / 2, r_mass, width=width,
               label="robust (bottom 4)", alpha=0.8)
        perm = perm_results.get(m_label, {})
        p_perm = perm.get("permutation_p", float("nan"))
        p_mw = perm.get("mannwhitney_p", float("nan"))
        ax.set_title(f"{m_label}\nperm p={p_perm:.3g}, MW-U p={p_mw:.3g}", fontsize=8)
        ax.set_xlabel("Relative depth (layer / (n_layers - 1))", fontsize=8)
        if i == 0:
            ax.set_ylabel("Fraction of top-16 heads", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Layer distribution: fragile (top-4) vs robust (bottom-4) tasks",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(path: str, fragility: Dict[str, float],
                 fragile: List[str], robust: List[str],
                 entropies_per_model: Dict[str, Dict[str, float]],
                 perm_results: Dict[str, Dict[str, float]],
                 cross_model_pearson: List[Dict[str, object]]) -> None:
    lines = [
        "# Layer-position analysis — REPORT",
        "",
        "## Setup",
        "",
        "- Top-16 heads per (model, task), parsed as `(layer, head)` from `detection/<model>/topk/long_context_<task>_top16.json` (or sliced from the full ranking when the top-K file isn't present, e.g. for 3 Llama tasks).",
        "- Models analysed: " + ", ".join(m[0] for m in MODELS) + ".",
        "- Unit of analysis: per-task layer entropy (one number per task → n=4 fragile vs n=4 robust). Pooling 4 × 16 = 64 head-layer indices for a KS test would violate independence (tasks share heads — `headquarters_city` ↔ `headquarters_state` overlap at 78% per FINDINGS.md §3).",
        "- Fragility ranking from Task 2 (`results/fragility_predictors/correlations.csv`):",
        "",
        "| Rank | Task | Fragility |",
        "| ---: | --- | ---: |",
    ]
    ranked = sorted(fragility.items(), key=lambda kv: -kv[1])
    for i, (t, f) in enumerate(ranked, 1):
        marker = "  ← fragile" if t in fragile else ("  ← robust" if t in robust else "")
        lines.append(f"| {i} | `{t}`{marker} | {f:.3f} |")

    lines.extend([
        "",
        "## Per-model verdict (permutation + Mann-Whitney U on per-task entropies)",
        "",
        "| Model | Mean fragile entropy | Mean robust entropy | Δ obs | Perm p | MW-U p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for m_label, _, _, _ in MODELS:
        e = entropies_per_model[m_label]
        ent_f = np.mean([e[t] for t in fragile])
        ent_r = np.mean([e[t] for t in robust])
        perm = perm_results.get(m_label, {})
        lines.append(
            f"| {m_label} | {ent_f:.3f} | {ent_r:.3f} | "
            f"{perm.get('observed_diff', float('nan')):+.3f} | "
            f"{perm.get('permutation_p', float('nan')):.3g} | "
            f"{perm.get('mannwhitney_p', float('nan')):.3g} |"
        )

    lines.extend([
        "",
        "## Verdict logic",
        "",
        "- If both permutation p < 0.05 AND Mann-Whitney U p < 0.05 in 3+ of the 5 models, **claim**: \"Per-task layer entropy differs significantly between fragile and robust tasks (permutation p<0.05 in N/5 models, Mann-Whitney consistent direction). The fragility signature is layer-localised.\"",
        "- If only the permutation test rejects (likely given Mann-Whitney's n=4 floor), **claim**: \"Permutation test detects systematic difference; Mann-Whitney is underpowered with n=4 vs n=4.\"",
        "- If neither rejects consistently, **fallback**: drop the layer-localisation claim; report the histograms descriptively in §4 without a mechanistic narrative.",
        "",
        "## Cross-model layer-mass agreement (per task, 14-bin relative-depth Pearson r)",
        "",
        "| Task | Llama↔Mistral | Llama↔OLMo-Inst | Llama↔Qwen | Mistral↔OLMo-Inst | Mistral↔Qwen | OLMo-Inst↔Qwen |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    by_task: Dict[str, Dict[str, float]] = {}
    for r in cross_model_pearson:
        t = r["task"]
        pair_key = f"{r['model_a_short']}↔{r['model_b_short']}"
        by_task.setdefault(t, {})[pair_key] = r["pearson_r"]
    pair_order = ["Llama↔Mistral", "Llama↔OLMo-Inst", "Llama↔Qwen",
                  "Mistral↔OLMo-Inst", "Mistral↔Qwen", "OLMo-Inst↔Qwen"]
    for t in TASKS:
        cells = []
        for pk in pair_order:
            v = by_task.get(t, {}).get(pk)
            cells.append(f"{v:+.2f}" if v is not None and not np.isnan(v) else "—")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Files",
        "",
        "- `per_task_layer_histograms.png`: 5×8 grid (rows=models, cols=tasks).",
        "- `fragile_vs_robust_overlay.png`: per-model fragile-vs-robust bars on the relative-depth axis with permutation/MW-U p-values annotated.",
        "- `layer_concentration_table.csv`: per (model, task) entropy / mode / IQR.",
        "- `permutation_tests.csv`: per model, observed Δ entropy + permutation null mean/std/p-value + Mann-Whitney U statistic and p-value + cross-model Pearson r per task.",
    ])
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def short_label(label: str) -> str:
    if "Llama" in label:
        return "Llama"
    if "Mistral" in label:
        return "Mistral"
    if "OLMo" in label and "Instruct" in label:
        return "OLMo-Inst"
    if "OLMo" in label:
        return "OLMo-Base"
    if "Qwen" in label:
        return "Qwen"
    return label


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "layer_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[layer_distribution] repo_root = {repo_root}")
    print(f"[layer_distribution] output_dir = {output_dir}")
    print()

    fragility = load_fragility(repo_root)
    ranked = sorted(fragility.items(), key=lambda kv: -kv[1])
    fragile = [t for t, _ in ranked[:4]]
    robust = [t for t, _ in ranked[4:]]
    print(f"[fragile (top 4)] {fragile}")
    print(f"[robust (bot 4)] {robust}")
    print()

    histograms_per_model: Dict[str, Dict[str, np.ndarray]] = {}
    relative_hists_per_model: Dict[str, Dict[str, np.ndarray]] = {}
    entropies_per_model: Dict[str, Dict[str, float]] = {}
    n_layers_per_model: Dict[str, int] = {}
    concentration_rows: List[Dict[str, object]] = []

    for m_label, det_dir, n_layers, _ in MODELS:
        n_layers_per_model[m_label] = n_layers
        histograms_per_model[m_label] = {}
        relative_hists_per_model[m_label] = {}
        entropies_per_model[m_label] = {}
        for t in TASKS:
            heads = load_top_k_heads(repo_root, det_dir, t, K)
            hist = layer_histogram(heads, n_layers)
            rel_hist = relative_depth_histogram(heads, n_layers, NUM_RELATIVE_BINS)
            ent = layer_entropy(hist)
            mode = int(hist.argmax())
            non_zero_layers = int((hist > 0).sum())
            histograms_per_model[m_label][t] = hist
            relative_hists_per_model[m_label][t] = rel_hist
            entropies_per_model[m_label][t] = ent
            concentration_rows.append({
                "model": m_label,
                "task": t,
                "n_heads": int(hist.sum()),
                "n_layers_with_heads": non_zero_layers,
                "mode_layer": mode,
                "layer_entropy": round(ent, 4),
                "is_fragile": t in fragile,
            })

    # Per-model permutation + Mann-Whitney
    perm_results: Dict[str, Dict[str, float]] = {}
    perm_rows: List[Dict[str, object]] = []
    for m_label, _, _, _ in MODELS:
        ents_in_order = [entropies_per_model[m_label][t] for t in TASKS]
        mask = [t in fragile for t in TASKS]
        obs, null_mean, null_std, p_perm = permutation_p_value(
            ents_in_order, mask, n_perm=NUM_PERMUTATIONS, seed=args.seed)
        f_ents = [entropies_per_model[m_label][t] for t in fragile]
        r_ents = [entropies_per_model[m_label][t] for t in robust]
        try:
            u_stat, p_mw = mannwhitneyu(f_ents, r_ents, alternative="two-sided")
            u_stat, p_mw = float(u_stat), float(p_mw)
        except ValueError:
            u_stat, p_mw = float("nan"), float("nan")
        perm_results[m_label] = {
            "observed_diff": obs,
            "permutation_null_mean": null_mean,
            "permutation_null_std": null_std,
            "permutation_p": p_perm,
            "mannwhitney_u": u_stat,
            "mannwhitney_p": p_mw,
        }
        perm_rows.append({
            "model": m_label,
            "observed_entropy_diff_fragile_minus_robust": round(obs, 4),
            "permutation_null_mean": round(null_mean, 4),
            "permutation_null_std": round(null_std, 4),
            "permutation_p_two_sided": round(p_perm, 6),
            "mannwhitney_u": round(u_stat, 3),
            "mannwhitney_p_two_sided": round(p_mw, 6),
        })

    # Cross-model layer-mass agreement (Pearson r per task per pair, on the
    # 14-bin relative-depth axis).
    cross_model_pearson_rows: List[Dict[str, object]] = []
    pair_order = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (2, 4)]  # skip OLMo-base for paper-size brevity
    for i, j in pair_order:
        a_label, _, _, _ = MODELS[i]
        b_label, _, _, _ = MODELS[j]
        for t in TASKS:
            a_vec = relative_hists_per_model[a_label][t]
            b_vec = relative_hists_per_model[b_label][t]
            if a_vec.std() == 0 or b_vec.std() == 0:
                r, p = float("nan"), float("nan")
            else:
                r, p = pearsonr(a_vec, b_vec)
                r, p = float(r), float(p)
            cross_model_pearson_rows.append({
                "model_a": a_label,
                "model_b": b_label,
                "model_a_short": short_label(a_label),
                "model_b_short": short_label(b_label),
                "task": t,
                "pearson_r": r,
                "pearson_p": p,
            })

    # ── Outputs ──
    write_csv(os.path.join(output_dir, "layer_concentration_table.csv"),
              concentration_rows)
    # Augment perm_rows with cross-model agreement summary at the end
    perm_rows.append({
        "model": "_cross_model_summary_",
        "observed_entropy_diff_fragile_minus_robust": "",
        "permutation_null_mean": "",
        "permutation_null_std": "",
        "permutation_p_two_sided": "",
        "mannwhitney_u": "",
        "mannwhitney_p_two_sided": "",
    })
    cross_model_summary_rows: List[Dict[str, object]] = []
    for r in cross_model_pearson_rows:
        cross_model_summary_rows.append({
            "model": f"pair:{r['model_a_short']}↔{r['model_b_short']}",
            "observed_entropy_diff_fragile_minus_robust": "",
            "permutation_null_mean": f"task={r['task']}",
            "permutation_null_std": "",
            "permutation_p_two_sided": "",
            "mannwhitney_u": f"r={r['pearson_r']:+.3f}" if not np.isnan(r['pearson_r']) else "",
            "mannwhitney_p_two_sided": f"p={r['pearson_p']:.3g}" if not np.isnan(r['pearson_p']) else "",
        })
    write_csv(os.path.join(output_dir, "permutation_tests.csv"),
              perm_rows + cross_model_summary_rows)

    plot_per_task_histograms(
        os.path.join(output_dir, "per_task_layer_histograms.png"),
        histograms_per_model, n_layers_per_model)
    plot_fragile_vs_robust_overlay(
        os.path.join(output_dir, "fragile_vs_robust_overlay.png"),
        relative_hists_per_model, fragile, robust, perm_results)
    write_report(os.path.join(output_dir, "REPORT.md"),
                 fragility, fragile, robust, entropies_per_model,
                 perm_results, cross_model_pearson_rows)

    # Echo headline
    print("=" * 78)
    print("Layer-position analysis — per-model verdict")
    print("=" * 78)
    print(f"\n{'Model':<28} {'Δ entropy':>10} {'perm p':>10} {'MW-U p':>10}")
    for m_label, _, _, _ in MODELS:
        pr = perm_results[m_label]
        print(f"{m_label:<28} {pr['observed_diff']:>+10.3f} "
              f"{pr['permutation_p']:>10.3g} {pr['mannwhitney_p']:>10.3g}")
    print()
    print(f"Wrote {os.path.join(output_dir, 'per_task_layer_histograms.png')}")
    print(f"Wrote {os.path.join(output_dir, 'fragile_vs_robust_overlay.png')}")
    print(f"Wrote {os.path.join(output_dir, 'layer_concentration_table.csv')}")
    print(f"Wrote {os.path.join(output_dir, 'permutation_tests.csv')}")
    print(f"Wrote {os.path.join(output_dir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
