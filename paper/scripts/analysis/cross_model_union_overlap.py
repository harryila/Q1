#!/usr/bin/env python3
"""Cross-model union overlap (Task 4 of the pre-submission analyses).

Goal: show that retrieval-head pools are shared across models even though
per-task efficacy doesn't correlate (cf. K-FE Table 1). The argument:

    "varies across models" can mean "varies vs. random alignment" or
    "varies vs. expected alignment". A reviewer will ask. This script
    answers: cross-model retrieval-head pools overlap N× above random
    expectation, while per-task efficacy correlations stay low — heads are
    a SHARED SUBSTRATE, DIFFERENTLY USED.

For each model M and each top-K threshold, build the union over 8 SEC tasks:

    U_M(K) = ∪_{t ∈ 8 tasks} top-K heads detected for task t

For each pair of models (M_a, M_b), compute two metrics, plus an empirical
random-baseline distribution:

    Jaccard          J = |U_a ∩ U_b| / |U_a ∪ U_b|
    Overlap coeff.   O = |U_a ∩ U_b| / min(|U_a|, |U_b|)
    Random expected  empirical mean over 1000 random head-subset pairs sampled
                     from each model's full head population (1024 for Llama /
                     Mistral / OLMo; 784 for Qwen).
    Lift             metric_obs / metric_rand_mean
    p-value          fraction of random samples with metric ≥ metric_obs

Plot strategy (head-population mismatch makes Jaccard awkward across
1024-head and 784-head models):
    Primary (paper figure): Jaccard restricted to {Llama, Mistral,
        OLMo-Instruct} — apples-to-apples, all 1024-head.
    Sidebar (appendix):     Overlap coefficient on all 4 models including
        Qwen — insensitive to set-size mismatch.

This script is DISTINCT from
    results/cross_model/cross_model_head_overlap.csv (within-model
        SEC/LME/NQ comparison)
    results/cross_model/cross_model_jaccard_*.csv (per-task within-model
        Jaccard).

Outputs:

    paper/results/cross_model/cross_model_union_overlap.csv
    paper/results/cross_model/cross_model_union_overlap_plot.png
    paper/results/cross_model/cross_model_union_overlap_REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
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

# Each model entry: (label, detection_dir, n_heads_total)
MODELS: List[Tuple[str, str, int]] = [
    ("Llama-3.1-8B-Instruct", "detection/llama_3_1_8B_instruct",  1024),
    ("Mistral-7B-Instruct",   "detection/mistral_7B_instruct",    1024),
    ("OLMo-7B-Instruct",      "detection/olmo_7B/instruct",       1024),
    ("Qwen2.5-7B-Instruct",   "detection/qwen_2_5_7B_instruct",   784),
]

# Models with the same head-population size — used for the primary Jaccard
# figure.
JACCARD_GROUP = {"Llama-3.1-8B-Instruct", "Mistral-7B-Instruct", "OLMo-7B-Instruct"}

KS = [8, 16, 32, 48, 64, 96, 128]
NUM_RANDOM_SAMPLES = 1000


# ─── Loading ────────────────────────────────────────────────────────────────
def load_topk_for_task(repo_root: str, det_dir: str, task: str, k: int) -> List[str]:
    """Return the top-K head identifiers ('layer-head' strings) for one task.

    Prefer the precomputed topk file. Fall back to slicing the full ranking.
    """
    topk_path = os.path.join(repo_root, det_dir, "topk",
                             f"long_context_{task}_top{k}.json")
    if os.path.exists(topk_path):
        with open(topk_path) as fh:
            d = json.load(fh)
        return [entry[0] for entry in d[:k]]
    full_path = os.path.join(repo_root, det_dir, f"long_context_{task}_heads.json")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"No top-{k} or full-ranking file for {task} in {det_dir}: "
            f"tried {topk_path} and {full_path}"
        )
    with open(full_path) as fh:
        d = json.load(fh)
    return [entry[0] for entry in d[:k]]


def union_top_k(repo_root: str, det_dir: str, k: int) -> Set[str]:
    """Union of top-K heads across all 8 tasks for a given model."""
    out: Set[str] = set()
    for t in TASKS:
        out.update(load_topk_for_task(repo_root, det_dir, t, k))
    return out


# ─── Metrics ────────────────────────────────────────────────────────────────
def jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / min(len(a), len(b))


def random_overlap_distribution(size_a: int, size_b: int,
                                n_heads_a: int, n_heads_b: int,
                                n_samples: int = NUM_RANDOM_SAMPLES,
                                seed: int = 0) -> Dict[str, np.ndarray]:
    """Build empirical null distributions of Jaccard and overlap-coeff for
    two random subsets of given sizes drawn from each model's head population.

    Cross-model random expected: heads from different models are non-comparable
    (different identities), so the natural null is "if model A's pool of size
    |U_a| and model B's pool of size |U_b| were two independent random subsets
    OF THE SAME N_heads space mapped onto each other, what overlap would we
    expect?"

    Implementation: when n_heads_a == n_heads_b, treat both as random subsets
    of the same {0..N-1} space. When they differ (Llama 1024 vs Qwen 784),
    Jaccard becomes ill-defined — overlap coefficient is the meaningful
    metric there. We compute random-expected Jaccard under the
    same-population assumption (taking N = min(n_heads_a, n_heads_b)) but
    flag this in the output, and rely on the overlap coefficient for the
    cross-population comparison.
    """
    rng = np.random.default_rng(seed)
    n_common = min(n_heads_a, n_heads_b)
    js = np.zeros(n_samples)
    os_ = np.zeros(n_samples)
    for i in range(n_samples):
        # Random subsets from {0 .. n_common-1} for both
        a = set(rng.choice(n_common, size=size_a, replace=False).tolist())
        b = set(rng.choice(n_common, size=size_b, replace=False).tolist())
        js[i] = len(a & b) / len(a | b) if (a | b) else 0.0
        os_[i] = len(a & b) / min(len(a), len(b)) if a and b else 0.0
    return {
        "jaccard": js,
        "overlap": os_,
    }


# ─── Core ───────────────────────────────────────────────────────────────────
def compute_pair_row(repo_root: str,
                     a_label: str, a_dir: str, a_n: int,
                     b_label: str, b_dir: str, b_n: int,
                     k: int) -> Dict[str, object]:
    u_a = union_top_k(repo_root, a_dir, k)
    u_b = union_top_k(repo_root, b_dir, k)
    inter = u_a & u_b
    j_obs = jaccard(u_a, u_b)
    o_obs = overlap_coefficient(u_a, u_b)
    null = random_overlap_distribution(len(u_a), len(u_b), a_n, b_n,
                                       n_samples=NUM_RANDOM_SAMPLES,
                                       seed=hash((a_label, b_label, k)) % (2 ** 32))
    j_rand_mean = float(null["jaccard"].mean())
    o_rand_mean = float(null["overlap"].mean())
    return {
        "K": k,
        "model_a": a_label,
        "model_b": b_label,
        "n_heads_a": a_n,
        "n_heads_b": b_n,
        "abs_U_a": len(u_a),
        "abs_U_b": len(u_b),
        "intersection": len(inter),
        "head_pop_match": a_n == b_n,
        "jaccard_obs": round(j_obs, 6),
        "jaccard_rand_mean": round(j_rand_mean, 6),
        "jaccard_rand_p2.5": round(float(np.percentile(null["jaccard"], 2.5)), 6),
        "jaccard_rand_p97.5": round(float(np.percentile(null["jaccard"], 97.5)), 6),
        "jaccard_lift": round(j_obs / max(j_rand_mean, 1e-9), 4),
        "jaccard_p": round(float((null["jaccard"] >= j_obs).mean()), 6),
        "overlap_obs": round(o_obs, 6),
        "overlap_rand_mean": round(o_rand_mean, 6),
        "overlap_rand_p2.5": round(float(np.percentile(null["overlap"], 2.5)), 6),
        "overlap_rand_p97.5": round(float(np.percentile(null["overlap"], 97.5)), 6),
        "overlap_lift": round(o_obs / max(o_rand_mean, 1e-9), 4),
        "overlap_p": round(float((null["overlap"] >= o_obs).mean()), 6),
    }


# ─── Reporting ──────────────────────────────────────────────────────────────
def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_plot(path: str, rows: List[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pairs_jaccard: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = {}
    pairs_overlap: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = {}
    for r in rows:
        a, b, k = r["model_a"], r["model_b"], r["K"]
        if a in JACCARD_GROUP and b in JACCARD_GROUP:
            pairs_jaccard.setdefault((a, b), []).append(
                (k, r["jaccard_obs"], r["jaccard_rand_mean"]))
        pairs_overlap.setdefault((a, b), []).append(
            (k, r["overlap_obs"], r["overlap_rand_mean"]))

    # Panel 1: Jaccard for 1024-head models only
    ax = axes[0]
    for (a, b), pts in pairs_jaccard.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys_obs = [p[1] for p in pts]
        ys_rand = [p[2] for p in pts]
        line, = ax.plot(xs, ys_obs, marker="o", label=f"{_short(a)}↔{_short(b)} (obs)")
        ax.plot(xs, ys_rand, color=line.get_color(), linestyle=":", alpha=0.5,
                label=f"{_short(a)}↔{_short(b)} (rand exp)")
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Top-K (per task)")
    ax.set_ylabel("Jaccard of cross-task head-set unions")
    ax.set_title("Primary: Jaccard, 1024-head models only\n{Llama, Mistral, OLMo-Instruct}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # Panel 2: Overlap coefficient for all 4 models
    ax = axes[1]
    for (a, b), pts in pairs_overlap.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys_obs = [p[1] for p in pts]
        ys_rand = [p[2] for p in pts]
        line, = ax.plot(xs, ys_obs, marker="o", label=f"{_short(a)}↔{_short(b)} (obs)")
        ax.plot(xs, ys_rand, color=line.get_color(), linestyle=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Top-K (per task)")
    ax.set_ylabel("Overlap coefficient")
    ax.set_title("Sidebar: overlap coefficient, all 4 models\n(insensitive to head-population mismatch)")
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Cross-model overlap of per-model retrieval-head pools (union over 8 SEC tasks)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _short(label: str) -> str:
    if "Llama" in label:
        return "Llama"
    if "Mistral" in label:
        return "Mistral"
    if "OLMo" in label:
        return "OLMo"
    if "Qwen" in label:
        return "Qwen"
    return label.split("-")[0]


def write_report(path: str, rows: List[Dict[str, object]]) -> None:
    # Summarise: at K=16, what's the Jaccard lift among 1024-head pairs?
    k16 = [r for r in rows if r["K"] == 16]
    matched = [r for r in k16 if r["head_pop_match"]]
    unmatched = [r for r in k16 if not r["head_pop_match"]]

    lines = [
        "# Cross-model union-overlap report",
        "",
        "## Method",
        "",
        "For each model `M` and top-K threshold, we build the union",
        "`U_M(K) = ∪_{t ∈ 8 SEC tasks} top-K heads detected for task t`.",
        "For each pair of models we compute Jaccard `|U_a ∩ U_b| / |U_a ∪ U_b|`",
        "and the overlap coefficient `|U_a ∩ U_b| / min(|U_a|, |U_b|)`,",
        "plus an empirical random-baseline distribution from 1000 random",
        "head-subset pairs sampled at the same sizes from each model's",
        "head population.",
        "",
        "Head populations: 1024 for Llama / Mistral / OLMo-Instruct, 784 for Qwen.",
        "",
        "## Headline (K=16) — same-head-population pairs",
        "",
        "| Pair | \\|U_a\\| | \\|U_b\\| | ∩ | Jaccard obs | Jaccard rand mean | Lift | p (one-sided) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in matched:
        lines.append(
            f"| {_short(r['model_a'])} ↔ {_short(r['model_b'])} | "
            f"{r['abs_U_a']} | {r['abs_U_b']} | {r['intersection']} | "
            f"{r['jaccard_obs']:.3f} | {r['jaccard_rand_mean']:.3f} | "
            f"**{r['jaccard_lift']:.1f}×** | {r['jaccard_p']:.3g} |"
        )

    if unmatched:
        lines.extend([
            "",
            "## Headline (K=16) — cross-population pairs (Qwen vs others)",
            "",
            "Reported using overlap coefficient `|U_a ∩ U_b| / min(|U_a|, |U_b|)`",
            "because Jaccard is awkward across head populations of different size.",
            "",
            "| Pair | \\|U_a\\| | \\|U_b\\| | ∩ | Overlap obs | Overlap rand mean | Lift | p (one-sided) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for r in unmatched:
            lines.append(
                f"| {_short(r['model_a'])} ↔ {_short(r['model_b'])} | "
                f"{r['abs_U_a']} | {r['abs_U_b']} | {r['intersection']} | "
                f"{r['overlap_obs']:.3f} | {r['overlap_rand_mean']:.3f} | "
                f"**{r['overlap_lift']:.1f}×** | {r['overlap_p']:.3g} |"
            )

    lines.extend([
        "",
        "## Verdict template",
        "",
        "If observed lifts at K=16 are ≥ 5×, the paper claim becomes",
        "*'Cross-model retrieval-head-pool overlap is N× above random*",
        "*expectation while per-task efficacy correlations stay below 0.15 R² —*",
        "*heads are SHARED SUBSTRATE, DIFFERENTLY USED.'*",
        "",
        "If observed lifts are ≤ 2×, the cross-model substrate claim weakens",
        "and the §4 prose pivots to per-model phenomenon.",
        "",
        "## Note on existing files in `results/cross_model/`",
        "",
        "This analysis is **distinct from**:",
        "- `cross_model_head_overlap.csv` (within-model SEC/LME/NQ comparison).",
        "- `cross_model_jaccard_*.csv` (per-task within-model Jaccard).",
        "",
        "All three live in the same directory because they all touch *some* notion",
        "of head-set overlap; this one is the cross-model union comparison.",
        "",
        "## All-K full table",
        "",
        "See `cross_model_union_overlap.csv` for the K ∈ {8, 16, 32, 48, 64, 96, 128}",
        "values for every pair.",
    ])
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "cross_model")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[cross_model_union_overlap] repo_root = {repo_root}")
    print(f"[cross_model_union_overlap] output_dir = {output_dir}")
    print()

    rows: List[Dict[str, object]] = []
    for (a_label, a_dir, a_n), (b_label, b_dir, b_n) in combinations(MODELS, 2):
        for k in KS:
            row = compute_pair_row(repo_root, a_label, a_dir, a_n,
                                   b_label, b_dir, b_n, k)
            rows.append(row)

    write_csv(os.path.join(output_dir, "cross_model_union_overlap.csv"), rows)
    write_plot(os.path.join(output_dir, "cross_model_union_overlap_plot.png"), rows)
    write_report(os.path.join(output_dir, "cross_model_union_overlap_REPORT.md"), rows)

    # Echo headline
    print("=" * 78)
    print("Cross-model union overlap — K=16 headline")
    print("=" * 78)
    print(f"\n{'Pair':<28} {'|U_a|':>6} {'|U_b|':>6} {'∩':>4} "
          f"{'J_obs':>7} {'J_rand':>7} {'lift':>7} "
          f"{'O_obs':>7} {'O_rand':>7} {'O_lift':>7}")
    for r in rows:
        if r["K"] != 16:
            continue
        print(f"{_short(r['model_a'])+'↔'+_short(r['model_b']):<28} "
              f"{r['abs_U_a']:>6} {r['abs_U_b']:>6} {r['intersection']:>4} "
              f"{r['jaccard_obs']:>7.3f} {r['jaccard_rand_mean']:>7.3f} "
              f"{r['jaccard_lift']:>7.2f} "
              f"{r['overlap_obs']:>7.3f} {r['overlap_rand_mean']:>7.3f} "
              f"{r['overlap_lift']:>7.2f}")
    print()
    print(f"Wrote {os.path.join(output_dir, 'cross_model_union_overlap.csv')}")
    print(f"Wrote {os.path.join(output_dir, 'cross_model_union_overlap_plot.png')}")
    print(f"Wrote {os.path.join(output_dir, 'cross_model_union_overlap_REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
