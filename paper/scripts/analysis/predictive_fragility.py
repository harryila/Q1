#!/usr/bin/env python3
"""Predictive fragility check (Task 2 of the pre-submission analyses).

Goal: rule out "fragility = baseline difficulty in disguise" before claiming
that some SEC tasks are structurally fragile under head ablation.

We define **fragility[t]** as the cross-model mean of target sensitivity
averaged over early K:

    sensitivity[t][K] = mean over sources s of drop[s][t][K]
    fragility[t]      = mean over models M of mean over K in {8, 16} of sensitivity_M[t][K]

and regress it against three predictors that vary per task:

    P1  baseline accuracy[t]
        per-task K=0 accuracy from QRScore-SEC_results.json, averaged across
        the 3 models.

    P2  query-to-answer token distance[t]
        For each NIAH instance: minimum word-distance from any task keyword's
        first occurrence in the haystack to the position of the gold answer.
        Median over instances per task.

        This replaces the broken "needle depth from edges" predictor: the
        upstream methodology fixes needle_position="middle" for every
        instance, so the depth metric has near-zero variance. Word-distance
        is a CPU-only proxy for tokenizer-distance (~1.3 tokens/word for
        English; relative ordering across tasks is preserved).

    P3  gold answer token length[t]
        Median word count of the gold answer string per task.

Decision rule on |r_baseline| (predictor 1 only is gating):

    |r| < 0.4         -> claim sharpens to "fragility is not difficulty"
    0.4 <= |r| <= 0.7 -> downgrade to "fragility partially decouples from difficulty"
    |r| > 0.7         -> claim is at risk; re-orient analysis section

Outputs:

    paper/results/fragility_predictors/correlations.csv
    paper/results/fragility_predictors/scatter_plots.png
    paper/results/fragility_predictors/REPORT.md

Run:

    python scripts/analysis/predictive_fragility.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────
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

MODELS = [
    ("Llama-3.1-8B-Instruct", "results/llama_3_1_8B_instruct"),
    ("Qwen2.5-7B-Instruct",   "results/qwen_2_5_7B_instruct"),
    ("Mistral-7B-Instruct",   "results/mistral_7B_instruct"),
]

EARLY_KS = (8, 16)

# Task-specific content keywords extracted from the question text.
# Stopwords ("what", "is", "the") removed; stems chosen so that "headquarters"
# matches "headquartered", "incorporation" matches "incorporated", etc.
TASK_KEYWORDS: Dict[str, List[str]] = {
    "ceo_lastname":         ["ceo", "chief executive", "officer", "lastname", "last name"],
    "employees_count_total": ["employee", "employees", "total"],
    "headquarters_city":    ["headquarter", "city", "principal office"],
    "headquarters_state":   ["headquarter", "state", "principal office"],
    "holder_record_amount": ["holder", "record"],
    "incorporation_state":  ["incorporat", "state", "organized in", "organised in"],
    "incorporation_year":   ["incorporat", "organized", "organised"],
    "registrant_name":      ["registrant", "company name", "name"],
}

# Decision thresholds (absolute Pearson r against baseline accuracy).
THRESHOLD_SHARP = 0.4
THRESHOLD_FAIL  = 0.7
NUM_BOOTSTRAP   = 1000


# ─── Helpers ───────────────────────────────────────────────────────────────
def words(s: str) -> List[str]:
    """Whitespace + simple-punctuation tokenizer (CPU-only proxy for LLM tokens)."""
    return re.findall(r"\w+", s.lower())


def first_word_index(haystack_words: List[str], pattern: str) -> int:
    """Return word index of the first occurrence of pattern (case-insensitive,
    substring-matched at the word level), or -1 if not found.

    Patterns can be multi-word (e.g. "principal office"); we slide a window
    of the right size.
    """
    p_words = pattern.lower().split()
    n = len(p_words)
    if n == 0:
        return -1
    if n == 1:
        # substring match at the word level (allows stems like "incorporat"
        # to match "incorporated", "incorporation")
        for i, w in enumerate(haystack_words):
            if p_words[0] in w:
                return i
        return -1
    # Multi-word pattern: exact contiguous match
    target = " ".join(p_words)
    for i in range(len(haystack_words) - n + 1):
        if " ".join(haystack_words[i:i + n]) == target:
            return i
    return -1


def keyword_to_answer_distance(context: str, answer: str, keywords: List[str]) -> float:
    """Return min word-distance from any keyword's first occurrence to the
    first occurrence of the answer in the context. NaN if either is missing."""
    ctx_words = words(context)
    ans_idx = first_word_index(ctx_words, answer)
    if ans_idx < 0:
        # Try matching just the first word of a multi-word answer
        ans_first = words(answer)
        if not ans_first:
            return float("nan")
        ans_idx = first_word_index(ctx_words, ans_first[0])
        if ans_idx < 0:
            return float("nan")
    kw_indices: List[int] = []
    for kw in keywords:
        idx = first_word_index(ctx_words, kw)
        if idx >= 0:
            kw_indices.append(idx)
    if not kw_indices:
        return float("nan")
    return float(min(abs(idx - ans_idx) for idx in kw_indices))


def bootstrap_correlation_ci(x: np.ndarray, y: np.ndarray, n_iter: int = 1000,
                             seed: int = 1) -> Tuple[float, float, float, float]:
    """Return (pearson_r, p_value, ci_low, ci_high) using percentile bootstrap on n_iter resamples."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r_obs, p_obs = pearsonr(x, y)
    rs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            rs.append(pearsonr(xs, ys)[0])
    rs = np.array(rs)
    return float(r_obs), float(p_obs), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# ─── Data loaders ──────────────────────────────────────────────────────────
def load_fragility(repo_root: str) -> Dict[str, float]:
    """Return per-task fragility = cross-model mean of mean-over-EARLY_KS sensitivity."""
    per_model_per_task: Dict[str, Dict[str, float]] = {t: {} for t in TASKS}
    for model_label, model_dir in MODELS:
        matrix_path = os.path.join(repo_root, model_dir, "transfer", "cross_task_transfer_matrix.json")
        with open(matrix_path) as fh:
            m = json.load(fh)
        for t in m["targets"]:
            sens_per_k: List[float] = []
            for k in EARLY_KS:
                drops = [m["results"][s][t]["by_k"][str(k)]["drop_from_k0"]
                         for s in m["sources"]]
                sens_per_k.append(float(np.mean(drops)))
            per_model_per_task[t][model_label] = float(np.mean(sens_per_k))
    return {t: float(np.mean(list(per_model_per_task[t].values()))) for t in TASKS}


def load_baseline_accuracy(repo_root: str) -> Dict[str, float]:
    """Return per-task baseline (K=0) accuracy averaged across 3 models."""
    per_model_per_task: Dict[str, Dict[str, float]] = {t: {} for t in TASKS}
    for model_label, model_dir in MODELS:
        results_path = os.path.join(repo_root, model_dir, "raw_results", "QRScore-SEC_results.json")
        with open(results_path) as fh:
            d = json.load(fh)
        ptc = d["per_task_curves"]
        for t in TASKS:
            per_model_per_task[t][model_label] = float(ptc[t]["0"])
    return {t: float(np.mean(list(per_model_per_task[t].values()))) for t in TASKS}


def load_query_answer_distance(repo_root: str) -> Dict[str, float]:
    """Median word-distance from task keyword to gold answer per task."""
    out: Dict[str, float] = {}
    for t in TASKS:
        path = os.path.join(repo_root, "data", "niah_input", f"{t}_test.json")
        with open(path) as fh:
            instances = json.load(fh)
        distances = []
        for inst in instances:
            d = keyword_to_answer_distance(inst["context"], inst["needle_value"],
                                           TASK_KEYWORDS[t])
            if not np.isnan(d):
                distances.append(d)
        out[t] = float(np.median(distances)) if distances else float("nan")
    return out


def load_answer_token_length(repo_root: str) -> Dict[str, float]:
    """Median word count of the gold answer string per task."""
    out: Dict[str, float] = {}
    for t in TASKS:
        path = os.path.join(repo_root, "data", "niah_input", f"{t}_test.json")
        with open(path) as fh:
            instances = json.load(fh)
        lengths = [len(words(inst["needle_value"])) for inst in instances]
        out[t] = float(np.median(lengths))
    return out


# ─── Reporting ─────────────────────────────────────────────────────────────
def decide(abs_r: float) -> str:
    if abs_r < THRESHOLD_SHARP:
        return "SHARP"
    if abs_r <= THRESHOLD_FAIL:
        return "PARTIAL"
    return "AT_RISK"


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_report(path: str, fragility: Dict[str, float],
                 predictors: Dict[str, Dict[str, float]],
                 correlations: Dict[str, Dict[str, float]]) -> None:
    p1_abs_r = abs(correlations["baseline_accuracy"]["pearson_r"])
    verdict_code = decide(p1_abs_r)
    verdict_text = {
        "SHARP":   "**Verdict: SHARP** — fragility does not correlate with baseline accuracy. The structural-fragility claim survives.",
        "PARTIAL": "**Verdict: PARTIAL** — fragility partially decouples from baseline difficulty. Downgrade the claim and report this correlation in §5 limitations.",
        "AT_RISK": "**Verdict: AT_RISK** — fragility tracks baseline accuracy too closely. Re-orient §3 around joint difficulty/fragility analysis.",
    }[verdict_code]

    lines = [
        "# Predictive fragility check — REPORT",
        "",
        "## Decision rule",
        "",
        "| |r_baseline| | Verdict | Action |",
        "| --- | --- | --- |",
        f"| < {THRESHOLD_SHARP} | SHARP | Claim sharpens to 'fragility is not difficulty in disguise' |",
        f"| {THRESHOLD_SHARP} ≤ x ≤ {THRESHOLD_FAIL} | PARTIAL | Downgrade to 'partially decouples'; report in §5 limitations |",
        f"| > {THRESHOLD_FAIL} | AT_RISK | Re-orient §3 around joint difficulty/fragility analysis |",
        "",
        f"Observed `|r_baseline| = {p1_abs_r:.3f}` → {verdict_text}",
        "",
        "## Per-task values",
        "",
        "| Task | Fragility | Baseline acc | Q→A token-dist | Answer tok-len |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for t in TASKS:
        lines.append(
            f"| `{t}` | {fragility[t]:.3f} | "
            f"{predictors['baseline_accuracy'][t]:.3f} | "
            f"{predictors['query_answer_distance'][t]:.0f} | "
            f"{predictors['answer_token_length'][t]:.1f} |"
        )

    lines.extend(["", "## Correlations vs fragility", "",
                  "| Predictor | Pearson r | 95% CI | Pearson p | Spearman ρ | Spearman p |",
                  "| --- | ---: | --- | ---: | ---: | ---: |"])
    for p_label, c in correlations.items():
        lines.append(
            f"| {p_label} | {c['pearson_r']:+.3f} | "
            f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
            f"{c['pearson_p']:.3g} | {c['spearman_r']:+.3f} | {c['spearman_p']:.3g} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        f"- Fragility is `mean_M mean_{{K∈{list(EARLY_KS)}}} sensitivity_M[t][K]` "
        f"with `sensitivity = mean_s drop[s][t][K]`. Computed on the 3 transfer matrices in `results/<model>/transfer/`.",
        "- Baseline accuracy is `K=0` per-task accuracy from `QRScore-SEC_results.json`, averaged across the 3 models.",
        "- Query→answer token distance is the median over instances of the minimum word-distance from any task-keyword's first occurrence to the answer's first occurrence in the haystack. Word-distance is a CPU-only proxy for LLM-token-distance (~1.3 tokens/word; relative ordering preserved).",
        "- Answer token length is the median word count of `needle_value` per task.",
        "- All p-values reported are from 2-sided Pearson/Spearman with `n = 8` tasks; bootstrap CI uses 1000 task-resamples (low power; treat CIs as descriptive, not confirmatory).",
        "",
        "## Original (broken) predictor",
        "",
        "The earlier draft proposed `Predictor 2 = min(needle_char_offset, total_chars - needle_char_offset)` (depth from haystack edges). Every instance in `data/niah_input/*_test.json` has `needle_position == \"middle\"` (we verified all 8 files), so this metric is constant across instances and tasks and has zero variance — it cannot correlate with anything. Replaced with query→answer token distance, which is the within-document retrieval-difficulty proxy that does vary with task structure.",
    ])

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_scatter(path: str, fragility: Dict[str, float],
                  predictors: Dict[str, Dict[str, float]],
                  correlations: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    p_labels = ["baseline_accuracy", "query_answer_distance", "answer_token_length"]
    p_titles = ["Baseline acc (K=0)", "Q→A word distance (median)", "Answer word count (median)"]
    y = np.array([fragility[t] for t in TASKS])
    for ax, p_label, p_title in zip(axes, p_labels, p_titles):
        x = np.array([predictors[p_label][t] for t in TASKS])
        ax.scatter(x, y, s=60)
        for t, xi, yi in zip(TASKS, x, y):
            ax.annotate(t, (xi, yi), fontsize=7, alpha=0.8,
                        xytext=(4, 4), textcoords="offset points")
        c = correlations[p_label]
        ax.set_xlabel(p_title)
        ax.set_ylabel("Fragility (cross-model mean sensitivity, K∈{8,16})")
        ax.set_title(f"{p_label}\nr={c['pearson_r']:+.2f}  ρ={c['spearman_r']:+.2f}",
                     fontsize=10)
        # OLS line for reference
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, "r--", alpha=0.4, lw=1)
    fig.suptitle("Fragility predictors", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_root", default=None,
                   help="Path to the paper/ directory. Defaults to script's grandparent.")
    p.add_argument("--output_dir", default=None,
                   help="Override output directory. Defaults to "
                        "<repo_root>/results/fragility_predictors/")
    args = p.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "fragility_predictors")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[predictive_fragility] repo_root = {repo_root}")
    print(f"[predictive_fragility] output_dir = {output_dir}")
    print()

    print("[1/4] Loading per-task fragility from 3 transfer matrices ...")
    fragility = load_fragility(repo_root)
    print("[2/4] Loading per-task baseline accuracy ...")
    baseline = load_baseline_accuracy(repo_root)
    print("[3/4] Computing query-to-answer token distance from niah_input ...")
    qad = load_query_answer_distance(repo_root)
    print("[4/4] Computing answer token length ...")
    atl = load_answer_token_length(repo_root)

    predictors = {
        "baseline_accuracy":     baseline,
        "query_answer_distance": qad,
        "answer_token_length":   atl,
    }

    y = np.array([fragility[t] for t in TASKS])

    correlations: Dict[str, Dict[str, float]] = {}
    for p_label, vals in predictors.items():
        x = np.array([vals[t] for t in TASKS])
        if np.any(np.isnan(x)):
            print(f"  [warn] {p_label} has NaN values; skipping correlation")
            correlations[p_label] = {"pearson_r": float("nan"), "pearson_p": float("nan"),
                                     "ci_low": float("nan"), "ci_high": float("nan"),
                                     "spearman_r": float("nan"), "spearman_p": float("nan")}
            continue
        r, p_val, lo, hi = bootstrap_correlation_ci(x, y, NUM_BOOTSTRAP, seed=42)
        rho, sp = spearmanr(x, y)
        correlations[p_label] = {"pearson_r": r, "pearson_p": p_val,
                                 "ci_low": lo, "ci_high": hi,
                                 "spearman_r": float(rho), "spearman_p": float(sp)}

    # ── Write CSV ──
    rows: List[Dict[str, object]] = []
    for t in TASKS:
        rows.append({
            "task": t,
            "fragility": round(fragility[t], 6),
            "baseline_accuracy": round(baseline[t], 6),
            "query_answer_distance": round(qad[t], 3),
            "answer_token_length": round(atl[t], 2),
        })
    rows.append({"task": "_summary_", "fragility": "", "baseline_accuracy": "",
                 "query_answer_distance": "", "answer_token_length": ""})
    for p_label, c in correlations.items():
        rows.append({
            "task": f"corr:{p_label}",
            "fragility": "",
            "baseline_accuracy": f"r={c['pearson_r']:+.3f} p={c['pearson_p']:.3g}",
            "query_answer_distance": f"CI=[{c['ci_low']:+.3f},{c['ci_high']:+.3f}]",
            "answer_token_length": f"rho={c['spearman_r']:+.3f} p={c['spearman_p']:.3g}",
        })
    write_csv(os.path.join(output_dir, "correlations.csv"), rows)

    # ── Plot + report ──
    write_scatter(os.path.join(output_dir, "scatter_plots.png"),
                  fragility, predictors, correlations)
    write_report(os.path.join(output_dir, "REPORT.md"),
                 fragility, predictors, correlations)

    # ── Echo headline ──
    print()
    print("=" * 78)
    print("Predictive fragility check — headline numbers")
    print("=" * 78)
    print(f"\n{'Task':<25} {'fragility':>10} {'baseline':>10} {'q→a dist':>10} {'ans len':>10}")
    for t in TASKS:
        print(f"{t:<25} {fragility[t]:>10.3f} {baseline[t]:>10.3f} "
              f"{qad[t]:>10.0f} {atl[t]:>10.1f}")
    print(f"\n{'Predictor':<25} {'pearson r':>12} {'95% CI':<22} {'pearson p':>10} {'spearman ρ':>12}")
    for p_label, c in correlations.items():
        ci = f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}]"
        print(f"{p_label:<25} {c['pearson_r']:>+12.3f} {ci:<22} "
              f"{c['pearson_p']:>10.3g} {c['spearman_r']:>+12.3f}")

    p1_r = correlations["baseline_accuracy"]["pearson_r"]
    print()
    print(f"Decision (gating predictor 1): |r_baseline| = {abs(p1_r):.3f} → {decide(abs(p1_r))}")
    print()
    print(f"Wrote {os.path.join(output_dir, 'correlations.csv')}")
    print(f"Wrote {os.path.join(output_dir, 'scatter_plots.png')}")
    print(f"Wrote {os.path.join(output_dir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
