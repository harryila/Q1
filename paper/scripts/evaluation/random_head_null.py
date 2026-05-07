#!/usr/bin/env python3
"""Random source-head null distribution (Task 3).

GPU-required. Runs the same head-masking + generation pipeline as
``run_ablation.py``, but with **random** 16-head subsets in place of the
QRScore-detected ones, to build a null distribution of per-task efficacy.

Reviewer objection this pre-empts:

    "Any partially-overlapping detection method would look like this — your
    QRScore detection might just be picking 16 heads with non-zero relevance
    by chance."

This script samples N random 16-head subsets from each model's full head
population and for each subset measures per-task drop on all 8 SEC tasks.
The resulting per-task null distribution is then compared against the
**observed per-task efficacy** from the existing
``cross_task_transfer_matrix.json``.

Per-sample procedure:
    1. Sample 16 random (layer, head) pairs from the model's head population
       with rng = np.random.default_rng(seed_base + i).
    2. Knock those 16 heads out via the same forward-pre-hook on o_proj that
       run_ablation.py uses.
    3. Generate on all 8 NIAH tasks × 24 instances each = 192 generations.
    4. Compute per-task accuracy + per-task drop_from_k0 (relative to
       baseline accuracy at K=0 from --baseline_accuracy_json or recomputed).
    5. For each of the 8 tasks, compute the "if-this-subset-were-source"
       efficacy: mean over t' != t of per-task drop[t']. (Each random subset
       therefore yields 8 efficacy values — one nominal-source per task.)

Aggregation across N subsets gives, per task, a null distribution of N
efficacy values. The observed task-source efficacy from
cross_task_transfer_matrix.json is then located in this null.

GPU envelope (set by smoke test, not pre-committed):
    Run --num_random_subsets 3 --max_instances_per_task 24 first to measure
    per-instance generation time; then choose deployment tier:

        ≤ 1 sec/instance  -> full 100x3 models (8-12 GPU-hours total)
        1-2 sec/instance  -> full 100x3 models (16-32 GPU-hours)
        2-4 sec/instance  -> 50x3 models       (16-32 GPU-hours)
        > 4 sec/instance  -> Llama-only at 50  (~13 GPU-hours)

Outputs (per model, in --output_dir):
    null_samples.json   list of N dicts {seed, heads, per_task_drop, ...}
    null_summary.csv    per task: mean, std, p2.5, p50, p97.5 of drop/efficacy
    observed_vs_null.csv  observed efficacy vs null percentile per task
    per_task_null_vs_observed.png  per-task null violin + observed marker
    REPORT.md           verdict + chosen-tier provenance

CLI example:
    python scripts/evaluation/random_head_null.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --niah_dir data/niah_input \\
        --transfer_matrix results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json \\
        --num_random_subsets 100 \\
        --K 16 \\
        --max_instances_per_task 24 \\
        --max_context_tokens 8192 \\
        --output_dir results/random_head_null/meta-llama__Llama-3.1-8B-Instruct \\
        --seed_base 1000

Smoke run:
    python scripts/evaluation/random_head_null.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --num_random_subsets 3 \\
        --K 16 \\
        --max_instances_per_task 24 \\
        --output_dir results/random_head_null/_smoke \\
        --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Numpy is required at parse time; torch/transformers only when running.
import numpy as np


# ─── Lazy imports for the GPU bits ───────────────────────────────────────────
def _lazy_torch():
    import torch  # noqa: WPS433
    return torch


def _lazy_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
    return AutoModelForCausalLM, AutoTokenizer


# ─── Constants ───────────────────────────────────────────────────────────────
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


# ─── Lift forward-hook utilities from run_ablation.py to keep apples-to-apples ──
def install_head_masking(model):
    from collections import defaultdict
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Unsupported model structure for head masking")

    def make_hook(attn_module):
        def hook(_module, args):
            x = args[0]
            indices = attn_module._masked_head_indices
            if indices is None:
                return args
            bsz, seq_len, _ = x.shape
            x = x.view(bsz, seq_len, -1, head_dim)
            x[:, :, indices, :] = 0
            return (x.view(bsz, seq_len, -1),) + args[1:]
        return hook

    for layer in layers:
        attn = layer.self_attn
        attn._masked_head_indices = None
        attn.o_proj.register_forward_pre_hook(make_hook(attn))

    def set_head_mask(masked_heads):
        for layer in layers:
            layer.self_attn._masked_head_indices = None
        if not masked_heads:
            return
        per_layer = defaultdict(list)
        for layer_idx, head_idx in masked_heads:
            per_layer[layer_idx].append(head_idx)
        for layer_idx, head_indices in per_layer.items():
            if layer_idx < len(layers):
                layers[layer_idx].self_attn._masked_head_indices = head_indices

    model.set_head_mask = set_head_mask
    return len(layers)


def normalize_answer(s: str) -> str:
    import re
    if s is None or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"[,.;:!?\"']+$", "", s)
    return " ".join(s.split())


def extract_short_answer(text: str) -> str:
    import re
    if not text:
        return ""
    text = text.strip().split("\n")[0].strip()
    text = re.split(r"[.!?]\s", text)[0].strip()
    return re.sub(r"[.!?]+$", "", text)


def answers_match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    return p == g or g in p or p in g


def build_prompt(context: str, question: str) -> str:
    return (
        "Read the following document and answer the question.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely with just the answer value."
    )


def generate_answer(model, tokenizer, context: str, question: str,
                    max_new_tokens: int, max_context_tokens: int) -> str:
    torch = _lazy_torch()
    tokenizer.truncation_side = "left"
    prompt_text = build_prompt(context, question)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=True,
            truncation=True, max_length=max_context_tokens,
            return_dict=True, return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=max_context_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(model.config, "pad_token_id", None) or tokenizer.eos_token_id,
            use_cache=True,
        )
    start = inputs["input_ids"].shape[1]
    new_tokens = out[0][start:]
    return extract_short_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


# ─── Data ────────────────────────────────────────────────────────────────────
def load_niah(niah_dir: str, tasks: List[str], cap_per_task: int) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for t in tasks:
        path = os.path.join(niah_dir, f"{t}_test.json")
        with open(path) as fh:
            instances = json.load(fh)
        out[t] = instances[:cap_per_task]
    return out


def evaluate_subset(model, tokenizer, head_set: List[Tuple[int, int]],
                    instances: Dict[str, List[dict]],
                    max_new_tokens: int, max_context_tokens: int) -> Tuple[Dict[str, float], float]:
    """Return per-task accuracy + total elapsed seconds."""
    if hasattr(model, "set_head_mask"):
        model.set_head_mask(head_set if head_set else None)
    out: Dict[str, float] = {}
    t0 = time.time()
    n_total = 0
    for task, items in instances.items():
        correct = 0
        for inst in items:
            pred = generate_answer(model, tokenizer,
                                   inst["context"], inst["question"],
                                   max_new_tokens, max_context_tokens)
            if answers_match(pred, inst["needle_value"]):
                correct += 1
            n_total += 1
        out[task] = correct / max(len(items), 1)
    return out, time.time() - t0


def random_head_subset(rng, n_layers: int, n_heads_per_layer: int, k: int) -> List[Tuple[int, int]]:
    total = n_layers * n_heads_per_layer
    flat = rng.choice(total, size=k, replace=False)
    return [(int(i // n_heads_per_layer), int(i % n_heads_per_layer)) for i in sorted(flat.tolist())]


# ─── Reporting ──────────────────────────────────────────────────────────────
def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def task_observed_efficacy(transfer_matrix_path: str, k: int) -> Dict[str, float]:
    """Read observed per-task efficacy from cross_task_transfer_matrix.json.

    efficacy[s] = mean over t != s of drop[s][t][k]
    """
    with open(transfer_matrix_path) as fh:
        m = json.load(fh)
    sources = m["sources"]
    out: Dict[str, float] = {}
    for s in sources:
        targets = [t for t in m["targets"] if t != s]
        drops = [m["results"][s][t]["by_k"][str(k)]["drop_from_k0"] for t in targets]
        out[s] = float(np.mean(drops))
    return out


def write_plot(path: str, null_efficacy: Dict[str, np.ndarray],
               observed_efficacy: Dict[str, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    pos = list(range(len(TASKS)))
    data = [null_efficacy[t] for t in TASKS]
    parts = ax.violinplot(data, positions=pos, showmeans=True, showmedians=True)
    for i, t in enumerate(TASKS):
        obs = observed_efficacy.get(t, np.nan)
        if not np.isnan(obs):
            ax.scatter([i], [obs], color="red", s=80, zorder=5,
                       label="observed" if i == 0 else None)
    ax.set_xticks(pos)
    ax.set_xticklabels([t.replace("_", "\n") for t in TASKS], fontsize=8)
    ax.set_ylabel("Per-task efficacy (off-target mean drop)")
    ax.set_title("Random-16-head null vs observed efficacy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--niah_dir", default="data/niah_input")
    p.add_argument("--transfer_matrix", default=None,
                   help="Path to cross_task_transfer_matrix.json for observed efficacy "
                        "(required unless --skip_observed)")
    p.add_argument("--baseline_accuracy_json", default=None,
                   help="Optional QRScore-SEC_results.json from which to read "
                        "per-task K=0 accuracy. If absent, baseline is recomputed "
                        "(adds 1 unablated pass).")
    p.add_argument("--num_random_subsets", type=int, default=100)
    p.add_argument("--K", type=int, default=16, help="Number of random heads per subset")
    p.add_argument("--max_instances_per_task", type=int, default=24)
    p.add_argument("--max_context_tokens", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed_base", type=int, default=1000)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: skips the full ablation set; useful with "
                        "--num_random_subsets 3 to verify output schema and time")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--dtype", default="bfloat16",
                   help="Torch dtype for model load (bfloat16 / float16 / float32)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_observed if hasattr(args, "skip_observed") else False:
        pass  # placeholder for future toggling

    # ── Load model ──
    torch = _lazy_torch()
    AutoModelForCausalLM, AutoTokenizer = _lazy_transformers()
    print(f"[random_head_null] loading {args.model}")
    if args.device == "auto":
        device_map = "auto"
    else:
        device_map = None
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device_map,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    if device_map is None:
        model.to(args.device)
    n_layers = install_head_masking(model)
    n_heads = model.config.num_attention_heads
    print(f"[random_head_null] n_layers={n_layers} n_heads_per_layer={n_heads}")
    model.eval()

    # ── Load NIAH ──
    print(f"[random_head_null] loading NIAH from {args.niah_dir} (cap={args.max_instances_per_task})")
    instances = load_niah(args.niah_dir, TASKS, args.max_instances_per_task)
    n_total = sum(len(v) for v in instances.values())

    # ── Baseline (K=0) ──
    if args.baseline_accuracy_json and os.path.exists(args.baseline_accuracy_json):
        with open(args.baseline_accuracy_json) as fh:
            d = json.load(fh)
        baseline = {t: float(d["per_task_curves"][t]["0"]) for t in TASKS}
        print(f"[random_head_null] baseline loaded from {args.baseline_accuracy_json}")
    else:
        print("[random_head_null] computing baseline (no head mask) ...")
        t_start = time.time()
        baseline, _ = evaluate_subset(model, tokenizer, [], instances,
                                      args.max_new_tokens, args.max_context_tokens)
        elapsed = time.time() - t_start
        per_inst = elapsed / n_total
        print(f"[random_head_null] baseline computed in {elapsed:.1f}s "
              f"({per_inst:.2f}s/instance over {n_total} instances)")

    # ── Random subsets ──
    print(f"[random_head_null] running {args.num_random_subsets} random {args.K}-head subsets")
    rng_master = np.random.default_rng(args.seed_base)
    samples: List[dict] = []
    null_drop: Dict[str, List[float]] = {t: [] for t in TASKS}
    null_eff: Dict[str, List[float]] = {t: [] for t in TASKS}
    total_start = time.time()
    instance_count = 0

    for i in range(args.num_random_subsets):
        rng = np.random.default_rng(args.seed_base + i + 1)
        heads = random_head_subset(rng, n_layers, n_heads, args.K)
        sample_start = time.time()
        per_task_acc, _ = evaluate_subset(model, tokenizer, heads, instances,
                                          args.max_new_tokens, args.max_context_tokens)
        sample_elapsed = time.time() - sample_start
        n_inst_this_sample = sum(len(v) for v in instances.values())
        instance_count += n_inst_this_sample

        per_task_drop = {t: baseline[t] - per_task_acc[t] for t in TASKS}
        # Source-of-efficacy convention: efficacy[t_nominal] = mean over t' != t of drop[t']
        per_task_efficacy: Dict[str, float] = {}
        for t in TASKS:
            offs = [per_task_drop[tp] for tp in TASKS if tp != t]
            per_task_efficacy[t] = float(np.mean(offs))

        for t in TASKS:
            null_drop[t].append(per_task_drop[t])
            null_eff[t].append(per_task_efficacy[t])

        samples.append({
            "i": i,
            "seed": args.seed_base + i + 1,
            "heads": [f"{l}-{h}" for l, h in heads],
            "per_task_acc": per_task_acc,
            "per_task_drop": per_task_drop,
            "per_task_efficacy": per_task_efficacy,
            "elapsed_seconds": round(sample_elapsed, 3),
        })
        print(f"  sample {i+1}/{args.num_random_subsets}: {sample_elapsed:.1f}s "
              f"({sample_elapsed/n_inst_this_sample:.2f}s/instance)")

    total_elapsed = time.time() - total_start
    per_inst_global = total_elapsed / max(instance_count, 1)
    print(f"\n[random_head_null] all samples done. {total_elapsed:.1f}s total "
          f"({per_inst_global:.2f}s/instance over {instance_count} instances)")

    # ── Observed efficacy ──
    observed: Dict[str, float] = {}
    if args.transfer_matrix and os.path.exists(args.transfer_matrix):
        observed = task_observed_efficacy(args.transfer_matrix, args.K)
        print(f"[random_head_null] observed efficacy loaded from {args.transfer_matrix}")
    else:
        print("[random_head_null] no transfer matrix supplied; observed efficacy will be empty")

    # ── Outputs ──
    with open(os.path.join(args.output_dir, "null_samples.json"), "w") as fh:
        json.dump({
            "model": args.model,
            "K": args.K,
            "num_random_subsets": args.num_random_subsets,
            "max_instances_per_task": args.max_instances_per_task,
            "seed_base": args.seed_base,
            "baseline_accuracy_per_task": baseline,
            "per_instance_seconds_global": per_inst_global,
            "samples": samples,
        }, fh, indent=2)

    summary_rows: List[Dict[str, object]] = []
    obs_rows: List[Dict[str, object]] = []
    for t in TASKS:
        drops = np.array(null_drop[t])
        effs = np.array(null_eff[t])
        summary_rows.append({
            "task": t,
            "null_drop_mean": round(float(drops.mean()), 6),
            "null_drop_p2.5": round(float(np.percentile(drops, 2.5)), 6),
            "null_drop_p50":  round(float(np.percentile(drops, 50)), 6),
            "null_drop_p97.5": round(float(np.percentile(drops, 97.5)), 6),
            "null_eff_mean": round(float(effs.mean()), 6),
            "null_eff_p2.5": round(float(np.percentile(effs, 2.5)), 6),
            "null_eff_p50":  round(float(np.percentile(effs, 50)), 6),
            "null_eff_p97.5": round(float(np.percentile(effs, 97.5)), 6),
        })
        obs = observed.get(t, float("nan"))
        if not np.isnan(obs):
            pct = float(((effs <= obs).sum()) / len(effs))
        else:
            pct = float("nan")
        obs_rows.append({
            "task": t,
            "observed_efficacy": round(obs, 6) if not np.isnan(obs) else "",
            "null_eff_mean": round(float(effs.mean()), 6),
            "null_eff_p95": round(float(np.percentile(effs, 95)), 6),
            "observed_percentile": round(pct, 4) if not np.isnan(pct) else "",
            "above_p95": bool(obs > np.percentile(effs, 95)) if not np.isnan(obs) else "",
        })

    write_csv(os.path.join(args.output_dir, "null_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "observed_vs_null.csv"), obs_rows)

    null_eff_arrays = {t: np.array(null_eff[t]) for t in TASKS}
    write_plot(os.path.join(args.output_dir, "per_task_null_vs_observed.png"),
               null_eff_arrays, observed)

    # ── REPORT.md ──
    n_above = sum(1 for r in obs_rows
                  if isinstance(r["above_p95"], bool) and r["above_p95"])
    n_with_obs = sum(1 for r in obs_rows if r["observed_efficacy"] != "")
    lines = [
        "# Random-head null distribution — REPORT",
        "",
        f"Model: `{args.model}`",
        f"K (heads per random subset): {args.K}",
        f"Number of random subsets: {args.num_random_subsets}",
        f"Instances per task (cap): {args.max_instances_per_task}",
        f"Seed base: {args.seed_base}",
        f"Per-instance generation time (global): {per_inst_global:.2f}s",
        "",
        "## Headline (observed efficacy vs random-16-head null)",
        "",
        f"`{n_above}/{n_with_obs}` tasks have observed efficacy above the 95th percentile of the null.",
        "",
        "| Task | Observed | Null mean | Null p95 | Pctile of obs in null | > p95 |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in obs_rows:
        flag = "Y" if (isinstance(r["above_p95"], bool) and r["above_p95"]) else "N"
        obs_s = f"{r['observed_efficacy']:.3f}" if r["observed_efficacy"] != "" else "—"
        pct_s = f"{r['observed_percentile']*100:.1f}%" if r["observed_percentile"] != "" else "—"
        lines.append(f"| `{r['task']}` | {obs_s} | {r['null_eff_mean']:.3f} | "
                     f"{r['null_eff_p95']:.3f} | {pct_s} | {flag} |")
    lines.extend([
        "",
        "## Verdict",
        "",
        f"- If `n_above >= 7/8`: QRScore detection is unambiguously different from random; **claim**: \"Observed task-specific efficacy lies above the 95th percentile of the random-16-head null in {n_above}/8 tasks.\"",
        f"- If `n_above` is mixed: per-task efficacy is real for some tasks and at chance for others. The pooled SEC ablation claim (which doesn't depend on this null) is unaffected.",
        "",
        "## Provenance",
        "",
        f"- Per-task baseline taken from " + (
            f"`{args.baseline_accuracy_json}`" if args.baseline_accuracy_json
            else "an unablated pass run at the start of this script"
        ),
        f"- Observed efficacy taken from " + (
            f"`{args.transfer_matrix}`" if args.transfer_matrix
            else "(none — observed not computed)"
        ),
        f"- Total runtime: {total_elapsed:.1f}s",
        f"- See `null_samples.json` for the per-sample detail.",
    ])
    with open(os.path.join(args.output_dir, "REPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    print(f"Wrote {os.path.join(args.output_dir, 'null_samples.json')}")
    print(f"Wrote {os.path.join(args.output_dir, 'null_summary.csv')}")
    print(f"Wrote {os.path.join(args.output_dir, 'observed_vs_null.csv')}")
    print(f"Wrote {os.path.join(args.output_dir, 'per_task_null_vs_observed.png')}")
    print(f"Wrote {os.path.join(args.output_dir, 'REPORT.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
