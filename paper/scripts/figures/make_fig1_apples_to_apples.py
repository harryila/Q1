#!/usr/bin/env python3
"""Apples-to-apples Fig 1: drop curves on the n=192 SEC subset, all four methods.

Restricts LME / NQ / Random results to the same 192 instance indices that the
QRScore-SEC run was evaluated on, so the four curves are comparable on a
common test set. Writes:

    paper/figures/main/fig1_apples_to_apples_llama.pdf
    paper/figures/main/fig1_apples_to_apples_llama.png

Run from the paper/ working directory:

    cd paper
    python scripts/figures/make_fig1_apples_to_apples.py
"""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("results/llama_3_1_8B_instruct/raw_results")
sec = json.load(open(ROOT/"QRScore-SEC_results.json"))
lme = json.load(open(ROOT/"QRScore-8B-LME-TRAIN_results.json"))
nq  = json.load(open(ROOT/"QRScore-8B-NQ-TRAIN_results.json"))
r42 = json.load(open(ROOT/"Random-seed42_results.json"))
r123= json.load(open(ROOT/"Random-seed123_results.json"))
r456= json.load(open(ROOT/"Random-seed456_results.json"))

sec_idxs = set(x['idx'] for x in sec['details']['0'])
Ks = ['0','8','16','32','48','64','96','128']

def restrict(d, idxs):
    out = {}
    for K in Ks:
        sub = [x for x in d['details'][K] if x['idx'] in idxs]
        out[K] = sum(x['correct'] for x in sub)/len(sub) if sub else float('nan')
    return out

sec_curve = {K: sec['accuracy_curve'][K] for K in Ks}
lme_curve = restrict(lme, sec_idxs)
nq_curve  = restrict(nq, sec_idxs)
r_curves  = [restrict(r, sec_idxs) for r in [r42, r123, r456]]
r_mean    = {K: float(np.mean([rc[K] for rc in r_curves])) for K in Ks}

x = [int(K) for K in Ks]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(x, [sec_curve[K] for K in Ks], 'o-', label='QRScore-SEC',     color='#d62728', lw=1.8)
ax.plot(x, [lme_curve[K] for K in Ks], 's-', label='QRScore-LME',     color='#ff7f0e', lw=1.5)
ax.plot(x, [nq_curve[K] for K in Ks],  '^-', label='QRScore-NQ',      color='#2ca02c', lw=1.5)
ax.plot(x, [r_mean[K] for K in Ks],    'x--',label='Random (3-seed mean)', color='#7f7f7f', lw=1.2)

ax.set_xlabel('Number of top-$K$ heads ablated')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.25, linestyle=':')
ax.legend(loc='upper right', fontsize=8.5, frameon=True)
plt.tight_layout()
out_pdf = Path("figures/main/fig1_apples_to_apples_llama.pdf")
out_png = Path("figures/main/fig1_apples_to_apples_llama.png")
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Wrote {out_pdf} and {out_png}")
print("Drop@K=16 numbers (n=192):")
print(f"  QRScore-SEC: {sec_curve['0']*100:.1f} -> {sec_curve['16']*100:.1f} (drop {(sec_curve['0']-sec_curve['16'])*100:.1f}pp)")
print(f"  QRScore-LME: {lme_curve['0']*100:.1f} -> {lme_curve['16']*100:.1f} (drop {(lme_curve['0']-lme_curve['16'])*100:.1f}pp)")
print(f"  QRScore-NQ:  {nq_curve['0']*100:.1f} -> {nq_curve['16']*100:.1f} (drop {(nq_curve['0']-nq_curve['16'])*100:.1f}pp)")
print(f"  Random-mean: {r_mean['0']*100:.1f} -> {r_mean['16']*100:.1f} (drop {(r_mean['0']-r_mean['16'])*100:.1f}pp)")
