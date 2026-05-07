#!/usr/bin/env python3
"""Fig 2 v3: side-by-side panels, sized for figure* (both columns).

Two-panel scatter showing the K-FE asymmetry between target sensitivity
(left) and source efficacy (right) across model pairs. Each axis is the
K-residualized (task, K) flattened vector for a model. Writes:

    paper/figures/main/fig2_kfe_asymmetry.pdf
    paper/figures/main/fig2_kfe_asymmetry.png

Run from the paper/ working directory:

    cd paper
    python scripts/figures/make_fig2_kfe_asymmetry.py
"""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats

models = {
    'Llama':   'results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json',
    'Qwen':    'results/qwen_2_5_7B_instruct/transfer/cross_task_transfer_matrix.json',
    'Mistral': 'results/mistral_7B_instruct/transfer/cross_task_transfer_matrix.json',
}

def build_matrices(path):
    d = json.load(open(path))
    sources, targets = d['sources'], d['targets']
    Ks = ['8','16','32','48','64','96','128']
    n_t, n_K = len(targets), len(Ks)
    drop = np.zeros((len(sources), n_t, n_K))
    for i,s in enumerate(sources):
        for j,t in enumerate(targets):
            for k,K in enumerate(Ks):
                drop[i,j,k] = d['results'][s][t]['by_k'][K]['drop_from_k0']
    sens = drop.mean(axis=0)
    eff = np.zeros((len(sources), n_K))
    for i in range(len(sources)):
        mask = np.ones(n_t, dtype=bool); mask[i] = False
        eff[i] = drop[i, mask, :].mean(axis=0)
    sens_r = sens - sens.mean(axis=0, keepdims=True)
    eff_r  = eff  - eff.mean(axis=0, keepdims=True)
    return targets, Ks, sens_r, eff_r

mats = {m: build_matrices(p) for m,p in models.items()}
pairs = [('Llama','Qwen'), ('Llama','Mistral'), ('Qwen','Mistral')]

# Wider, shorter — fits as figure* without dominating column height
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax_i, (title, getter) in enumerate([
    ('Target sensitivity (which tasks collapse)', lambda m: mats[m][2]),
    ('Source efficacy (which heads disrupt)',     lambda m: mats[m][3]),
]):
    ax = axes[ax_i]
    for ma, mb in pairs:
        xa = getter(ma).flatten(); xb = getter(mb).flatten()
        r,_ = stats.pearsonr(xa, xb)
        ax.scatter(xa, xb, s=12, alpha=0.65, label=f'{ma}\u2194{mb}: $R^2$={r**2:.2f}')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.4)
    mn, mx = -0.5, 0.5
    ax.plot([mn,mx],[mn,mx], '--', color='gray', alpha=0.35, lw=0.7)
    ax.set_xlim(mn,mx); ax.set_ylim(mn,mx)
    ax.set_xlabel('model A (K-residualized)', fontsize=8.5)
    ax.set_ylabel('model B (K-residualized)', fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.legend(loc='upper left', fontsize=7, frameon=True, framealpha=0.85, handletextpad=0.3)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.18, linestyle=':')

plt.tight_layout()
out_pdf = Path("figures/main/fig2_kfe_asymmetry.pdf")
out_png = Path("figures/main/fig2_kfe_asymmetry.png")
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=180, bbox_inches='tight')
print(f"Wrote {out_pdf}, {out_png}")
