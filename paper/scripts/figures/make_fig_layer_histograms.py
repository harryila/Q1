#!/usr/bin/env python3
"""1x3 layer-occupancy histograms for the combined top-16 detected heads,
one panel per model (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B).

The x-axis label "Layer index" is positioned with proportional spacing to
the y-axis label "Number of top-16 heads" --- i.e. close to the axis rather
than far below it. Matplotlib's default ``supxlabel`` places the figure-wide
label flush with the bottom margin; we override with an explicit y position.

Writes:
    paper/figures/main/fig_layer_histograms.pdf
    paper/figures/main/fig_layer_histograms.png

Run from paper/:
    python scripts/figures/make_fig_layer_histograms.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = [
    ("Llama-3.1-8B", "detection/llama_3_1_8B_instruct", 32),
    ("Qwen2.5-7B",   "detection/qwen_2_5_7B_instruct",  28),
    ("Mistral-7B",   "detection/mistral_7B_instruct",   32),
]


def load_top16_layers(model_dir: str) -> list[int]:
    """Return list of layer indices for the model's combined top-16 heads.

    Uses ``topk/long_context_combined_top16.json`` if present; falls back to
    slicing ``[:16]`` from the full combined ranking.
    """
    base = Path(model_dir)
    top16_path = base / "topk" / "long_context_combined_top16.json"
    if top16_path.exists():
        ranked = json.load(open(top16_path))
    else:
        full = json.load(open(base / "long_context_combined_heads.json"))
        ranked = full[:16]
    return [int(key.split("-", 1)[0]) for key, _score in ranked]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), sharey=True)

    for ax, (label, model_dir, n_layers) in zip(axes, MODELS):
        layers = load_top16_layers(model_dir)
        counts = Counter(layers)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        ax.bar(xs, ys, width=0.7, color="#a8c8e8", edgecolor="#3a6e9b", linewidth=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-0.6, n_layers - 0.4)
        ax.set_xticks(list(range(0, n_layers, 4)))
        ax.tick_params(axis="both", labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Number of top-16 heads", fontsize=10)

    # Lay out subplots first, leaving room at the bottom for the shared x-label.
    # Bottom margin sized to match the left margin reserved for the y-label,
    # so the two labels sit at proportional distances from their axes.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.20, wspace=0.10)

    # Place the shared x-axis label inside the reserved bottom band, close to
    # the tick labels rather than at the figure edge. y=0.06 puts it about as
    # far below the ticks as the y-label sits to the left of its ticks.
    fig.text(0.5, 0.06, "Layer index", ha="center", va="center", fontsize=10)

    out_dir = Path("figures/main")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig_layer_histograms.pdf"
    out_png = out_dir / "fig_layer_histograms.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_pdf}, {out_png}")

    print()
    print("Layer occupancy of combined top-16 per model:")
    for label, model_dir, _ in MODELS:
        layers = load_top16_layers(model_dir)
        c = Counter(layers)
        line = ", ".join(f"L{k}:{c[k]}" for k in sorted(c))
        print(f"  {label:<14} {line}")


if __name__ == "__main__":
    main()
