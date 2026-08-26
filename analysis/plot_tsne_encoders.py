#!/usr/bin/env python3
"""t-SNE of the GraphSAGE vs GAT object-graph encoders, from tsne_encoder_sage_vs_gat.npz.

    ~/hamedenv/bin/python plot_tsne_encoders.py

Both encoders are the `converged` recipe (1000 epochs, tau=0.5, lr=1e-3, 2 layers, same
corpus, same seed) and differ ONLY in backbone, so any difference in cluster structure is
attributable to the encoder. t-SNE is fit separately per panel -- the two are different
embedding spaces, so a shared fit would not be meaningful (same convention as the
FREEDOM-vs-CRANE reference figure).

Colour = weak room label (diagnostic only -- derived from object identity via anchor words,
so clustering by room is partly true by construction; not a claim that the encoder "discovered"
rooms). Size = log(scene co-occurrence frequency).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Same validated categorical palette used throughout this study's figures (make_figures.py) --
# colorblind-checked, light surface. Kept identical so this figure sits consistently alongside
# the others if both go in the same paper.
INK, MUTED, GRID, SURFACE = "#17160f", "#6b6a63", "#e2e0d8", "#ffffff"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
UNLABELLED = "#b9b7ae"  # deliberately outside the categorical set: not a real class

ROOMS = ["bathroom", "bedroom", "kitchen", "livingroom", "dining_room"]
ROOM_LABEL = {"bathroom": "Bathroom", "bedroom": "Bedroom", "kitchen": "Kitchen",
              "livingroom": "Living room", "dining_room": "Dining room",
              "unlabelled": "Unlabelled (diagnostic gap)"}
ROOM_COLOR = dict(zip(ROOMS, S)) | {"unlabelled": UNLABELLED}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "legend.frameon": False, "figure.dpi": 150,
})


def fit(emb, seed=0):
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed,
               learning_rate="auto").fit_transform(emb)


def main():
    d = np.load(HERE / "tsne_encoder_sage_vs_gat.npz", allow_pickle=True)
    room = d["room"]
    freq = d["scene_freq"].astype(float)
    # log-scaled marker area: raw co-occurrence counts span orders of magnitude (singleton
    # objects to hub objects like 'Chair'), and a linear size map would make the long tail
    # invisible and the hub a single dominating blob.
    size = 10 + 55 * (np.log1p(freq) / np.log1p(freq).max())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), gridspec_kw={"wspace": 0.06})
    panels = [("emb_sage", "GraphSAGE", "test AUC 0.789"), ("emb_gat", "GAT", "test AUC 0.810")]

    for ax, (key, title, auc) in zip(axes, panels):
        xy = fit(d[key])
        # Draw the diagnostic-gap class first and underneath, so it reads as background
        # context rather than competing with the five real room clusters for attention.
        order = ["unlabelled"] + ROOMS
        for r in order:
            m = room == r
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=size[m], c=ROOM_COLOR[r],
                      alpha=0.55 if r == "unlabelled" else 0.85,
                      linewidths=0.4, edgecolors=SURFACE, zorder=1 if r == "unlabelled" else 2)
        # AUC sits just above the axes edge; the bold title needs enough pad to clear it
        # rather than share the same baseline (they collided at the same y before this).
        ax.annotate(auc, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 4),
                   textcoords="offset points", va="bottom", fontsize=9.5, color=MUTED)
        ax.set_title(f"{title}", loc="left", fontsize=13, fontweight="bold", pad=24)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                      markerfacecolor=ROOM_COLOR[r], markeredgecolor=SURFACE,
                      label=ROOM_LABEL[r]) for r in ROOMS + ["unlabelled"]]
    fig.legend(handles=handles, loc="lower center", ncols=6, bbox_to_anchor=(0.5, -0.02),
              fontsize=9.5, handletextpad=0.4, columnspacing=1.3)

    fig.suptitle("t-SNE of the object co-occurrence graph encoder — GraphSAGE vs GAT",
                x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.03)
    fig.text(0.01, 0.985,
             "1,068 object nodes, identical recipe (1000 ep, τ=0.5, lr=1e-3, 2 layers) and "
             "corpus — backbone is the only difference. Room label is a weak diagnostic "
             "(anchor-word rule), not ground truth.",
             fontsize=8.8, color=MUTED, ha="left")

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"F15_tsne_sage_vs_gat.{fmt}", bbox_inches="tight",
                   facecolor=SURFACE, **({"dpi": 200} if fmt == "png" else {}))
    plt.close(fig)
    print(f"-> {OUT}/F15_tsne_sage_vs_gat.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
