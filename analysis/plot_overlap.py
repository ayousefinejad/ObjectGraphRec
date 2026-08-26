#!/usr/bin/env python3
"""F17 -- how far the MIT / NYU-Depth detected-object vocabulary reaches into the Amazon
Home & Kitchen catalogue, from overlap_mit_nyu_amazon.npz.

    ~/hamedenv/bin/python plot_overlap.py

Panel (a) is the headline: 51.8% of the 14,503 items carry a label that is *verbatim* a node of
the trained object graph. (b) shows that share is not scattered across the catalogue's tail --
the labels carrying the most items are almost all exact hits. (c) is the limitation, in the same
figure rather than in a footnote: the labels the vocabulary genuinely misses, with the proxy node
each one is silently assigned instead.

Palette, rcParams and the svg.fonttype rule are the study's, copied from make_figures.py so every
figure in this directory renders identically.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keeps SVG text as <text>: ~4x smaller, fits Notion
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID, SURFACE = "#17160f", "#6b6a63", "#e2e0d8", "#ffffff"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
EXACT, NEAR, FAR = S[0], "#9dc2ec", "#d8d5cb"     # one hue family: exact -> near -> unreached
TAU = 0.6

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 150,
})


def tiers(d, corpus, tau=TAU):
    ex = d[f"{corpus}_exact_idx"] >= 0
    cos = d[f"{corpus}_near_cos"]
    return ex, (~ex) & (cos >= tau), (~ex) & (cos < tau)


def main():
    d = np.load(HERE / "overlap_mit_nyu_amazon.npz", allow_pickle=False)
    n = d["label_n_items"]
    total = int(n.sum())
    labels = d["labels_uniq"]
    nodes = d["union_nodes"]
    ex, near, far = tiers(d, "union")

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.42,
                          left=0.05, right=0.985, top=0.755, bottom=0.085)

    # ---------------------------------------------------------------- (a) coverage
    ax = fig.add_subplot(gs[0, 0])
    corpora = [("mit", "MIT Indoor"), ("nyu", "NYU-Depth"), ("union", "MIT + NYU\n(shipped)")]
    x = np.arange(len(corpora))
    parts = []
    for key, _ in corpora:
        e, nr, fr = tiers(d, key)
        parts.append([n[e].sum(), n[nr].sum(), n[fr].sum()])
    parts = np.array(parts, dtype=float)
    bottom = np.zeros(len(corpora))
    for j, (col, name) in enumerate([(EXACT, "exact node name"),
                                     (NEAR, f"near (cos $\\geq$ {TAU})"),
                                     (FAR, f"unreached (cos < {TAU})")]):
        ax.bar(x, parts[:, j], 0.62, bottom=bottom, color=col, label=name,
               edgecolor=SURFACE, linewidth=0.8)
        bottom += parts[:, j]
    for i in range(len(corpora)):
        ax.text(i, parts[i, 0] / 2, f"{100 * parts[i, 0] / total:.1f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    ax.set_xticks(x, [c[1] for c in corpora])
    ax.set_ylabel("Amazon items (of 14,503)")
    ax.set_title("(a) Catalogue reached by each\nscene corpus", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # ---------------------------------------------------------------- (b) covered head
    ax = fig.add_subplot(gs[0, 1])
    top = np.argsort(-n)[:22][::-1]
    cols = [EXACT if ex[i] else (NEAR if near[i] else FAR) for i in top]
    ax.barh(np.arange(len(top)), n[top], color=cols, height=0.72)
    ax.set_yticks(np.arange(len(top)), [labels[i] for i in top])
    ax.tick_params(axis="y", labelsize=7.5)
    for k, i in enumerate(top):
        if not ex[i]:
            ax.text(n[i] + 4, k, f"→ {nodes[d['union_match_idx'][i]]}", va="center",
                    fontsize=6.8, color=MUTED)
    ax.set_xlabel("items carrying this label")
    ax.set_title("(b) The catalogue's 22 largest labels\nare almost all exact object nodes",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, n[top].max() * 1.30)

    # ---------------------------------------------------------------- (c) the miss
    ax = fig.add_subplot(gs[0, 2])
    far_idx = np.where(far)[0]
    far_top = far_idx[np.argsort(-n[far_idx])][:14][::-1]
    ax.barh(np.arange(len(far_top)), n[far_top], color=FAR, height=0.72,
            edgecolor=MUTED, linewidth=0.5)
    ax.set_yticks(np.arange(len(far_top)), [labels[i] for i in far_top])
    ax.tick_params(axis="y", labelsize=7.5)
    for k, i in enumerate(far_top):
        ax.text(n[i] + 2, k, f"→ {nodes[d['union_match_idx'][i]]} ({d['union_near_cos'][i]:.2f})",
                va="center", fontsize=6.8, color=MUTED)
    ax.set_xlabel("items carrying this label")
    ax.set_title(f"(c) Labels the vocabulary misses, and\nthe proxy node each is assigned instead",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, n[far_top].max() * 1.75)

    fig.suptitle("Detected indoor objects (MIT + NYU-Depth) vs the Amazon Home & Kitchen catalogue",
                 x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.975)
    e, nr, fr = parts[2]
    fig.text(0.05, 0.925,
             f"Objects: the {len(nodes):,} nodes of the trained co-occurrence graph, verified "
             f"equal to the default_fixed encoder checkpoint. Items: raw_graph.txt, one label per "
             f"item, 1,815 distinct — the same file build_object_feat reads.\n"
             f"Tiers are that function's own branch: exact name match, else nearest MiniLM node. "
             f"The shipped corpus reaches {int(e):,} items exactly ({100 * e / total:.1f}%) and "
             f"{int(nr):,} more by proximity ({100 * nr / total:.1f}%), leaving "
             f"{int(fr):,} ({100 * fr / total:.1f}%) unreached.\n"
             f"τ sets only the near/unreached split, never the exact share: 3.9% / 10.4% / 21.1% "
             f"unreached at τ = 0.5 / 0.6 / 0.7.",
             fontsize=8, color=MUTED, ha="left", va="top", linespacing=1.6)

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"F17_vocabulary_overlap.{fmt}", bbox_inches="tight",
                    facecolor=SURFACE, **({"dpi": 200} if fmt == "png" else {}))
    plt.close(fig)
    print(f"-> {OUT}/F17_vocabulary_overlap.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
