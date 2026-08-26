#!/usr/bin/env python3
"""t-SNE of LATTICE user (star) and item (dot) embeddings, dense vs GAT item-graph propagation,
from tsne_lattice_users_dense_vs_gat.npz. Reproduces the FREEDOM-vs-CRANE reference figure's
layout: one star per sampled user, its interacted items as same-coloured dots with a connecting
line, t-SNE fit jointly on user+item embeddings per panel (so stars and dots share one space).

    ~/hamedenv/bin/python plot_tsne_recommender.py

Both panels are the SAME LATTICE run (openai_mit, seed 0) -- same users, same items, same
encoder -- differing only in item_prop (dense vs gat), so any difference in how tightly a star
sits among its items is attributable to the propagation operator alone.

10 users need 10 distinguishable identities. The validated 8-colour categorical palette used
elsewhere in this study is deliberately not stretched to 10 (the method's own rule: past 8,
fold to "Other" or facet) -- but folding isn't an option when the figure's entire point is
naming 10 individual users, matching the reference figure's own convention. Matplotlib's tab10
is used instead, with user ID as a direct label on every star as a non-colour identity backup.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID, SURFACE = "#17160f", "#6b6a63", "#e2e0d8", "#ffffff"
USER_COLORS = plt.get_cmap("tab10").colors  # 10 users -- see module docstring

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "legend.frameon": False, "figure.dpi": 150,
})


def joint_tsne(ua, ia, seed=0):
    """t-SNE fit on [users; items] together, so both live in one shared 2D space."""
    x = np.concatenate([ua, ia], axis=0)
    perp = max(5, min(15, (x.shape[0] - 1) // 3))
    xy = TSNE(n_components=2, init="pca", perplexity=perp, random_state=seed,
             learning_rate="auto").fit_transform(x)
    return xy[:ua.shape[0]], xy[ua.shape[0]:]


def main():
    d = np.load(HERE / "tsne_lattice_users_dense_vs_gat.npz", allow_pickle=True)
    import json
    meta = json.loads(str(d["meta_json"]))
    user_ids = d["user_ids"]
    eu, ei = d["edge_user_idx"], d["edge_item_idx"]
    n_users = len(user_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), gridspec_kw={"wspace": 0.05})
    panels = [("ua_dense", "ia_dense", "Dense propagation (published)", meta["dense_r20"]),
              ("ua_gat", "ia_gat", "GAT propagation (learned attention)", meta["gat_r20"])]

    for ax, (uk, ik, title, r20) in zip(axes, panels):
        uxy, ixy = joint_tsne(d[uk], d[ik])

        # Lines first (background), then item dots, then user stars on top -- so a star is
        # never hidden behind its own spokes, matching the reference figure's layering.
        for k in range(n_users):
            col = USER_COLORS[k % 10]
            mask = eu == k
            for j in np.where(mask)[0]:
                ip = ixy[ei[j]]
                ax.plot([uxy[k, 0], ip[0]], [uxy[k, 1], ip[1]],
                       color=col, linewidth=0.9, alpha=0.55, zorder=1)
        for k in range(n_users):
            col = USER_COLORS[k % 10]
            mask = eu == k
            ax.scatter(ixy[ei[mask], 0], ixy[ei[mask], 1], s=48, color=col,
                      edgecolors=SURFACE, linewidths=0.6, zorder=2)
        for k in range(n_users):
            col = USER_COLORS[k % 10]
            ax.scatter([uxy[k, 0]], [uxy[k, 1]], marker="*", s=340, color=col,
                      edgecolors=INK, linewidths=0.9, zorder=3)

        ax.annotate(f"test R@20 {r20:.4f}", xy=(0, 1.0), xycoords="axes fraction",
                   xytext=(0, 4), textcoords="offset points", va="bottom",
                   fontsize=9.5, color=MUTED)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=24)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Line2D([0], [0], marker="*", linestyle="", markersize=13,
                          markerfacecolor=USER_COLORS[k % 10], markeredgecolor=INK,
                          label=f"User {uid}") for k, uid in enumerate(user_ids)]
    fig.legend(handles=handles, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.06),
              fontsize=9.5, handletextpad=0.4, columnspacing=1.4)

    fig.suptitle("t-SNE of user (★) and item embeddings — LATTICE item-graph propagation",
                x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.03)
    fig.text(0.01, 0.985,
             f"Same recommender, same {n_users} sampled users, same {len(d['item_ids'])} "
             "interacted items, same encoder — only the post-fusion item-graph propagation "
             "operator differs (openai_mit, seed 0). t-SNE fit jointly per panel on "
             "[user; item] embeddings.",
             fontsize=8.8, color=MUTED, ha="left")

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"F16_tsne_lattice_dense_vs_gat.{fmt}", bbox_inches="tight",
                   facecolor=SURFACE, **({"dpi": 200} if fmt == "png" else {}))
    plt.close(fig)
    print(f"-> {OUT}/F16_tsne_lattice_dense_vs_gat.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
