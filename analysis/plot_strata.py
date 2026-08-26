#!/usr/bin/env python3
"""F18 -- where the object modality's gain actually lands, and whether it survives a placebo.

    ~/hamedenv/bin/python plot_strata.py

Reads strata_summary.csv and strata_global.json (written by make_strata_data.py).

(a) is the mechanism claim: if the object graph helps because its vocabulary covers the
catalogue, the gain must concentrate on the items that vocabulary reaches.
(b) is the confound: covered labels are the catalogue's head, and popular items are easier to
retrieve, so the same comparison is repeated inside each popularity decile.
(c) is the placebo: identical features, identical graph up to isomorphism, only the
item-to-object correspondence permuted. A gain that survives that is not about objects.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID, SURFACE = "#17160f", "#6b6a63", "#e2e0d8", "#ffffff"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GOOD, BAD = "#1baf7a", "#e34948"
TIERS = ("exact", "near", "far")
TIER_COL = {"exact": S[0], "near": "#9dc2ec", "far": "#d8d5cb", "all": MUTED}

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


def load():
    rows = {}
    with (HERE / "strata_summary.csv").open() as f:
        for r in csv.DictReader(f):
            rows[r["stratum"]] = r
    seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in next(iter(rows.values()))
                    if k.startswith("hit20_control_s")})
    return rows, seeds


def deltas(row, seeds, a="control", b="noobj"):
    """Per-seed paired difference in per-test-item hit@20."""
    return np.array([float(row[f"hit20_{a}_s{s}"]) - float(row[f"hit20_{b}_s{s}"]) for s in seeds])


def main():
    rows, seeds = load()
    glob = json.loads((HERE / "strata_global.json").read_text())

    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 0.95], wspace=0.32,
                          left=0.055, right=0.985, top=0.72, bottom=0.13)

    # ------------------------------------------------------------------ (a) by tier
    ax = fig.add_subplot(gs[0, 0])
    cats = ["all"] + list(TIERS)
    x = np.arange(len(cats))
    for i, c in enumerate(cats):
        d = deltas(rows[c], seeds)
        col = GOOD if d.mean() > 0 else BAD
        ax.bar(i, 100 * d.mean(), 0.6, color=col, alpha=0.85 if c != "all" else 0.45,
               edgecolor=SURFACE)
        ax.scatter([i] * len(d), 100 * d, s=22, color=INK, zorder=3, linewidths=0)
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.set_xticks(x, [f"{c}\nn={int(rows[c]['exposures']):,}" for c in cats])
    ax.set_ylabel("Δ per-item hit@20  (pp)")
    ax.set_title("(a) The gain lands on the items\nthe vocabulary reaches", loc="left",
                 fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------ (b) popularity control
    ax = fig.add_subplot(gs[0, 1])
    w = 0.38
    for j, tier in enumerate(("exact", "far")):
        xs, ys = [], []
        for d in range(10):
            key = f"{tier}|decile{d}"
            if key not in rows:
                continue
            xs.append(d + (j - 0.5) * w)
            ys.append(100 * deltas(rows[key], seeds).mean())
        ax.bar(xs, ys, w, color=TIER_COL[tier], edgecolor=MUTED, linewidth=0.5,
               label=f"{tier} tier")
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.set_xticks(range(10), [f"D{d}" for d in range(10)])
    ax.set_xlabel("training-interaction decile  (D0 = least popular)")
    ax.set_ylabel("Δ per-item hit@20  (pp)")
    ax.set_title("(b) Same comparison inside each popularity decile,\n"
                 "since covered items are also the popular ones",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # ------------------------------------------------------------------ (c) placebo
    ax = fig.add_subplot(gs[0, 2])
    arms = [("noobj", "image+text", "#d8d5cb"), ("shufobj", "+ shuffled\nobject", S[3]),
            ("control", "+ object", S[0])]
    for i, (arm, lab, col) in enumerate(arms):
        vals = [g["recall@20"] for g in glob if g["arm"] == arm]
        if not vals:
            continue
        ax.bar(i, np.mean(vals), 0.6, color=col, edgecolor=SURFACE)
        ax.scatter([i] * len(vals), vals, s=22, color=INK, zorder=3, linewidths=0)
        ax.annotate(f"{np.mean(vals):.5f}", xy=(i, max(vals)), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(arms)), [a[1] for a in arms])
    ax.set_ylabel("test Recall@20")
    lo = min(g["recall@20"] for g in glob)
    hi = max(g["recall@20"] for g in glob)
    ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_title("(c) Placebo: permuting which item\ngets which object vector",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("Does the object modality help *because* its vocabulary covers the catalogue?",
                 x=0.055, ha="left", fontsize=13, fontweight="bold", y=0.96)
    fig.text(0.055, 0.895,
             f"LATTICE on the shipped default_fixed variant, seeds {seeds}, paired by seed. "
             "Tiers are F17's: exact = the item's label is a node of the object graph, far = no "
             "node within cosine 0.6.\nMetric is per-test-item hit@20 (of all (user, test item) "
             "pairs in the stratum, the share where the item made that user's top-20); the "
             "offline ranking reproduces every run's logged Recall@20 to <3e-5.",
             fontsize=8, color=MUTED, ha="left", va="top", linespacing=1.6)

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"F18_coverage_strata.{fmt}", bbox_inches="tight", facecolor=SURFACE,
                    **({"dpi": 200} if fmt == "png" else {}))
    plt.close(fig)
    print(f"-> {OUT}/F18_coverage_strata.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
