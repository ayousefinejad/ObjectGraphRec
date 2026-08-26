#!/usr/bin/env python3
"""F19 -- MIT-Indoors vs NYU-Depth as object-graph corpora: they differ on every input axis and
on none of the output ones.

    ~/hamedenv/bin/python plot_corpus.py

Reads the scene corpora through the encoder's own loader, each variant's provenance.json, and the
tuned-LightGCN runs (3 seeds per corpus). The size-matched arm -- MIT subsampled to NYU's exact
scene count -- is what makes the comparison interpretable: without it, MIT's advantage is
confounded with MIT simply being 4.5x larger.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
if str(OG) not in sys.path:
    sys.path.insert(0, str(OG))
from ObjectGraph.graph_data import build_cooccurrence, load_scenes  # noqa: E402

OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
EMB = OG / "objectgraph-eval" / "embeddings"
RUNS = OG / "data" / "lattice-runs"

INK, MUTED, GRID, SURFACE = "#17160f", "#6b6a63", "#e2e0d8", "#ffffff"
MITC, NYUC, SUBC, UNIC, OFFC = "#2a78d6", "#eb6834", "#9dc2ec", "#4a3aa7", "#d8d5cb"
# Must anchor on `test==`. main.py prints `val==` every --verbose epochs but `test==` only when
# validation improves, so a log almost always ENDS on a val line -- a bare `recall=[...]` regex
# taking the last match silently reports the final validation score as if it were the test score.
TEST = re.compile(r"test==\[.*?\], recall=\[[\d.]+, ([\d.]+)\]")

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

CORPORA = [("mit", "MIT", "openai_mit.json", "openai_mit", MITC),
           ("sub", "MIT@579", "mit_sub579.json", "mit_sub579", SUBC),
           ("nyu", "NYU", "nyu-depth.json", "nyu_default_fixed", NYUC)]


def corpus_stats():
    out = {}
    for key, label, fn, var, col in CORPORA:
        sc = load_scenes({"scenes_path": str(OG / "data" / fn)})
        nodes, pairs, _ = build_cooccurrence(sc)
        prov = json.loads((RUNS / var / "provenance.json").read_text())
        out[key] = {"label": label, "color": col, "scenes": len(sc), "nodes": len(nodes),
                    "edges": len(pairs), "auc": prov["intrinsic"]["test_auc"],
                    "vocab": {n.lower() for n in nodes}}
    return out


def r20(pattern):
    v = []
    for s in (0, 1, 2):
        f = EMB / f"{pattern}_seed{s}.log"
        if f.exists():
            m = TEST.findall(f.read_text())
            if m:
                v.append(float(m[-1]))
    return np.array(v)


def r20_cell(cell):
    v = [float(r["recall@20"]) for r in csv.DictReader((RUNS / "tuning.csv").open())
         if r["cell"] == cell]
    return np.array(v)


def main():
    st = corpus_stats()
    fig = plt.figure(figsize=(13.4, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.5], wspace=0.34,
                          left=0.05, right=0.985, top=0.72, bottom=0.14)

    # ---------------------------------------------------------------- (a) scale
    ax = fig.add_subplot(gs[0, 0])
    metrics = [("scenes", "scenes"), ("nodes", "graph nodes"), ("edges", "co-occurrence edges")]
    x = np.arange(len(metrics)); w = 0.26
    for i, (key, _, _, _, col) in enumerate(CORPORA):
        vals = [st[key][m] for m, _ in metrics]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=st[key]["label"], edgecolor=SURFACE)
    ax.set_yscale("log")
    ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=8)
    ax.set_ylabel("count (log scale)")
    ax.set_title("(a) The corpora differ in scale", loc="left", fontweight="bold")
    ax.legend(loc="upper left", ncols=3, fontsize=7.5, columnspacing=0.9, handletextpad=0.4)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    # ---------------------------------------------------------------- (b) vocabulary
    ax = fig.add_subplot(gs[0, 1])
    m, n = st["mit"]["vocab"], st["nyu"]["vocab"]
    parts = [(len(m - n), MITC, "MIT only"), (len(m & n), "#7a9bbf", "shared"),
             (len(n - m), NYUC, "NYU only")]
    bottom = 0
    for v, c, lab in parts:
        ax.bar(0, v, 0.5, bottom=bottom, color=c, label=f"{lab} ({v})", edgecolor=SURFACE)
        ax.text(0, bottom + v / 2, str(v), ha="center", va="center", fontsize=9,
                color="white" if lab != "shared" else INK, fontweight="bold")
        bottom += v
    ax.set_xticks([]); ax.set_ylabel("object node names")
    ax.set_title("(b) …and in vocabulary,\nbut NYU is mostly inside MIT", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    ax.annotate(f"{100 * len(m & n) / len(n):.0f}% of NYU's\nvocabulary is in MIT",
                xy=(0.30, bottom * 0.55), fontsize=8, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    # ---------------------------------------------------------------- (c) downstream
    ax = fig.add_subplot(gs[0, 2])
    arms = [("no object\n(image+text)", r20_cell("lgn3"), OFFC, None),
            ("NYU", r20("corp_nyu_default_fixed"), NYUC, st["nyu"]["auc"]),
            ("MIT@579\n(size-matched)", r20("corp_mit_sub579"), SUBC, st["sub"]["auc"]),
            ("MIT+NYU", r20_cell("obj_lgn3"), UNIC, None),
            ("MIT", r20("corp_openai_mit"), MITC, st["mit"]["auc"])]
    for i, (lab, v, col, auc) in enumerate(arms):
        if not len(v):
            continue
        ax.bar(i, v.mean(), 0.6, color=col, edgecolor=SURFACE,
               yerr=v.std(ddof=1), capsize=4, error_kw=dict(ecolor=MUTED, lw=1.1))
        ax.scatter([i] * len(v), v, s=18, color=INK, zorder=3, linewidths=0)
        if auc:
            ax.annotate(f"AUC {auc:.3f}", xy=(i, v.mean() + v.std(ddof=1)), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    base = r20_cell("lgn3").mean()
    ax.axhline(base, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(4.45, base, " no-object\n baseline", va="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms], fontsize=7.8)
    ax.set_ylabel("test Recall@20")
    ax.set_ylim(0.0428, 0.0450)
    ax.set_title("(c) …and not at all downstream. Tuned LightGCN, 3 seeds; "
                 "no pair is resolvable", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    fig.suptitle("MIT-Indoors vs NYU-Depth as object-graph corpora",
                 x=0.05, ha="left", fontsize=13, fontweight="bold", y=0.965)
    mit_v, nyu_v = r20("corp_openai_mit"), r20("corp_nyu_default_fixed")
    sub_v = r20("corp_mit_sub579")
    fig.text(0.05, 0.885,
             "MIT is 4.5x NYU's size and holds 83% of its vocabulary, yet every corpus lands "
             f"within {1000 * (max(a[1].mean() for a in arms if len(a[1])) - min(a[1].mean() for a in arms if len(a[1]))):.2f}e-3 "
             "of the others — inside this backbone's ~0.0017 resolution.\n"
             f"Raw MIT − NYU is +{1000 * (mit_v.mean() - nyu_v.mean()):.2f}e-3 (t=+1.44, n.s.); "
             f"size-matched it falls to +{1000 * (sub_v.mean() - nyu_v.mean()):.2f}e-3 (t=+0.62). "
             "What little advantage MIT has is corpus size, not corpus quality — which is the "
             "arm the size-matched\ncontrol exists to isolate. NYU also has the highest intrinsic "
             "encoder AUC and the lowest downstream score, the fourth instance in this study of "
             "intrinsic quality failing to predict recommendation.",
             fontsize=8, color=MUTED, ha="left", va="top", linespacing=1.6)

    for fmt in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"F19_corpus_comparison.{fmt}", bbox_inches="tight", facecolor=SURFACE,
                    **({"dpi": 200} if fmt == "png" else {}))
    plt.close(fig)
    print(f"-> {OUT}/F19_corpus_comparison.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
