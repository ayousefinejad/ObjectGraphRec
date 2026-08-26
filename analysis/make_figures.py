#!/usr/bin/env python3
"""Figures for the object-graph modality evaluations (E1-E18).

Reads only this directory's extracts -- og_runs.csv, og_curves.csv,
og_baselines.csv, og_dataset.json. Nothing here touches object-graph/.

    ~/hamedenv/bin/python make_figures.py [--only F09 F06] [--fmt png pdf svg]

Design notes
  * Palette is the validated categorical set (colorblind-checked, light surface):
    worst adjacent CVD dE 9.1, worst normal-vision dE 19.6. Do not re-order.
  * One measure per axis. Where two measures must be compared (overall AUC vs
    low-degree AUC) they get two panels or two bar groups, never twin y-axes.
  * Bars are a single hue when they encode magnitude of one series; categorical
    hues are used only where color carries identity.
  * The shipped configuration is always marked as the reference row, per the
    study's sensitivity-study framing -- not the best-scoring row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# keeps SVG glyphs as <text> rather than paths: ~4x smaller, and required to
# fit Notion's 200 KiB attachment cap
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- palette
INK = "#17160f"
MUTED = "#6b6a63"
GRID = "#e2e0d8"
SURFACE = "#ffffff"
ACCENT = "#2a78d6"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#eaf1fb", "#c5daf4", "#9dc2ec", "#6ba4e0", "#3f86d2", "#2a78d6", "#1f5da6", "#153f71"]
GOOD, BAD, WARN = "#1baf7a", "#e34948", "#c98500"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150,
})


def style(ax, ygrid=True):
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=1.0, linestyle="-")
    ax.tick_params(length=0)
    return ax


def save(fig, name, fmts):
    for f in fmts:
        fig.savefig(OUT / f"{name}.{f}", bbox_inches="tight",
                    facecolor=SURFACE, **({"dpi": 200} if f == "png" else {}))
    plt.close(fig)
    print(f"  {name}  ({', '.join(fmts)})")


# ---------------------------------------------------------------- data
def load():
    runs = list(csv.DictReader((HERE / "og_runs.csv").open()))
    for r in runs:
        for k, v in list(r.items()):
            if v == "":
                r[k] = None
    base = list(csv.DictReader((HERE / "og_baselines.csv").open()))
    ds = json.loads((HERE / "og_dataset.json").read_text())
    curves = defaultdict(list)
    for c in csv.DictReader((HERE / "og_curves.csv").open()):
        if c["val_auc"]:
            curves[c["run_key"]].append((int(c["epoch"]), float(c["val_auc"]),
                                         float(c["loss"]) if c["loss"] else None))
    for k in curves:
        curves[k].sort()
    return runs, base, ds, curves


F = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None


def agg(rows, key="test_auc"):
    """mean, std, n, raw values -- std is the seed-to-seed spread."""
    v = [F(r, key) for r in rows]
    v = [x for x in v if x is not None]
    if not v:
        return None
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v), v)


def sel(runs, **kw):
    out = runs
    for k, v in kw.items():
        vals = {str(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])}
        out = [r for r in out if r.get(k) in vals]
    return out


def seeddots(ax, x, vals, color=INK, jitter=0.055):
    """Per-seed points over a bar: the honest version of an error bar."""
    rng = np.random.default_rng(0)
    xs = x + rng.uniform(-jitter, jitter, len(vals))
    ax.scatter(xs, vals, s=11, color=color, zorder=5,
               edgecolor=SURFACE, linewidth=0.7, alpha=.85)


def barlabel(ax, x, y, txt, dy=0.004, color=INK, size=8):
    ax.text(x, y + dy, txt, ha="center", va="bottom", fontsize=size,
            color=color, fontweight="bold")


# ================================================================ FIGURES
def f01_baselines(runs, base, ds, curves, fmts):
    """E1+E2+E3 -- what the encoder must beat, and why the negative regime matters."""
    shipped = agg(sel(runs, sweep="stageF", tag="default"))
    tuned = agg(sel(runs, sweep="stageF", tag="tuned"))
    nice = {"adamic_adar": "Adamic–Adar", "common_neighbors": "Common neighbours",
            "preferential_attachment": "Preferential attach.",
            "raw_minilm_cosine": "Raw MiniLM cosine", "untrained_sage": "Untrained SAGE"}
    shipped_u = agg(sel(runs, sweep="stageF", tag="default"), "test_auc_uniform")
    tuned_u = agg(sel(runs, sweep="stageF", tag="tuned"), "test_auc_uniform")
    items = [(nice[b["method"]], float(b["auc"]), float(b["auc_uniform"]), False)
             for b in base]
    items += [("Encoder (shipped)", shipped[0], shipped_u[0], True),
              ("Encoder (tuned)", tuned[0], tuned_u[0], True)]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"wspace": .30})
    for ax, which, title in [
        (axes[0], 1, "Degree-matched negatives  (primary)"),
        (axes[1], 2, "Uniform negatives  (literature comparability)")]:
        # each panel is ranked by its OWN metric -- that is the point
        order = sorted(items, key=lambda t: t[which])
        ys = np.arange(len(order))
        ax.barh(ys, [o[which] for o in order], height=.62,
                color=[ACCENT if o[3] else "#b9c6d3" for o in order])
        for y, o in zip(ys, order):
            ax.text(o[which] + .008, y, f"{o[which]:.3f}", va="center", fontsize=8,
                    color=INK, fontweight="bold" if o[3] else "normal")
        ax.set_yticks(ys, [o[0] for o in order])
        for lbl, o in zip(ax.get_yticklabels(), order):
            if o[3]:
                lbl.set_color(ACCENT)
                lbl.set_fontweight("bold")
        ax.set_xlim(0.5, 1.05)
        ax.set_xlabel("test AUC")
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=1.0)
        ax.tick_params(length=0)
        # The compression claim is specifically about the three graph-structural
        # baselines -- MiniLM is a text method and does not ride the degree
        # signal, so pooling it in would misstate the effect.
        struct = [o[which] for o in order
                  if o[0] in ("Adamic–Adar", "Common neighbours", "Preferential attach.")]
        ax.text(0.0, -.16,
                f"the 3 structural baselines span {max(struct) - min(struct):.3f} here",
                transform=ax.transAxes, fontsize=7.8,
                color=GOOD if which == 1 else BAD, fontweight="bold")
    fig.suptitle("E1–E3  The negative-sampling regime decides the ranking, not just the scale",
                 x=.012, ha="left", fontsize=11.5, fontweight="bold", y=1.06)
    fig.text(.012, .99, "Under uniform negatives Adamic–Adar (0.956) outranks the tuned "
                        "encoder (0.862); under degree-matched negatives that reverses "
                        "(0.744 vs 0.824). Same models, opposite conclusion.",
             fontsize=8.2, color=MUTED, ha="left")
    save(fig, "F01_baselines_and_negatives", fmts)


def f02_edge_weighting(runs, base, ds, curves, fmts):
    """E4 -- the shipped accident beats the published Eq. (2)."""
    order = [("dedup", "Unweighted\n(binary)"), ("multiplicity", "Count-weighted\n(shipped)"),
             ("weighted", "c/√(c_a·c_b)\n(paper Eq. 2)")]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i, (mode, label) in enumerate(order):
        m, s, n, vals = agg(sel(runs, sweep="stageA", min_cooc="1", edge_mode=mode))
        col = ACCENT if mode == "multiplicity" else "#b9c6d3"
        ax.bar(i, m, width=.58, color=col)
        seeddots(ax, i, vals)
        barlabel(ax, i, m + s, f"{m:.3f}")
    ax.set_xticks(range(3), [o[1] for o in order])
    ax.set_ylim(0.75, 0.80)
    ax.set_ylabel("test AUC  (degree-matched)")
    style(ax)
    ax.set_title("E4  Edge weighting at δ=1 — shipped ‘accident’ beats the published formula\n"
                 "5 seeds; dots are individual seeds. Blue = shipped.",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F02_edge_weighting", fmts)


def f03_threshold(runs, base, ds, curves, fmts):
    """E5 -- delta. AUC across delta is NOT like-for-like; coverage is the finding."""
    thr = {t["min_cooc"]: t for t in ds["threshold"]}
    deltas = [1, 2, 3, 5, 10]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": .18})
    ax = axes[0]
    for j, mode in enumerate(["dedup", "multiplicity", "weighted"]):
        ms, ss = [], []
        for d in deltas:
            a = agg(sel(runs, sweep="stageA", edge_mode=mode, min_cooc=str(d)))
            ms.append(a[0]); ss.append(a[1])
        ax.errorbar(range(len(deltas)), ms, yerr=ss, marker="o", ms=5, lw=2,
                    capsize=0, color=S[j], label=mode, zorder=3)
    ax.set_ylabel("test AUC")
    ax.legend(loc="lower left", ncols=3)
    style(ax)
    ax.set_title("E5  Co-occurrence threshold δ", loc="left", fontweight="bold", pad=8)
    ax.text(0, 1.02, "⚠ each δ rebuilds the graph, so these AUCs are scored on DIFFERENT "
                     "test sets — not a like-for-like comparison",
            transform=ax.transAxes, fontsize=7.6, color=BAD, va="bottom")

    ax2 = axes[1]
    cov = [thr[d]["node_coverage_pct"] for d in deltas]
    edges = [thr[d]["edges"] for d in deltas]
    ax2.bar(range(len(deltas)), cov, width=.55, color=ACCENT)
    for i, (c, e) in enumerate(zip(cov, edges)):
        barlabel(ax2, i, c, f"{c:.0f}%", dy=1.2)
        ax2.text(i, 3, f"{e:,}\nedges", ha="center", fontsize=7, color=SURFACE,
                 fontweight="bold")
    ax2.set_ylabel("non-isolated node coverage (%)")
    ax2.set_xlabel("min_cooc  (δ)")
    ax2.set_xticks(range(len(deltas)), [str(d) for d in deltas])
    ax2.set_ylim(0, 112)
    style(ax2)
    ax2.set_title("δ=2 already isolates 53% of object labels — the long-tail coverage the "
                  "modality exists to provide", loc="left", fontsize=8.5, color=MUTED, pad=6)
    save(fig, "F03_cooccurrence_threshold", fmts)


def f04_ofat(runs, base, ds, curves, fmts):
    """E6 -- one-factor-at-a-time from the stageB baseline cell."""
    B = sel(runs, sweep="stageB")
    knobs = ["epochs", "lr", "temperature", "dropout", "neg_ratio", "neg_mode"]
    from collections import Counter
    modal = {k: Counter(r[k] for r in B).most_common(1)[0][0] for k in knobs}
    titles = {"epochs": "epochs", "lr": "learning rate", "temperature": "temperature τ",
              "dropout": "dropout", "neg_ratio": "negative ratio", "neg_mode": "negative mode"}
    logx = {"epochs", "lr", "temperature"}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2),
                             gridspec_kw={"hspace": .46, "wspace": .22})
    # ONE shared y-range across all six panels. Per-panel autoscaling would give
    # every knob an equally dramatic slope and destroy the actual finding --
    # that five of the six are flat.
    YLO, YHI = 0.744, 0.801
    for ax, k in zip(axes.ravel(), knobs):
        others = [o for o in knobs if o != k]
        rows = [r for r in B if all(r[o] == modal[o] for o in others)]
        lv = sorted({r[k] for r in rows},
                    key=lambda x: float(x) if k != "neg_mode" else x)
        ms, ss = [], []
        for v in lv:
            a = agg([r for r in rows if r[k] == v])
            ms.append(a[0]); ss.append(a[1])
        rng = max(ms) - min(ms)
        sensitive = rng >= 0.02
        col = ACCENT if sensitive else "#9aa7b4"
        if k in logx:
            xs = [float(v) for v in lv]
            ax.errorbar(xs, ms, yerr=ss, marker="o", ms=5, lw=2, color=col, capsize=0)
            ax.set_xscale("log")
            ax.set_xticks(xs, [str(v).rstrip("0").rstrip(".") if "." in str(v) else str(v)
                               for v in lv], fontsize=7.4)
            ax.minorticks_off()
        else:
            xs = np.arange(len(lv))
            ax.errorbar(xs, ms, yerr=ss, marker="o", ms=5, lw=2, color=col, capsize=0)
            ax.set_xticks(xs, lv, fontsize=7.4)
        if modal[k] in lv:
            i = lv.index(modal[k])
            xv = float(modal[k]) if k in logx else i
            ax.scatter([xv], [ms[i]], s=90, facecolor="none", edgecolor=BAD,
                       linewidth=1.6, zorder=6)
        ax.set_ylim(YLO, YHI)
        ax.set_title(f"{titles[k]}   ", loc="left", fontsize=9,
                     fontweight="bold" if sensitive else "normal")
        ax.text(1.0, 1.015, f"range {rng:.3f}", transform=ax.transAxes, ha="right",
                fontsize=7.6, color=BAD if sensitive else MUTED,
                fontweight="bold" if sensitive else "normal")
        ax.set_ylabel("test AUC" if k in ("epochs", "dropout") else "")
        style(ax)
    fig.suptitle("E6  Optimizer sensitivity, one factor at a time — shared y-axis, so "
                 "'flat' means genuinely insensitive",
                 x=.012, ha="left", fontsize=11, fontweight="bold", y=1.005)
    fig.text(.012, .955, "Blue = moves AUC by ≥0.02 across its range; grey = does not. "
                         "Red ring = the value in the stageB reference cell.  3 seeds, "
                         "bars are seed std.",
             fontsize=8.2, color=MUTED, ha="left")
    save(fig, "F04_ofat_sensitivity", fmts)


def f05_lr_tau(runs, base, ds, curves, fmts):
    """E7 -- the interaction OFAT cannot see. Sequential ramp: one hue, light->dark."""
    C = sel(runs, sweep="stageC")
    lrs = sorted({r["lr"] for r in C}, key=float)
    taus = sorted({r["temperature"] for r in C}, key=float)
    M = np.full((len(taus), len(lrs)), np.nan)
    for i, t in enumerate(taus):
        for j, l in enumerate(lrs):
            a = agg([r for r in C if r["temperature"] == t and r["lr"] == l])
            if a:
                M[i, j] = a[0]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
    for i in range(len(taus)):
        for j in range(len(lrs)):
            if not math.isnan(M[i, j]):
                # label inside the cell: white on dark fills, ink on light
                lum = (M[i, j] - np.nanmin(M)) / (np.nanmax(M) - np.nanmin(M))
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7.6,
                        color=SURFACE if lum > .62 else INK,
                        fontweight="bold" if M[i, j] == np.nanmax(M) else "normal")
    ax.set_xticks(range(len(lrs)), lrs)
    ax.set_yticks(range(len(taus)), taus)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("temperature τ")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    cb.set_label("test AUC", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("E7  lr × τ joint grid — at τ=0.05 the achievable AUC spans 0.749–0.812\n"
                 "depending on lr; at the shipped τ=0.5 lr looks almost inert  (3 seeds/cell)",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F05_lr_temperature_grid", fmts)


def f06_backbone(runs, base, ds, curves, fmts):
    """E8 + E18 -- accuracy, stability, cost. Three measures => three panels."""
    order = ["gat", "sage", "wsage", "gcn"]
    nice = {"gat": "GAT", "sage": "SAGE\n(shipped)", "wsage": "Weighted SAGE", "gcn": "GCN"}
    D = sel(runs, sweep="stageD", hidden_dim="64", num_layers="2")
    stats = {b: agg([r for r in D if r["backbone"] == b]) for b in order}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), gridspec_kw={"wspace": .32})

    ax = axes[0]
    for i, b in enumerate(order):
        m, s, n, vals = stats[b]
        ax.bar(i, m, width=.58, color=ACCENT if b == "sage" else "#b9c6d3")
        seeddots(ax, i, vals)
        barlabel(ax, i, max(vals), f"{m:.3f}")
    ax.set_xticks(range(4), [nice[b] for b in order])
    ax.set_ylim(0.75, 0.825)
    ax.set_ylabel("test AUC")
    style(ax)
    ax.set_title("Accuracy", loc="left", fontweight="bold")

    ax = axes[1]
    stds = [stats[b][1] for b in order]
    ax.bar(range(4), stds, width=.58,
           color=[GOOD if b == "gat" else "#b9c6d3" for b in order])
    for i, s in enumerate(stds):
        barlabel(ax, i, s, f"{s:.4f}", dy=.00006)
    ax.set_xticks(range(4), [nice[b] for b in order])
    ax.set_ylabel("seed-to-seed std of test AUC")
    style(ax)
    ax.set_title("Stability  (lower = better)", loc="left", fontweight="bold")
    ax.text(0, 1.005, "GAT has the LOWEST spread — contradicts the paper's §4.3.1 claim",
            transform=ax.transAxes, fontsize=7.6, color=GOOD, va="bottom")

    ax = axes[2]
    ms_ep = []
    for b in order:
        rows = [r for r in D if r["backbone"] == b]
        v = [F(r, "wall_clock_s") / max(F(r, "epochs_run"), 1) * 1000 for r in rows
             if F(r, "wall_clock_s") and F(r, "epochs_run")]
        ms_ep.append(float(np.mean(v)) if v else 0.0)
    ax.bar(range(4), ms_ep, width=.58, color="#b9c6d3")
    for i, v in enumerate(ms_ep):
        barlabel(ax, i, v, f"{v:.1f}", dy=.06)
    ax.set_xticks(range(4), [nice[b] for b in order])
    ax.set_ylabel("ms / epoch")
    style(ax)
    ax.set_title("Cost  (not a differentiator)", loc="left", fontweight="bold")

    fig.suptitle("E8 + E18  Backbone ablation at hidden=64, 2 layers  (5 seeds; "
                 "SAGE cell is the shared reference of three sub-sweeps)",
                 x=.012, ha="left", fontsize=11, fontweight="bold")
    save(fig, "F06_backbone", fmts)


def f07_capacity(runs, base, ds, curves, fmts):
    """E9 -- width and depth, the two 'reportably insensitive' knobs."""
    D = sel(runs, sweep="stageD", backbone="sage")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"wspace": .26})
    for ax, knob, other, oval, lab in [
            (axes[0], "hidden_dim", "num_layers", "2", "hidden dim"),
            (axes[1], "num_layers", "hidden_dim", "64", "num layers")]:
        rows = [r for r in D if r[other] == oval]
        lv = sorted({r[knob] for r in rows}, key=float)
        ms, ss = [], []
        for v in lv:
            a = agg([r for r in rows if r[knob] == v])
            ms.append(a[0]); ss.append(a[1])
        xs = np.arange(len(lv))
        ax.errorbar(xs, ms, yerr=ss, marker="o", ms=6, lw=2, color=ACCENT, capsize=0)
        ax.set_xticks(xs, lv)
        ax.set_xlabel(lab)
        ax.set_ylabel("test AUC")
        style(ax)
        spread = max(ms) - min(ms)
        ax.set_title(f"{lab}: total spread {spread:.3f} AUC", loc="left", fontsize=9)
        # shipped value ring
        shipped = "64" if knob == "hidden_dim" else "2"
        if shipped in lv:
            i = lv.index(shipped)
            ax.scatter([i], [ms[i]], s=95, facecolor="none", edgecolor=BAD, lw=1.6, zorder=6)
    fig.suptitle("E9  Capacity is reportably insensitive on this graph — red ring = shipped "
                 "(SAGE; 5 seeds)", x=.012, ha="left", fontsize=10.5, fontweight="bold")
    save(fig, "F07_capacity", fmts)


def f08_graphmae(runs, base, ds, curves, fmts):
    """E10 -- the headline: shipped Stage 2 is a no-op, repaired it is harmful.

    remask=True is the SHIPPED (buggy) re-mask; remask=False is the REPAIRED
    gradient path. Labels here say so explicitly -- 'remask on/off' would read
    backwards to any reader.
    """
    E = sel(runs, sweep="stageE")
    # The repaired Stage 1->2 bar uses the OFAT *default cell* only
    # (mask 0.75 / 100 ep / alpha 3.0). Pooling all 70 OFAT runs would average
    # over deliberately-varied knobs and no longer match the reported 0.752.
    recipes = [
        ("Stage 1 only\n(contrastive)", dict(stage="s1")),
        ("Stage 2 only\nas shipped", dict(stage="s2", remask="True")),
        ("Stage 1→2\nas shipped", dict(stage="s1->s2", remask="True")),
        ("Stage 2 only\nrepaired", dict(stage="s2", remask="False")),
        ("Stage 1→2\nrepaired", dict(stage="s1->s2", remask="False",
                                     mask_rate="0.75", mae_epochs="100", mae_alpha="3.0")),
    ]
    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], wspace=.24, hspace=.75)
    ax = fig.add_subplot(gs[:, 0])
    s1 = agg(sel(E, stage="s1"))[0]
    for i, (lab, filt) in enumerate(recipes):
        a = agg(sel(E, **filt))
        if not a:
            continue
        m, s, n, vals = a
        col = GOOD if i == 0 else (MUTED if "shipped" in lab else BAD)
        ax.bar(i, m, width=.6, color=col, alpha=.9)
        seeddots(ax, i, vals)
        barlabel(ax, i, max(vals), f"{m:.3f}")
        ax.text(i, 0.575, f"n={n}", ha="center", fontsize=7, color=MUTED)
    ax.axhline(s1, color=GOOD, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.text(4.48, s1, " Stage-1 baseline", va="center", fontsize=7.5, color=GOOD)
    ax.set_xticks(range(len(recipes)), [r[0] for r in recipes], fontsize=7.4)
    ax.set_ylim(0.55, 0.83)
    ax.set_ylabel("test AUC")
    style(ax)
    ax.set_title("Every repaired recipe lands BELOW Stage-1-only", loc="left",
                 fontweight="bold", fontsize=9.5, pad=6)

    # The repaired s1->s2 arm is OFAT (not a grid) around mask_rate=0.75,
    # mae_epochs=100, mae_alpha=3.0 -- so it gets three small panels, and each
    # selects rows with the other two knobs held at their default.
    R = sel(E, stage="s1->s2", remask="False")
    DEFAULTS = {"mask_rate": "0.75", "mae_epochs": "100", "mae_alpha": "3.0"}
    labels = {"mask_rate": "mask_rate", "mae_epochs": "mae_epochs", "mae_alpha": "SCE α"}
    allvals = []
    for row, knob in enumerate(DEFAULTS):
        axk = fig.add_subplot(gs[row, 1])
        others = [o for o in DEFAULTS if o != knob]
        rows_k = [r for r in R if all(r[o] == DEFAULTS[o] for o in others)]
        lv = sorted({r[knob] for r in rows_k if r[knob]}, key=float)
        ms, ss = [], []
        for v in lv:
            a = agg([r for r in rows_k if r[knob] == v])
            ms.append(a[0]); ss.append(a[1])
        allvals += ms
        axk.errorbar(range(len(lv)), ms, yerr=ss, marker="o", ms=4.5, lw=1.8,
                     color=BAD, capsize=0)
        axk.axhline(s1, color=GOOD, lw=1.2, ls=(0, (4, 3)), zorder=1)
        axk.set_xticks(range(len(lv)), lv, fontsize=7)
        axk.set_ylim(0.735, 0.80)
        axk.set_ylabel(labels[knob], fontsize=8)
        axk.tick_params(labelsize=7)
        style(axk)
        if row == 0:
            axk.set_title("Repaired Stage 1→2, one knob at a time "
                          "(green = Stage-1 baseline)",
                          loc="left", fontsize=8.4, fontweight="bold", pad=10)
    fig.text(.995, .02, f"every repaired setting lands in {min(allvals):.3f}–"
                        f"{max(allvals):.3f}, all below Stage-1's {s1:.3f}",
             ha="right", fontsize=7.4, color=BAD, fontweight="bold")

    fig.suptitle("E10  GraphMAE stage ablation — 'as shipped' = the re-mask bug that zeroes "
                 "every encoder gradient (a provable no-op)",
                 x=.012, ha="left", fontsize=11, fontweight="bold", y=1.04)
    save(fig, "F08_graphmae_stage", fmts)


def f09_final_configs(runs, base, ds, curves, fmts):
    """E11 + E12 -- the trade-off figure: overall AUC up, long-tail AUC down."""
    order = [("default", "Shipped default\n(20 epochs)"),
             ("default_long", "… then train to\nconvergence"),
             ("tuned_tau_only", "… then τ = 0.2"),
             ("tuned_sage", "… then lr, neg.\nratio, 1 layer"),
             ("tuned", "… then GAT\nbackbone"),
             ("tuned_joint_sage", "Stage-C joint\noptimum (SAGE)"),
             ("tuned_joint_gat", "Stage-C joint\noptimum (GAT)")]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(order))
    w = .38
    for i, (tag, _lab) in enumerate(order):
        rows = sel(runs, sweep="stageF", tag=tag)
        o = agg(rows, "test_auc")
        q = agg(rows, "auc_degq1")
        ax.bar(i - w / 2, o[0], width=w, color=ACCENT,
               label="overall test AUC" if i == 0 else None)
        ax.bar(i + w / 2, q[0], width=w, color=S[3],
               label="AUC, lowest-degree quartile (Q1)" if i == 0 else None)
        seeddots(ax, i - w / 2, o[3], color=INK, jitter=.045)
        seeddots(ax, i + w / 2, q[3], color=INK, jitter=.045)
        barlabel(ax, i - w / 2, max(o[3]), f"{o[0]:.3f}", size=7.5)
        barlabel(ax, i + w / 2, max(q[3]), f"{q[0]:.3f}", size=7.5, color="#8a6100")
    ax.set_xticks(x, [o[1] for o in order], fontsize=7.8)
    ax.set_ylim(0.55, 0.87)
    ax.set_ylabel("AUC (degree-matched)")
    ax.legend(loc="upper right", ncols=2)
    style(ax)
    # Mark the reference row. The label sits just under the axis, clear of the
    # bars -- inside the span it collides with the 0.752/0.790 pair.
    ax.axvspan(-.5, .5, color=ACCENT, alpha=.06, zorder=0)
    ax.annotate("REFERENCE — paper's Table 1", xy=(0, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -34), textcoords="offset points", ha="center",
                fontsize=7.2, color=ACCENT, fontweight="bold")
    ax.set_title("E11 + E12  Cumulative tuning: overall AUC and long-tail AUC move in "
                 "OPPOSITE directions\n"
                 "+0.072 overall (0.752→0.824) costs −0.103 on the rarest 25% of objects "
                 "(0.790→0.687).  10 seeds.",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F09_final_configs", fmts)


def f10_degree_quartiles(runs, base, ds, curves, fmts):
    """E12 -- the same trade-off resolved across all four degree quartiles."""
    arms = [("default", "Shipped (20 ep)", S[0]), ("default_long", "Converged", S[2]),
            ("tuned", "Tuned (GAT)", S[1])]
    qs = ["auc_degq1", "auc_degq2", "auc_degq3", "auc_degq4"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for tag, lab, col in arms:
        rows = sel(runs, sweep="stageF", tag=tag)
        ms = [agg(rows, q)[0] for q in qs]
        ax.plot(range(4), ms, marker="o", ms=7, lw=2, color=col, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{ms[0]:.3f}", (0, ms[0]), textcoords="offset points",
                    xytext=(-8, 0), ha="right", fontsize=7.5, color=col, fontweight="bold")
    ax.set_xticks(range(4), ["Q1\n(rarest)", "Q2", "Q3", "Q4\n(hubs)"])
    ax.set_xlabel("node degree quartile")
    ax.set_ylabel("test AUC")
    ax.legend(loc="lower right")
    style(ax)
    ax.set_title("E12  Where tuning helps and where it hurts\n"
                 "Tuning wins on hubs (Q3–Q4) and loses on the long tail (Q1) — the "
                 "modality's own selling point",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F10_degree_quartiles", fmts)


def f11_strata(runs, base, ds, curves, fmts):
    """E13 -- AUC by co-occurrence frequency. delta>=2 rows are dropped, not imputed."""
    arms = [("default", "Shipped (20 ep)", S[0]), ("default_long", "Converged", S[2]),
            ("tuned", "Tuned (GAT)", S[1])]
    strata = [("auc_c=1", "n_c=1", "c=1\n(singleton)"),
              ("auc_c=2-4", "n_c=2-4", "c=2–4"),
              ("auc_c>=5", "n_c>=5", "c≥5")]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    w = .26
    for k, (tag, lab, col) in enumerate(arms):
        rows = sel(runs, sweep="stageF", tag=tag)
        ms = [agg(rows, s[0])[0] for s in strata]
        xs = np.arange(len(strata)) + (k - 1) * w
        ax.bar(xs, ms, width=w * .92, color=col, label=lab)
        for x, m in zip(xs, ms):
            ax.text(x, m + .004, f"{m:.3f}", ha="center", fontsize=6.8, color=INK)
    ns = [agg(sel(runs, sweep="stageF", tag="default"), s[1])[0] for s in strata]
    ax.set_xticks(range(len(strata)),
                  [f"{s[2]}\nn≈{int(n)}" for s, n in zip(strata, ns)])
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.5, 0.95)
    ax.legend(loc="upper left", ncols=3)
    style(ax)
    ax.set_title("E13  AUC by co-occurrence frequency — rare edges are the hard case\n"
                 "Singleton edges are 62% of the graph.  10 seeds.",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F11_cooccurrence_strata", fmts)


def f12_geometry(runs, base, ds, curves, fmts):
    """E14 + E15 -- four diagnostics, four separate y-scales, never one axis."""
    arms = [("default", "Shipped"), ("default_long", "Converged"), ("tuned", "Tuned")]
    panels = [("effective_rank", "Effective rank  (of 64)", "higher = less collapsed"),
              ("degree_bias_rho", "Degree bias  ρ(deg, cos)", "higher = more hub-biased"),
              ("alignment", "Alignment  (lower = better)", "linked objects pulled closer"),
              ("knn_room_purity", "kNN room purity", "DIAGNOSTIC ONLY — partly circular")]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5), gridspec_kw={"wspace": .38})
    for ax, (key, title, sub) in zip(axes, panels):
        for i, (tag, lab) in enumerate(arms):
            a = agg(sel(runs, sweep="stageF", tag=tag), key)
            col = S[3] if key == "knn_room_purity" else ACCENT
            ax.bar(i, a[0], width=.58, color=col)
            seeddots(ax, i, a[3])
            ax.text(i, max(a[3]) + (max(a[3]) - min(a[3])) * .12 + abs(a[0]) * .01,
                    f"{a[0]:.2f}", ha="center", fontsize=7.6, fontweight="bold")
        ax.set_xticks(range(3), [a[1] for a in arms], fontsize=7.6)
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=7,
                color=WARN if "DIAGNOSTIC" in sub else MUTED, va="bottom")
        style(ax)
    fig.suptitle("E14 + E15  Embedding diagnostics — each has its own scale, so they get "
                 "their own panel  (10 seeds)",
                 x=.012, ha="left", fontsize=11, fontweight="bold")
    save(fig, "F12_embedding_geometry", fmts)


def f13_curves(runs, base, ds, curves, fmts):
    """E16 -- is the 20-epoch budget converged? (spoiler: no)"""
    arms = [("default_long", "Shipped recipe, run long (SAGE)", S[0]),
            ("tuned", "Tuned (GAT)", S[1])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for tag, lab, col in arms:
        rows = sel(runs, sweep="stageF", tag=tag, seed="0")
        if not rows:
            continue
        pts = curves.get(rows[0]["run_key"], [])
        if not pts:
            continue
        ep = [p[0] for p in pts]; va = [p[1] for p in pts]
        ax.plot(ep, va, lw=2, color=col, label=lab)
        ax.scatter([ep[-1]], [va[-1]], s=42, color=col, zorder=5,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate(f"{va[-1]:.3f}", (ep[-1], va[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=8, color=col, fontweight="bold")
    ax.axvline(20, color=BAD, lw=1.4, ls=(0, (4, 3)))
    ax.text(21, ax.get_ylim()[0] + .004, "shipped budget\n= 20 epochs", fontsize=7.6,
            color=BAD, fontweight="bold", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("epoch  (log scale)")
    ax.set_ylabel("validation AUC")
    ax.legend(loc="lower right")
    style(ax)
    ax.set_title("E16  The shipped 20-epoch budget stops far short of convergence\n"
                 "seed 0 shown; averaging curves across seeds would blur the very thing "
                 "this panel is for",
                 loc="left", fontsize=9.5, pad=10)
    save(fig, "F13_training_curves", fmts)


def f14_leakage(runs, base, ds, curves, fmts):
    """E17 -- the tripwire. train_auc must sit above test_auc on every run."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sweeps = sorted({r["sweep"] for r in runs})
    for i, sw in enumerate(sweeps):
        rows = sel(runs, sweep=sw)
        xs = [F(r, "test_auc") for r in rows]
        ys = [F(r, "train_auc") for r in rows]
        ax.scatter(xs, ys, s=13, color=S[i % len(S)], label=sw, alpha=.75,
                   edgecolor=SURFACE, linewidth=.4)
    lo, hi = 0.55, 0.90
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(hi - .01, hi - .015, "train = test", fontsize=7.5, color=MUTED,
            ha="right", rotation=38)
    below = sum(1 for r in runs if F(r, "train_auc") is not None
                and F(r, "test_auc") is not None and F(r, "train_auc") < F(r, "test_auc"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("test AUC"); ax.set_ylabel("train AUC")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", ncols=2, fontsize=7.5)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=1.0)
    ax.set_title(f"E17  Leakage tripwire — all 447 runs\n"
                 f"{below} runs fall below the diagonal "
                 f"({'clean' if below == 0 else 'INVESTIGATE'})",
                 loc="left", fontsize=9.5, pad=10,
                 color=INK if below == 0 else BAD)
    save(fig, "F14_leakage_tripwire", fmts)


FIGS = {
    "F01": f01_baselines, "F02": f02_edge_weighting, "F03": f03_threshold,
    "F04": f04_ofat, "F05": f05_lr_tau, "F06": f06_backbone, "F07": f07_capacity,
    "F08": f08_graphmae, "F09": f09_final_configs, "F10": f10_degree_quartiles,
    "F11": f11_strata, "F12": f12_geometry, "F13": f13_curves, "F14": f14_leakage,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="figure ids, e.g. F09 F06")
    ap.add_argument("--fmt", nargs="*", default=["png", "pdf", "svg"])
    a = ap.parse_args()
    data = load()
    todo = a.only or sorted(FIGS)
    print(f"writing to {OUT}/")
    for fid in todo:
        FIGS[fid.upper()](*data, a.fmt)
