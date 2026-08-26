#!/usr/bin/env python3
"""Every figure for the encoder study report, as PNG (for Notion) and PDF (for LaTeX).

    python scripts/figures_objectgraph.py --out docs/notion/images

Reads only sweeps/results.csv, the per-run JSON histories and eval_baselines.json. Writes
nothing outside --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
# Keep glyphs as text rather than paths: cuts SVG size ~4x, which matters because Notion only
# accepts an SVG inline as UTF-8 content and caps it at 200 KiB.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.graph_data import build_cooccurrence, load_scenes

# A single palette so the same configuration is the same colour in every figure -- across ten
# plots the reader learns the mapping once instead of re-reading ten legends.
C_SHIPPED, C_LONG, C_TUNED, C_BASE, C_BAD = "#c0392b", "#2980b9", "#27ae60", "#7f8c8d", "#8e44ad"
SEED_STD = 0.003  # measured same-config spread; anything smaller than this is not a result


def save(fig, out: Path, name: str) -> None:
    # svg as well as png/pdf: Notion will not take a local raster upload, but it accepts an SVG
    # inline as text, so the vector copy is what actually gets embedded in the writeup.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg")


def agg(d: pd.DataFrame, by, cols=("test_auc",)):
    g = d.groupby(by, dropna=False)
    out = g[list(cols)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n"] = g.size()
    return out.reset_index()


def node_coverage() -> dict[int, int]:
    """Non-isolated node count per co-occurrence threshold, measured from the scenes.

    The sweep's `n_nodes` column cannot answer this: thresholding drops edges but keeps the
    node set at 1,068, so plotting `n_nodes` gives a flat 100% line and hides the whole cost
    of Eq. (2)'s threshold.
    """
    nodes, pair_counts, _ = build_cooccurrence(load_scenes())
    return {t: len({i for pair, c in pair_counts.items() if c >= t for i in pair})
            for t in (1, 2, 3, 5, 10)} | {"total": len(nodes)}


def ofat_rows(d: pd.DataFrame, axis: str, base: dict) -> pd.DataFrame:
    """Rows where every axis other than `axis` sits at its base value.

    Without this an OFAT plot silently averages over the other axes' off-base cells and every
    curve gets flattened toward the global mean.
    """
    m = pd.Series(True, index=d.index)
    for k, v in base.items():
        if k != axis and k in d.columns:
            m &= d[k].astype(str) == str(v)
    return d[m]


# --------------------------------------------------------------------------------------


def fig_training_curve(df, out, sweeps):
    curves: dict[str, list] = {}
    for p in sorted(sweeps.glob("*/*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        tag = j.get("cfg", {}).get("tag", "")
        if tag in ("default_long", "tuned") and len(j.get("history", [])) > 5:
            curves.setdefault(tag, []).append(j["history"])
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style = {"default_long": (C_LONG, "shipped config (SAGE, $\\tau$=0.5, lr=$10^{-3}$)"),
             "tuned": (C_TUNED, "tuned config (GAT, 1 layer, $\\tau$=0.2)")}
    for tag, cs in sorted(curves.items()):
        L = min(len(c) for c in cs)
        ep = [h["epoch"] for h in cs[0][:L]]
        M = np.array([[h["val_auc"] for h in c[:L]] for c in cs])
        col, lab = style[tag]
        ax.plot(ep, M.mean(0), color=col, lw=2.2, label=f"{lab}, {len(cs)} seeds")
        ax.fill_between(ep, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col, alpha=0.18)
    ax.axvline(DEFAULT_CFG["epochs"], color=C_SHIPPED, ls="--", lw=1.8)
    ax.annotate(f"shipped budget\n({DEFAULT_CFG['epochs']} epochs)", xy=(DEFAULT_CFG["epochs"], 0.74),
                xytext=(26, 0.70), color=C_SHIPPED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_SHIPPED))
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Validation AUC (degree-matched negatives)")
    ax.set_title("The shipped encoder stops long before convergence", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, out, "fig01_val_auc_vs_epoch")


def fig_ofat(df, out):
    """Every one-factor-at-a-time axis on one page, each with the shipped value marked."""
    b = df[df.sweep == "stageB"]
    d = df[df.sweep == "stageD"]
    base = {"epochs": 1000, "lr": 0.001, "temperature": 0.5, "neg_mode": "uniform",
            "neg_ratio": 1.0, "dropout": 0.0, "hidden_dim": 64, "num_layers": 2, "backbone": "sage"}
    panels = [
        (b, "epochs", "Epochs", True, DEFAULT_CFG["epochs"]),
        (b, "lr", "Learning rate", True, DEFAULT_CFG["lr"]),
        (b, "temperature", r"Temperature $\tau$", True, DEFAULT_CFG["temperature"]),
        (b, "neg_ratio", "Negative ratio", False, 1.0),
        (b, "dropout", "Dropout", False, 0.0),
        (d, "hidden_dim", "Hidden dim", True, DEFAULT_CFG["hidden_dim"]),
        (d, "num_layers", "Number of layers", False, 2),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (src, axis, label, logx, shipped) in zip(axes.ravel(), panels):
        sub = ofat_rows(src, axis, base)
        if sub.empty or axis not in sub.columns:
            ax.axis("off")
            continue
        sub = sub.copy()
        sub[axis] = pd.to_numeric(sub[axis], errors="coerce")
        a = agg(sub, [axis]).dropna(subset=[axis]).sort_values(axis)
        ax.errorbar(a[axis], a.test_auc_mean, yerr=a.test_auc_std, marker="o", ms=5,
                    capsize=3, lw=1.8, color=C_LONG)
        ax.axvline(shipped, color=C_SHIPPED, ls="--", lw=1.5)
        ax.text(0.03, 0.06, f"shipped = {shipped:g}", transform=ax.transAxes,
                color=C_SHIPPED, fontsize=8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Test AUC")
        ax.grid(alpha=0.3)
        span = a.test_auc_mean.max() - a.test_auc_mean.min()
        ax.set_title(f"range {span:.3f}" + ("  (< seed noise)" if span < SEED_STD else ""),
                     fontsize=9, color="black" if span >= SEED_STD else C_BASE)
    # Last panel: negative sampling mode, categorical.
    ax = axes.ravel()[7]
    sub = ofat_rows(b, "neg_mode", base)
    if not sub.empty:
        a = agg(sub, ["neg_mode"])
        ax.bar(a.neg_mode, a.test_auc_mean, yerr=a.test_auc_std, capsize=4,
               color=[C_SHIPPED if m == "uniform" else C_LONG for m in a.neg_mode], width=0.55)
        ax.set_ylim(0.70, 0.82)
        ax.set_xlabel("Training negatives")
        ax.set_ylabel("Test AUC")
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("shipped = uniform", fontsize=9)
    fig.suptitle("One-factor-at-a-time sensitivity, 3–5 seeds per point, shipped value dashed red",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "fig02_ofat_sensitivity")


def fig_lr_tau(df, out):
    """The interaction OFAT cannot see. Selection is on validation, so validation is plotted."""
    d = df[df.sweep == "stageC"]
    if d.empty:
        return
    d = d.copy()
    for c in ("lr", "temperature", "val_auc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    piv = d.pivot_table(index="temperature", columns="lr", values="val_auc", aggfunc="mean")
    piv = piv.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{v:g}" for v in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < piv.values.max() - 0.04 else "black")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(r"Temperature $\tau$")
    ax.set_title(r"lr $\times$ $\tau$ interaction: at $\tau$=0.05 AUC spans 0.749–0.812,"
                 "\n" r"at $\tau$=0.2 it is flat — OFAT at one lr cannot see this", fontsize=10)
    fig.colorbar(im, ax=ax, label="Validation AUC")
    save(fig, out, "fig03_lr_temperature_interaction")


def fig_backbone(df, out):
    d = df[(df.sweep == "stageD") & (df.hidden_dim.astype(str) == "64")
           & (df.num_layers.astype(str) == "2")]
    if d.empty:
        return
    d = d.copy()
    d["ms_per_epoch"] = 1000 * d.wall_clock_s.astype(float) / d.epochs_run.astype(float)
    a = agg(d, ["backbone"], cols=("test_auc", "ms_per_epoch"))
    a["n_params"] = d.groupby("backbone").n_params.first().values
    order = ["gcn", "sage", "wsage", "gat"]
    a = a.set_index("backbone").loc[[b for b in order if b in a.backbone.values]].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    names = {"sage": "SAGE\n(shipped)", "wsage": "Weighted\nSAGE", "gat": "GAT", "gcn": "GCN"}
    cols = [C_SHIPPED if b == "sage" else (C_TUNED if b == "gat" else C_LONG) for b in a.backbone]
    lab = [names.get(b, b) for b in a.backbone]
    axes[0].bar(lab, a.test_auc_mean, yerr=a.test_auc_std, capsize=4, color=cols, width=0.6)
    axes[0].set_ylim(0.74, 0.83)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("GAT wins on the mean...", fontsize=10)
    axes[1].bar(lab, a.test_auc_std, color=cols, width=0.6)
    axes[1].set_ylabel("Seed std of test AUC")
    axes[1].set_title("...and on stability, contradicting\n"
                      r'$\S$4.3.1 "less stable"', fontsize=10)
    axes[2].bar(lab, a.ms_per_epoch_mean, yerr=a.ms_per_epoch_std, capsize=4, color=cols, width=0.6)
    axes[2].set_ylabel("ms / epoch (RTX 4090)")
    axes[2].set_title("Cost is not a differentiator\nat this graph size", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(a.itertuples()):
        axes[2].text(i, r.ms_per_epoch_mean + 0.6, f"{int(r.n_params/1000)}k params",
                     ha="center", fontsize=7.5, color=C_BASE)
    fig.tight_layout()
    save(fig, out, "fig04_backbone")


def fig_backbone_loss(out, sweeps):
    """Training loss and val AUC per epoch for the two backbones, at the matched 64/2 cell.

    fig04 compares the backbones on their end points; this compares the paths they take there.
    Only stageD carries GAT, and only at hidden_dim=64 / num_layers=2, so that is the one cell
    where the comparison is not confounded by width or depth.
    """
    runs: dict[str, dict[int, list]] = {"sage": {}, "gat": {}}
    n_params: dict[str, int] = {}
    for p in sorted((sweeps / "stageD").glob("*.json")):
        j = json.loads(p.read_text())
        c = j["cfg"]
        if c.get("backbone") not in runs or (c.get("hidden_dim"), c.get("num_layers")) != ("64", "2"):
            continue
        # Overlapping sweeps wrote the sage 64/2 cell three times per seed; the runs are the same
        # config and land within 4e-5 of each other, so keep one per seed rather than triple-weight
        # sage in the mean.
        runs[c["backbone"]].setdefault(j["seed"], j["history"])
        n_params[c["backbone"]] = j["metrics"]["n_params"]
    if not runs["gat"] or not runs["sage"]:
        return
    # These runs predate DEFAULT_CFG's current eval_every, so take the cadence from the history
    # itself rather than labelling the axis with a number the runs never used.
    cadence = runs["gat"][min(runs["gat"])][0]["epoch"]

    style = {"sage": (C_SHIPPED, "SAGE (shipped)"), "gat": (C_TUNED, "GAT (tuned)")}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for bb, hs in runs.items():
        col, lab = style[bb]
        # Early stopping fires at a different epoch per seed, so a mean band is only defined over
        # the prefix where every seed is still running; individual seeds are drawn to their own end.
        L = min(len(h) for h in hs.values())
        ep = [r["epoch"] for r in next(iter(hs.values()))[:L]]
        for key, ax in ((("loss"), axes[0]), (("val_auc"), axes[1])):
            for h in hs.values():
                ax.plot([r["epoch"] for r in h], [r[key] for r in h], color=col, lw=0.7, alpha=0.35)
            M = np.array([[r[key] for r in h[:L]] for h in hs.values()])
            ax.plot(ep, M.mean(0), color=col, lw=2.4,
                    label=f"{lab}, {len(hs)} seeds, {n_params[bb]/1000:.0f}k params")
    axes[0].set_ylabel("Training loss (InfoNCE)")
    axes[0].set_title("GAT starts lower but plateaus higher", fontsize=10)
    axes[1].set_ylabel("Validation AUC (degree-matched negatives)")
    axes[1].set_title("GAT peaks later (epoch 130-570 vs 90-120);\nboth then decline", fontsize=10)
    for ax in axes:
        # Log x: one GAT seed early-stopped at epoch 770 while the rest finished by 460, and on a
        # linear axis that single tail flattens the first 100 epochs, which is where they differ.
        ax.set_xscale("log")
        ax.set_xlabel(f"Epoch, log scale (sampled every {cadence} epochs)")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Object-graph encoder, SAGE vs GAT at the matched cell (hidden 64, 2 layers). "
                 "Thin lines are seeds; the\nthick mean stops where the first seed early-stopped. "
                 "The two backbones differ in parameter count,\nso the absolute loss gap is not "
                 "interpretable on its own -- read the shapes.\nPeak validation AUC is a near-tie "
                 "(0.791 vs 0.793, seed sd 0.002); GAT's +0.017 test-AUC lead in fig04 is not "
                 "visible here.", fontsize=9.5, y=1.16)
    fig.tight_layout()
    save(fig, out, "fig15_backbone_loss_curves")


def fig_min_cooc(df, out):
    d = df[(df.sweep == "stageA") & (df.edge_mode == DEFAULT_CFG["edge_mode"])]
    if d.empty:
        return
    d = d.copy()
    d["min_cooc"] = pd.to_numeric(d.min_cooc)
    a = agg(d, ["min_cooc"]).sort_values("min_cooc")
    ncov = node_coverage()
    total = ncov["total"]
    cov = np.array([ncov[int(t)] for t in a.min_cooc], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(a.min_cooc, a.test_auc_mean, yerr=a.test_auc_std, marker="o", ms=6,
                capsize=4, lw=2, color=C_LONG, label="Test AUC")
    ax.set_xlabel(r"Co-occurrence threshold $\delta$ (Eq. 2)")
    ax.set_ylabel("Test AUC", color=C_LONG)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(a.min_cooc, 100 * cov / total, marker="s", ms=5, ls="--", color=C_SHIPPED,
             label="Node coverage")
    for t, c in zip(a.min_cooc, cov):
        ax2.annotate(f"{int(c)}", (t, 100 * c / total), textcoords="offset points",
                     xytext=(4, 5), fontsize=8, color=C_SHIPPED)
    ax2.set_ylabel(f"Non-isolated nodes (% of {total:,})", color=C_SHIPPED)
    ax2.set_ylim(0, 105)
    drop = 100 * (1 - cov[list(a.min_cooc).index(2)] / total) if 2 in set(a.min_cooc) else 0
    ax.set_title("Thresholding never helps AUC and destroys the long tail:\n"
                 rf"$\delta$=2 already isolates {drop:.0f}% of object labels", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig05_min_cooc")


def fig_edge_mode(df, out):
    d = df[(df.sweep == "stageA") & (df.min_cooc.astype(str) == "1")]
    if d.empty:
        return
    a = agg(d, ["edge_mode"])
    order = ["dedup", "multiplicity", "weighted"]
    a = a.set_index("edge_mode").loc[[o for o in order if o in a.edge_mode.values]].reset_index()
    names = {"dedup": "Unweighted\n(binary)", "multiplicity": "Count-weighted\n(shipped)",
             "weighted": r"$c_{ab}/\sqrt{c_a c_b}$" + "\n(paper Eq. 2)"}
    cols = [C_SHIPPED if m == "multiplicity" else C_LONG for m in a.edge_mode]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar([names[m] for m in a.edge_mode], a.test_auc_mean, yerr=a.test_auc_std,
           capsize=4, color=cols, width=0.6)
    ax.axhline(a.test_auc_mean.max(), color=C_BASE, ls=":", lw=1)
    ax.set_ylim(0.76, 0.80)
    ax.set_ylabel("Test AUC")
    ax.set_title("Edge weighting: what ships beats what the paper claims", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for i, r in enumerate(a.itertuples()):
        ax.text(i, r.test_auc_mean + r.test_auc_std + 0.0015, f"{r.test_auc_mean:.4f}",
                ha="center", fontsize=8.5)
    save(fig, out, "fig06_edge_weighting")


def fig_strata(df, out):
    """Where the encoder's accuracy actually comes from -- by edge frequency and by degree."""
    d = df[(df.sweep == "stageF") & (df.tag.isin(["default", "default_long", "tuned"]))]
    if d.empty:
        return
    labels = {"default": ("Shipped (20 epochs)", C_SHIPPED),
              "default_long": ("Shipped, converged", C_LONG),
              "tuned": ("Tuned", C_TUNED)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, cols, xt, title in (
        (axes[0], ["auc_c=1", "auc_c=2-4", "auc_c>=5"],
         ["$c_{ab}$ = 1\n(62% of edges)", "$c_{ab}$ = 2–4", r"$c_{ab} \geq$ 5"],
         "By co-occurrence count: singleton edges are the hard ones"),
        (axes[1], ["auc_degq1", "auc_degq2", "auc_degq3", "auc_degq4"],
         ["Q1\n(rarest)", "Q2", "Q3", "Q4\n(hubs)"],
         "By endpoint degree: tuning helps hubs, hurts the long tail"),
    ):
        x = np.arange(len(cols))
        w = 0.26
        for i, (tag, (lab, col)) in enumerate(labels.items()):
            sub = d[d.tag == tag]
            m = [sub[c].astype(float).mean() for c in cols]
            s = [sub[c].astype(float).std() for c in cols]
            ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=3, label=lab, color=col)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.text(len(cols) - 0.6, 0.505, "chance", fontsize=7.5, color="k")
        ax.set_xticks(x, xt)
        ax.set_ylabel("Test AUC")
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, out, "fig07_strata_and_degree")


def fig_stage(df, out):
    d = df[df.sweep == "stageE"].copy()
    if d.empty:
        return
    d["remask"] = d.remask.astype(str).str.lower()
    plain = d.tag.fillna("").eq("") & d.mask_rate.isna() & d.mae_epochs.isna() & d.mae_alpha.isna()
    arms = [
        ("Stage 1 only\n(contrastive)", plain & d.stage.eq("s1"), C_LONG),
        ("Stage 2 only\nAS SHIPPED", d.tag.eq("s2_shipped"), C_BAD),
        ("Stage 1$\\to$2\nAS SHIPPED", plain & d.stage.eq("s1->s2") & d.remask.eq("true"), C_BAD),
        ("Stage 2 only\nrepaired", plain & d.stage.eq("s2") & d.remask.eq("false"), C_SHIPPED),
        ("Stage 1$\\to$2\nrepaired", plain & d.stage.eq("s1->s2") & d.remask.eq("false"), C_SHIPPED),
        ("Tuned $\\to$ 2\nrepaired", d.tag.eq("tuned_s1s2"), C_SHIPPED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    labs = [a[0] for a in arms]
    m = [d[a[1]].test_auc.astype(float).mean() for a in arms]
    s = [d[a[1]].test_auc.astype(float).std() for a in arms]
    axes[0].bar(labs, m, yerr=s, capsize=4, color=[a[2] for a in arms], width=0.62)
    axes[0].axhline(m[0], color=C_LONG, ls="--", lw=1.4)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_ylim(0.55, 0.86)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_title("As shipped, stage 2 is provably a no-op (purple bars reproduce stage 1\n"
                      "exactly); repaired, it is harmful", fontsize=10)
    for i, (mi, si) in enumerate(zip(m, s)):
        axes[0].text(i, mi + si + 0.006, f"{mi:.3f}", ha="center", fontsize=8.5)
    # Right: every stage-2 hyperparameter, repaired, against the stage-1 line.
    ax = axes[1]
    off = 0
    ticks, tlabs = [], []
    for axis, name, col in (("mask_rate", "mask rate", C_TUNED),
                            ("mae_epochs", "stage-2 epochs", C_LONG),
                            ("mae_alpha", r"SCE $\alpha$", C_SHIPPED)):
        sub = d[d.remask.eq("false") & d.tag.fillna("").eq("") & d[axis].notna()]
        a = agg(sub, [axis]).sort_values(axis)
        xs = np.arange(len(a)) + off
        ax.errorbar(xs, a.test_auc_mean, yerr=a.test_auc_std, marker="o", ms=5, capsize=3,
                    lw=1.8, color=col, label=name)
        ticks += list(xs)
        tlabs += [f"{float(v):g}" for v in a[axis]]
        off += len(a) + 0.8
    ax.axhline(m[0], color="k", ls="--", lw=1.5)
    ax.text(0.02, m[0] + 0.002, "stage 1 alone", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xticks(ticks, tlabs, fontsize=8)
    ax.set_ylabel("Test AUC")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("No stage-2 setting reaches the stage-1 baseline", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig08_stage_ablation")


def fig_baselines(df, out, baselines: dict | None):
    if not baselines:
        return
    b = baselines.get("baselines", {})
    rows = [("Preferential attachment", b.get("preferential_attachment", {}).get("auc"), C_BASE),
            ("Untrained random SAGE", b.get("untrained_sage", {}).get("auc"), C_BASE),
            ("Raw MiniLM cosine\n(no message passing)", b.get("raw_minilm_cosine", {}).get("auc"), C_BASE),
            ("Common neighbours", b.get("common_neighbors", {}).get("auc"), C_BASE),
            ("Adamic–Adar", b.get("adamic_adar", {}).get("auc"), C_BAD)]
    f = df[df.sweep == "stageF"]
    for tag, lab, col in (("default", "ObjectGraph, shipped\n(20 epochs)", C_SHIPPED),
                          ("default_long", "ObjectGraph, converged", C_LONG),
                          ("tuned", "ObjectGraph, tuned", C_TUNED)):
        s = f[f.tag == tag]
        if not s.empty:
            rows.append((lab, s.test_auc.astype(float).mean(), col))
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.004, i, f"{r[1]:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.45, 0.88)
    ax.set_xlabel("Test AUC (degree-matched negatives)")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title("Sanity floor: the encoder must beat both the structural baselines\n"
                 "and its own untrained initialisation", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    save(fig, out, "fig09_baselines")


def fig_negatives(df, out, baselines: dict | None):
    """Why the evaluation protocol uses degree-matched negatives at all."""
    if not baselines:
        return
    b = baselines.get("baselines", {})
    names = [("adamic_adar", "Adamic–Adar"), ("common_neighbors", "Common neigh."),
             ("preferential_attachment", "Pref. attach."), ("raw_minilm_cosine", "Raw MiniLM")]
    f = df[(df.sweep == "stageF") & (df.tag == "tuned")]
    u = [b[k]["auc_uniform"] for k, _ in names] + [f.test_auc_uniform.astype(float).mean()]
    g = [b[k]["auc"] for k, _ in names] + [f.test_auc.astype(float).mean()]
    lab = [n for _, n in names] + ["ObjectGraph\n(tuned)"]
    x = np.arange(len(lab))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - 0.2, u, 0.4, label="Uniform negatives (literature default)", color=C_BASE)
    ax.bar(x + 0.2, g, 0.4, label="Degree-matched negatives (this study)", color=C_LONG)
    ax.set_xticks(x, lab, fontsize=8.5)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Against uniform negatives every method scores 0.92–0.96 and the protocol\n"
                 "cannot discriminate; degree-matching restores resolving power", fontsize=10)
    save(fig, out, "fig10_negative_sampling")


def fig_secondary(df, out):
    d = df[(df.sweep == "stageF") & (df.tag.isin(["default", "default_long", "tuned"]))]
    if d.empty:
        return
    order = ["default", "default_long", "tuned"]
    names = ["Shipped\n(20 ep)", "Converged", "Tuned"]
    cols = [C_SHIPPED, C_LONG, C_TUNED]
    metrics = [("effective_rank", "Effective rank (of 64)", "higher = less collapse"),
               ("degree_bias_rho", r"Degree bias $\rho$", "lower = fairer to rare objects"),
               ("alignment", "Alignment", "lower = linked objects closer"),
               ("uniformity", "Uniformity", "lower = better spread on the sphere"),
               ("knn_room_purity", "kNN room purity", "diagnostic only, partly circular")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, (col, title, sub) in zip(axes, metrics):
        m = [d[d.tag == t][col].astype(float).mean() for t in order]
        s = [d[d.tag == t][col].astype(float).std() for t in order]
        ax.bar(names, m, yerr=s, capsize=4, color=cols, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(sub, fontsize=8, color=C_BASE)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Secondary embedding diagnostics, 10 seeds each", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig11_secondary_metrics")


def fig_final(df, out):
    d = df[df.sweep == "stageF"]
    if d.empty:
        return
    order = [("default", "Shipped default\n(20 epochs)", C_SHIPPED),
             ("default_long", "+ trained to\nconvergence", C_LONG),
             ("tuned_tau_only", r"+ $\tau$ = 0.2", C_LONG),
             ("tuned_sage", "+ lr, neg ratio,\n1 layer", C_LONG),
             ("tuned", "+ GAT backbone\n(TUNED)", C_TUNED),
             ("tuned_joint_sage", "Stage-C joint\noptimum (SAGE)", C_BAD),
             ("tuned_joint_gat", "Stage-C joint\noptimum (GAT)", C_BAD)]
    order = [o for o in order if not d[d.tag == o[0]].empty]
    m = [d[d.tag == t].test_auc.astype(float).mean() for t, _, _ in order]
    s = [d[d.tag == t].test_auc.astype(float).std() for t, _, _ in order]
    low = [d[d.tag == t].auc_degq1.astype(float).mean() for t, _, _ in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, m, 0.4, yerr=s, capsize=4, color=[o[2] for o in order], label="Overall AUC")
    ax.bar(x + 0.2, low, 0.4, color=[o[2] for o in order], alpha=0.45,
           label="AUC, lowest degree quartile")
    ax.set_xticks(x, [o[1] for o in order], fontsize=8)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.6, 0.87)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    for i, (mi, si, li) in enumerate(zip(m, s, low)):
        ax.text(i - 0.2, mi + si + 0.005, f"{mi:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, li + 0.005, f"{li:.3f}", ha="center", fontsize=8, color=C_BASE)
    ax.set_title("Cumulative effect of tuning (10 seeds each). Overall AUC rises +0.072; the "
                 "long tail\nfalls. The last two bars show coordinate-wise optima failing to "
                 "compose.", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig12_final_configurations")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CFG["log_dir"] / "results.csv")
    p.add_argument("--out", type=Path, default=ROOT / "docs/notion/images")
    p.add_argument("--baselines", type=Path, default=ROOT / "docs/notion/eval_baselines.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    baselines = json.loads(args.baselines.read_text()) if args.baselines.exists() else None
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} runs -> {args.out}")

    fig_training_curve(df, args.out, args.csv.parent)
    fig_ofat(df, args.out)
    fig_lr_tau(df, args.out)
    fig_backbone(df, args.out)
    fig_backbone_loss(args.out, args.csv.parent)
    fig_min_cooc(df, args.out)
    fig_edge_mode(df, args.out)
    fig_strata(df, args.out)
    fig_stage(df, args.out)
    fig_baselines(df, args.out, baselines)
    fig_negatives(df, args.out, baselines)
    fig_secondary(df, args.out)
    fig_final(df, args.out)


if __name__ == "__main__":
    main()
