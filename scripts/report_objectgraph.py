#!/usr/bin/env python3
"""Turn sweeps/results.csv into LaTeX table bodies and figures.

Hand-transcribing a CSV into a paper is where errors enter, so every number that appears in
the draft is generated here.

    python scripts/report_objectgraph.py --tables --figures

Tables match the existing tab:threshold style (\\begin{latin}, \\rowcolor{gray!15} on the
shipped configuration) so they paste straight into the draft.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.graph_data import build_cooccurrence, load_scenes

PRIMARY = "test_auc"


def node_coverage() -> dict[int, int]:
    """Non-isolated node count per co-occurrence threshold, measured from the scenes.

    The sweep's `n_nodes` column is 1,068 at every threshold -- thresholding drops edges but
    keeps the node set -- so reporting it would hide the entire cost of Eq. (2)'s threshold.
    """
    nodes, pair_counts, _ = build_cooccurrence(load_scenes())
    cov = {t: len({i for pair, c in pair_counts.items() if c >= t for i in pair})
           for t in (1, 2, 3, 5, 10)}
    cov["total"] = len(nodes)
    return cov


def agg(df: pd.DataFrame, by: list[str], metrics=("test_auc", "test_ap", "test_auc_uniform")) -> pd.DataFrame:
    """Mean and std over seeds. Every reported cell carries its spread: on this graph the
    seed std is ~0.003 AUC, and several sweep axes move the mean by less than that."""
    g = df.groupby(by, dropna=False)
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g.size()
    for extra in ("best_epoch", "epochs_run", "n_params", "wall_clock_s", "effective_rank",
                  "n_edges", "n_nodes", "auc_degq1", "auc_c=1"):
        if extra in df.columns:
            out[extra] = g[extra].mean()
    return out.reset_index()


def fmt(mean: float, std: float, nd: int = 3) -> str:
    if pd.isna(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def latex_table(rows: list[list[str]], header: list[str], caption: str, label: str,
                align: str | None = None, highlight: int | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    out = [
        "\\begin{table}[htbp]", "\\centering", "\\begin{latin}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}", "\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline",
    ]
    for i, r in enumerate(rows):
        prefix = "\\rowcolor{gray!15} " if i == highlight else ""
        out.append(prefix + " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{latin}", "\\end{table}", ""]
    return "\n".join(out)


def table_graph_construction(df: pd.DataFrame) -> str:
    d = df[df.sweep == "stageA"]
    if d.empty:
        return ""
    a = agg(d, ["min_cooc", "edge_mode"]).sort_values(["min_cooc", "edge_mode"])
    cov = node_coverage()
    rows, hl = [], None
    for i, r in enumerate(a.itertuples()):
        if r.min_cooc == 1 and r.edge_mode == DEFAULT_CFG["edge_mode"]:
            hl = i
        n_active = cov[int(r.min_cooc)]
        rows.append([
            int(r.min_cooc), r.edge_mode.replace("_", "\\_"),
            f"{int(r.n_edges)}", f"{n_active}",
            f"{100 * n_active / cov['total']:.1f}\\%",
            fmt(r.test_auc_mean, r.test_auc_std), fmt(r.test_ap_mean, r.test_ap_std),
        ])
    return latex_table(
        rows, ["min\\_cooc", "Edge weighting", "Edges", "Non-isol.\\ nodes", "Coverage",
               "AUC", "AP"],
        "Graph construction: co-occurrence threshold and edge weighting. \"Non-isolated nodes\" "
        "counts object labels retaining at least one edge, out of "
        f"{cov['total']}; coverage is that as a percentage. AUC/AP are held-out "
        "link prediction against degree-matched negatives, mean $\\pm$ std over 5 seeds. The "
        "shipped configuration is shaded.",
        "tab:graph-construction", highlight=hl,
    )


def table_sensitivity(df: pd.DataFrame) -> str:
    d = df[df.sweep.isin(["stageB", "stageC"])]
    if d.empty:
        return ""
    axes = ["epochs", "lr", "temperature", "neg_mode", "neg_ratio", "dropout"]
    rows, hl = [], None
    for axis in axes:
        if axis not in d.columns:
            continue
        sub = d[d.sweep == "stageB"]
        # An OFAT row is one where every other swept axis sits at its base value.
        others = [a for a in axes if a != axis and a in sub.columns]
        base = {a: sub[a].mode().iloc[0] for a in others if not sub[a].mode().empty}
        mask = np.ones(len(sub), dtype=bool)
        for a, v in base.items():
            mask &= (sub[a] == v).values
        s = agg(sub[mask], [axis]).sort_values(axis)
        if len(s) < 2:
            continue
        default = DEFAULT_CFG.get(axis)
        best = s[f"{PRIMARY}_mean"].max()
        for r in s.itertuples():
            v = getattr(r, axis)
            is_def = str(v) == str(default)
            if is_def and hl is None:
                hl = len(rows)
            rows.append([
                axis.replace("_", "\\_"), v,
                fmt(r.test_auc_mean, r.test_auc_std),
                fmt(r.test_ap_mean, r.test_ap_std),
                f"{r.test_auc_uniform_mean:.3f}",
                f"{r.test_auc_mean - best:+.3f}",
            ])
    return latex_table(
        rows, ["Parameter", "Value", "AUC (deg.)", "AP (deg.)", "AUC (unif.)", "$\\Delta$ best"],
        "Encoder hyperparameter sensitivity, one factor at a time from the shipped "
        "configuration. Mean $\\pm$ std over 3 seeds.",
        "tab:encoder-sensitivity", highlight=hl,
    )


def table_backbone(df: pd.DataFrame, baselines: dict | None = None) -> str:
    d = df[df.sweep == "stageD"]
    if d.empty:
        return ""
    b = d[d.backbone.notna()]
    base_layers, base_hidden = DEFAULT_CFG["num_layers"], DEFAULT_CFG["hidden_dim"]
    b = b[(b.num_layers == base_layers) & (b.hidden_dim == base_hidden)]
    a = agg(b, ["backbone"])
    rows = []
    for r in a.itertuples():
        # Divide by epochs actually run, not by best_epoch: early stopping means the run
        # continues past its best evaluation, and using best_epoch inflates fast backbones.
        eps = getattr(r, "epochs_run", None) or r.best_epoch
        rows.append([
            r.backbone.upper(), fmt(r.test_auc_mean, r.test_auc_std),
            fmt(r.test_ap_mean, r.test_ap_std), f"{int(r.n_params)}",
            f"{1000 * r.wall_clock_s / max(eps, 1):.2f}",
        ])
    for name, m in (baselines or {}).items():
        rows.append([
            name.replace("_", "\\_"), f"{m['auc']:.3f}", f"{m['ap']:.3f}", "--", "--",
        ])
    return latex_table(
        rows, ["Model", "AUC (deg.)", "AP (deg.)", "Params", "ms/epoch"],
        "Backbone ablation and non-neural baselines under degree-matched negatives, "
        "mean $\\pm$ std over 5 seeds.",
        "tab:backbone", highlight=0,
    )


def table_final(df: pd.DataFrame) -> str:
    d = df[df.sweep == "stageF"]
    if d.empty:
        return ""
    # Ordered as a cumulative story rather than alphabetically: each row adds one change to
    # the row above, so the reader can attribute the total gain to its parts.
    labels = {
        "default": ("Shipped default (20 epochs)", 0),
        "default_long": ("\\quad + train to convergence", 1),
        "tuned_tau_only": ("\\quad\\quad + $\\tau = 0.2$", 2),
        "tuned_sage": ("\\quad\\quad\\quad + lr, neg.\\ ratio, 1 layer", 3),
        "tuned": ("\\quad\\quad\\quad\\quad + GAT backbone (tuned)", 4),
        "tuned_joint_sage": ("Stage-C joint optimum (SAGE)", 5),
        "tuned_joint_gat": ("Stage-C joint optimum (GAT)", 6),
    }
    a = agg(d, ["tag"], metrics=("test_auc", "test_ap", "test_auc_uniform"))
    a["_o"] = a.tag.map(lambda t: labels.get(t, (t, 99))[1])
    a = a.sort_values("_o")
    rows = []
    for r in a.itertuples():
        rows.append([
            labels.get(r.tag, (r.tag.replace("_", "\\_"), 99))[0],
            fmt(r.test_auc_mean, r.test_auc_std), fmt(r.test_ap_mean, r.test_ap_std),
            f"{r.test_auc_uniform_mean:.3f}",
            f"{getattr(r, 'auc_degq1', float('nan')):.3f}",
            f"{int(r.best_epoch)}",
        ])
    return latex_table(
        rows,
        ["Configuration", "AUC (deg.)", "AP (deg.)", "AUC (unif.)", "AUC low-deg.", "Best epoch"],
        "Shipped versus tuned encoder configuration, 10 seeds each, degree-matched negatives. "
        "Rows 2--5 are cumulative. The final two rows show that coordinate-wise optima do not "
        "compose. Reported as a sensitivity study: the shipped configuration remains the one "
        "used for the recommendation results. ``AUC low-deg.'' is the lowest quartile of "
        "minimum endpoint degree, where the ordering reverses.",
        "tab:final", highlight=0, align="lrrrrr",
    )


def table_stage(df: pd.DataFrame) -> str:
    """The GraphMAE stage ablation -- the training stage that is in the code and not the paper.

    Two blocks. The first reproduces the shipped `remask=True` implementation, whose latent
    re-mask against a linear decoder zeroes the encoder's gradient (F14): stage 2 is provably
    a no-op there, and the table shows it as `s1->s2` agreeing with `s1` to four decimals. The
    second block repairs it and asks the scientific question, where stage 2 turns out to be
    actively harmful. Reporting only the second would hide that the published artifact was
    never affected either way.
    """
    d = df[df.sweep == "stageE"]
    if d.empty:
        return ""
    d = d.copy()
    d["remask"] = d["remask"].astype(str).str.lower().map({"true": True, "false": False})

    def one(mask, label, keys=None):
        sub = d[mask]
        if sub.empty:
            return None
        f = lambda c: sub[c].astype(float)
        return [label, fmt(f("test_auc").mean(), f("test_auc").std()),
                fmt(f("test_ap").mean(), f("test_ap").std()),
                f'{f("auc_degq1").mean():.3f}', f'{f("degree_bias_rho").mean():.3f}',
                f'{f("effective_rank").mean():.1f}']

    base = d.tag.fillna("") == ""
    plain = base & d.mask_rate.isna() & d.mae_epochs.isna() & d.mae_alpha.isna()
    rows = [
        one(plain & (d.stage == "s1"), "Stage 1 only (contrastive)"),
        one((d.tag == "s2_shipped"), "\\quad Stage 2 only, as shipped"),
        one(plain & (d.stage == "s1->s2") & d.remask, "\\quad Stage 1 $\\to$ 2, as shipped"),
        one(plain & (d.stage == "s2") & ~d.remask, "\\quad Stage 2 only, gradient repaired"),
        one(plain & (d.stage == "s1->s2") & ~d.remask, "\\quad Stage 1 $\\to$ 2, gradient repaired"),
    ]
    for axis, tex in (("mask_rate", "mask rate"), ("mae_epochs", "stage-2 epochs"),
                      ("mae_alpha", r"SCE $\alpha$")):
        for v in sorted(d[axis].dropna().unique()):
            v_lab = f"{v:g}" if float(v) < 10 else f"{int(v)}"
            rows.append(one(~d.remask & (d[axis] == v) & (d.tag.fillna("") == ""),
                            f"\\quad\\quad {tex} = {v_lab}"))
    rows.append(one(d.tag == "tuned_s1s2", "Tuned config $\\to$ stage 2, repaired"))
    rows = [r for r in rows if r]
    return latex_table(
        rows,
        ["Training recipe", "AUC (deg.)", "AP (deg.)", "AUC low-deg.", r"$\rho_{\deg}$", "Eff.\\ rank"],
        "GraphMAE stage ablation, 5 seeds each. As shipped, the latent re-mask in front of a "
        "linear decoder leaves the encoder with zero gradient, so stage 2 provably cannot "
        "change the model and every stage-2 row reproduces stage 1 exactly. With the gradient "
        "repaired, masked reconstruction of each node's own text embedding is consistently "
        "harmful to co-occurrence link prediction --- it pulls the representation back toward "
        "the text space the object modality is meant to complement --- and costs the tuned "
        "configuration its entire gain. We therefore report the encoder as single-stage.",
        "tab:stage-ablation", highlight=0, align="lrrrrr",
    )


def figures(df: pd.DataFrame, out_dir: Path, sweeps_dir: Path) -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation AUC vs epoch, with the shipped 20-epoch budget marked.
    # One curve per arm, averaged over that arm's seeds. Averaging across configurations would
    # be meaningless -- the arms differ in learning rate, depth and backbone, so their curves
    # are not comparable point by point. Only stage F arms are plotted, since only there does a
    # tag identify a fixed configuration run over many seeds.
    curves: dict[str, list] = {}
    for p in sorted(sweeps_dir.glob("*/*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        tag = j.get("cfg", {}).get("tag", "")
        if tag in ("default_long", "tuned") and len(j.get("history", [])) > 5:
            curves.setdefault(tag, []).append(j["history"])
    if curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (tag, cs) in enumerate(sorted(curves.items())):
            L = min(len(c) for c in cs)
            ep = [h["epoch"] for h in cs[0][:L]]
            M = np.array([[h["val_auc"] for h in c[:L]] for c in cs])
            label = {"default_long": "shipped config, trained to convergence",
                     "tuned": "tuned config"}.get(tag, tag)
            ax.plot(ep, M.mean(0), color=f"C{i}", lw=2, label=f"{label} ({len(cs)} seeds)")
            ax.fill_between(ep, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=f"C{i}", alpha=0.2)
        ax.axvline(DEFAULT_CFG["epochs"], color="C3", ls="--",
                   label=f"shipped budget ({DEFAULT_CFG['epochs']} epochs)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC"); ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "val_auc_vs_epoch.pdf"); plt.close(fig)

    # 2. AUC and node coverage vs min_cooc, twin axes.
    d = df[(df.sweep == "stageA") & (df.edge_mode == DEFAULT_CFG["edge_mode"])]
    if not d.empty:
        a = agg(d, ["min_cooc"]).sort_values("min_cooc")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(a.min_cooc, a.test_auc_mean, yerr=a.test_auc_std, marker="o", color="C0")
        ax.set_xlabel("min\\_cooc threshold"); ax.set_ylabel("Test AUC", color="C0")
        ax2 = ax.twinx()
        ax2.plot(a.min_cooc, 100 * a.n_nodes / a.n_nodes.max(), marker="s", color="C1", ls="--")
        ax2.set_ylabel("Node coverage (%)", color="C1")
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / "auc_vs_min_cooc.pdf"); plt.close(fig)

    # 3. AUC by co-occurrence stratum.
    cols = [c for c in df.columns if c.startswith("auc_c")]
    if cols and not df[df.sweep == "stageF"].empty:
        d = df[df.sweep == "stageF"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for tag, sub in d.groupby("tag"):
            ax.bar(np.arange(len(cols)) + (0.35 if tag == "tuned" else 0), sub[cols].mean(),
                   width=0.35, yerr=sub[cols].std(), label=tag, capsize=3)
        ax.set_xticks(np.arange(len(cols)) + 0.175)
        ax.set_xticklabels([c.replace("auc_", "") for c in cols])
        ax.set_xlabel("Co-occurrence count stratum"); ax.set_ylabel("Test AUC")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(out_dir / "auc_by_stratum.pdf"); plt.close(fig)

    print(f"Figures -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path(DEFAULT_CFG["log_dir"]) / "results.csv")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "tables")
    p.add_argument("--tables", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--baselines-json", type=Path, default=None)
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"no results at {args.csv} -- run scripts/sweep_objectgraph.py first")
    df = pd.read_csv(args.csv)
    print(f"{len(df)} runs across {sorted(df.sweep.unique())}")
    args.out.mkdir(parents=True, exist_ok=True)

    baselines = None
    if args.baselines_json and args.baselines_json.exists():
        import json
        baselines = json.loads(args.baselines_json.read_text()).get("baselines")

    if args.tables or not args.figures:
        for name, tex in (
            ("graph_construction", table_graph_construction(df)),
            ("encoder_sensitivity", table_sensitivity(df)),
            ("backbone", table_backbone(df, baselines)),
            ("stage_ablation", table_stage(df)),
            ("final", table_final(df)),
        ):
            if tex:
                (args.out / f"{name}.tex").write_text(tex)
                print(f"  wrote {args.out / f'{name}.tex'}")
    if args.figures:
        figures(df, args.out, args.csv.parent)


if __name__ == "__main__":
    main()
