#!/usr/bin/env python3
"""Aggregate the downstream LATTICE runs into a table and a figure.

    python scripts/report_lattice.py --out docs/notion/images

Reads only `data/lattice-runs/downstream.csv` plus each variant's `provenance.json` (for the
intrinsic AUC that arm was selected on). Writes a LaTeX table, a markdown table for the report,
and one two-panel figure. Nothing under `data/home_v2-2/` is touched.

The four arms are ordered so the two effects separate:

    shipped -> default_fixed   pipeline repair alone (same hyperparameters, 293 -> 531 vectors)
    default_fixed -> tuned     training recipe alone (both on the repaired pipeline)

Reporting a single number per arm would confound them.
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

# Same palette as the intrinsic figures, so an arm keeps its colour across the whole report.
from scripts.figures_objectgraph import C_BASE, C_LONG, C_SHIPPED, C_TUNED

RUNS = ROOT / "data" / "lattice-runs"
CSV = RUNS / "downstream.csv"

# Order is the causal chain, not alphabetical. Intrinsic AUC is the number the study selected on;
# 'shipped' has no provenance.json (it is the frozen published artifact), so its AUC is the
# published one, re-measured against the same fixed split.
ORDER = ["shipped", "default_fixed", "converged", "tuned"]
LABEL = {"shipped": "Shipped (20 ep, published)",
         "default_fixed": "Pipeline fix only (20 ep)",
         "converged": "Converged (1000 ep, early-stopped)",
         "tuned": "Tuned (GAT, $\\tau$=0.2, lr 3e-4)"}
COLOR = {"shipped": C_SHIPPED, "default_fixed": C_BASE,
         "converged": C_LONG, "tuned": C_TUNED}
FALLBACK_AUC = {"shipped": 0.752}
UNIQUE_VEC = {"shipped": 293, "default_fixed": 531, "converged": 531, "tuned": 531}

METRICS = ["recall@10", "recall@20", "precision@20", "hit@20", "ndcg@10", "ndcg@20"]


def intrinsic_auc(variant: str) -> float | None:
    p = RUNS / variant / "provenance.json"
    if p.exists():
        return json.loads(p.read_text())["intrinsic"]["test_auc"]
    return FALLBACK_AUC.get(variant)


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"{CSV} not found -- run scripts/run_lattice_study.py first")
    df = pd.read_csv(CSV)
    df["variant"] = pd.Categorical(df.variant, categories=ORDER, ordered=True)
    return df.sort_values(["variant", "seed"])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant", observed=True)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out["best_epoch"] = g["best_epoch"].mean()
    out = out.reset_index()
    out["auc"] = [intrinsic_auc(v) for v in out.variant]
    out["uniq"] = [UNIQUE_VEC.get(v) for v in out.variant]
    return out


def fmt(m: float, s: float, nd: int = 4, tex: bool = True) -> str:
    if pd.isna(s):
        return f"{m:.{nd}f}"
    pm = " $\\pm$ " if tex else " ± "
    return f"{m:.{nd}f}{pm}{s:.{nd}f}"


def delta(row: pd.Series, base: pd.Series, metric: str) -> str:
    """Relative change vs the shipped arm, which is how the paper would quote it."""
    d = 100 * (row[f"{metric}_mean"] - base[f"{metric}_mean"]) / base[f"{metric}_mean"]
    return f"{d:+.1f}\\%"


def table_tex(s: pd.DataFrame) -> str:
    base = s[s.variant == "shipped"].iloc[0]
    rows = []
    for i in range(len(s)):
        row = s.iloc[i]
        rows.append([
            LABEL[row.variant], f"{row.auc:.3f}" if row.auc else "--", row.uniq,
            fmt(row["recall@20_mean"], row["recall@20_std"]),
            fmt(row["ndcg@20_mean"], row["ndcg@20_std"]),
            delta(row, base, "recall@20"), delta(row, base, "ndcg@20"),
        ])
    header = ["Object-graph encoder", "Intr.\\ AUC", "Uniq.\\ vec.",
              "Recall@20", "NDCG@20", "$\\Delta$R@20", "$\\Delta$N@20"]
    n = int(s.n_seeds.max())
    caption = (
        "Downstream recommendation performance of LATTICE under four object-graph encoders. "
        "\"Intr.\\ AUC\" is held-out co-occurrence link prediction against degree-matched "
        "negatives (the intrinsic study's selection metric); \"Uniq.\\ vec.\" is the number of "
        "distinct object vectors resolved across the 14{,}503 items. Downstream numbers are the "
        f"test scores at the best-validation epoch, mean $\\pm$ std over {n} seeds; $\\Delta$ is "
        "relative to the shipped encoder. Every arm rebuilds its own item-item graph cache, so "
        "no arm is scored through another's features."
    )
    align = "l" + "r" * (len(header) - 1)
    out = [
        "\\begin{table}[htbp]", "\\centering", "\\begin{latin}",
        f"\\caption{{{caption}}}", "\\label{tab:downstream}",
        f"\\begin{{tabular}}{{{align}}}", "\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline",
    ]
    for i, r in enumerate(rows):
        prefix = "\\rowcolor{gray!15} " if s.variant.iloc[i] == "shipped" else ""
        out.append(prefix + " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{latin}", "\\end{table}", ""]
    return "\n".join(out)


def table_md(s: pd.DataFrame) -> str:
    base = s[s.variant == "shipped"].iloc[0]
    head = ("| Encoder | Intrinsic AUC | Unique vectors | Recall@10 | Recall@20 | NDCG@10 "
            "| NDCG@20 | ΔR@20 |")
    lines = [head, "|" + "---|" * 8]
    for i in range(len(s)):
        row = s.iloc[i]
        lines.append("| " + " | ".join([
            LABEL[row.variant].replace("$\\tau$", "τ"),
            f"{row.auc:.3f}" if row.auc else "—", str(row.uniq),
            fmt(row["recall@10_mean"], row["recall@10_std"], tex=False),
            fmt(row["recall@20_mean"], row["recall@20_std"], tex=False),
            fmt(row["ndcg@10_mean"], row["ndcg@10_std"], tex=False),
            fmt(row["ndcg@20_mean"], row["ndcg@20_std"], tex=False),
            delta(row, base, "recall@20").replace("\\%", "%"),
        ]) + " |")
    return "\n".join(lines)


def figure(df: pd.DataFrame, s: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) per-arm downstream metrics, with the individual seeds overplotted -- with n=3 the
    # reader should see the raw spread, not just a std whisker.
    x = np.arange(len(s))
    w = 0.36
    for k, (metric, off) in enumerate({"recall@20": -w / 2, "ndcg@20": w / 2}.items()):
        ax1.bar(x + off, s[f"{metric}_mean"], w, yerr=s[f"{metric}_std"], capsize=3,
                color=[COLOR[v] for v in s.variant], alpha=0.85 if k == 0 else 0.45,
                edgecolor="white", label=metric.upper().replace("@", "@"))
        for i, v in enumerate(s.variant):
            pts = df[df.variant == v][metric]
            ax1.plot(np.full(len(pts), x[i] + off), pts, ".", color="k", ms=3, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace("_", "\n") for v in s.variant], fontsize=9)
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("(a) Downstream LATTICE performance\n(bars: mean over seeds; dots: each seed)",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    # (b) does the intrinsic metric predict the downstream one? With four points this is a
    # picture, not a regression -- annotate it as such rather than quoting an r.
    for i in range(len(s)):
        row = s.iloc[i]
        ax2.errorbar(row.auc, row["recall@20_mean"], yerr=row["recall@20_std"], fmt="o", ms=9,
                     color=COLOR[row.variant], capsize=3, elinewidth=1.2)
        ax2.annotate(row.variant, (row.auc, row["recall@20_mean"]), textcoords="offset points",
                     xytext=(7, 5), fontsize=8.5, color=COLOR[row.variant])
    ax2.set_xlabel("Intrinsic test AUC (degree-matched negatives)")
    ax2.set_ylabel("Downstream Recall@20")
    ax2.grid(alpha=0.25)
    n_arms = len(s)
    if n_arms >= 3:
        rho = s[["auc", "recall@20_mean"]].corr(method="spearman").iloc[0, 1]
        rho_txt = f"Spearman ρ={rho:.2f}"
    else:
        rho_txt = "too few arms for a correlation"
    ax2.set_title(f"(b) Intrinsic vs downstream (n={n_arms} arms so far, {rho_txt})\n"
                  "descriptive only — a handful of points cannot support a fit", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig13_downstream_lattice.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig13_downstream_lattice.png / .pdf / .svg -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "notion" / "images")
    p.add_argument("--tex", type=Path, default=ROOT / "docs" / "tables")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.tex.mkdir(parents=True, exist_ok=True)

    df = load()
    s = summarise(df)
    print(f"{len(df)} runs across {len(s)} arms "
          f"({', '.join(f'{v}:{n}' for v, n in zip(s.variant, s.n_seeds))})")
    print()
    print(table_md(s))
    print()

    (args.tex / "downstream.tex").write_text(table_tex(s))
    (args.out.parent / "downstream_table.md").write_text(table_md(s) + "\n")
    figure(df, s, args.out)
    print(f"  {args.tex / 'downstream.tex'}")


if __name__ == "__main__":
    main()
