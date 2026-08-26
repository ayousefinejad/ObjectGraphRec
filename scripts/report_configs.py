#!/usr/bin/env python3
"""The five-configuration evaluation table: Recall@20, Precision@20, NDCG@20.

    python scripts/report_configs.py            # plain text
    python scripts/report_configs.py --md       # markdown
    python scripts/report_configs.py --tex      # also write docs/tables/configs.tex

The five configurations were named by the modalities and the encoder behind them, not by the
internal variant names, so this script owns that mapping in one place. Rows come from two files
because they were collected under two schemas: the encoder study wrote downstream.csv keyed on
(variant, seed), the graph/fusion study wrote fusion_arms.csv keyed on (variant, arm, seed).
Nothing is recomputed here -- a configuration with no rows prints as missing rather than as zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "data" / "lattice-runs" / "downstream.csv"
ARMS = ROOT / "data" / "lattice-runs" / "fusion_arms.csv"
TEX_OUT = ROOT / "docs" / "tables" / "configs.tex"

# Smallest credible effect: 2 sigma on the encoder study's 0.0005 seed-to-seed std at n=3.
# Anything inside this band is reported flat, never as a gain.
RESOLUTION = 0.0012

METRICS = ["recall@20", "precision@20", "ndcg@20"]

# (key, label, source, selector, note). `source` is "arms" for fusion_arms.csv rows, which are
# selected by (variant, arm); "downstream" rows are selected by variant alone.
CONFIGS = [
    ("1", "text + image (no object graph)",
     "arms", ("default_fixed", "noobj"),
     "official LATTICE top-10 kNN; the object modality is masked to weight exactly 0, so the "
     "encoder behind it is irrelevant"),
    ("2", "+ object graph, SAGE / NYU Depth",
     "downstream", ("nyu_default_fixed", None),
     "GraphSAGE 2L, 20 ep, nyu-depth.json (579 scenes), intrinsic AUC 0.795"),
    ("3", "+ object graph, SAGE / MIT + NYU Depth",
     "downstream", ("default_fixed", None),
     "same recipe as config 2, scenes.json (3213 scenes), intrinsic AUC 0.757 -- config 2 vs 3 "
     "is a clean dataset A/B"),
    ("4", "+ object graph, GAT / MIT + NYU Depth",
     "downstream", ("tuned", None),
     "GAT 1L 4 heads, 1000 ep, lr 3e-4, tau 0.2, intrinsic AUC 0.825; note this changes the "
     "whole recipe, not only the backbone"),
    ("5", "config 4 + object kNN by threshold 0.8",
     "arms", ("tuned", "thr08"),
     "object edges are every pair with cos >= 0.8 and no k at all: mean out-degree 269 against "
     "10, image and text keep the official top-10"),
]

# Rows worth printing but not part of the headline five.
EXTRA = [
    ("3b", "+ object graph, SAGE / MIT + NYU, 1000 ep",
     "downstream", ("converged", None),
     "budget-matched SAGE row for config 4, intrinsic AUC 0.789"),
    ("5b", "config 5 without forced self-loops",
     "arms", ("tuned", "thr08_nosf"),
     "control: thresholding self-loops 100% of rows where the published graph self-loops 22.1%"),
]

REFERENCE = "3"          # deltas are read against the SAGE / MIT+NYU row by default

# Rows whose natural comparison is not REFERENCE but their own parent step: config 5 isolates the
# threshold-kNN change, so it must be read against config 4 (GAT, official kNN), not against the
# SAGE baseline -- comparing to config 3 would fold the SAGE->GAT swap into the kNN-mode delta.
# 5b (no self-loops) is a control for 5, so it reads against 5 for the same reason.
PARENT = {
    "5": "4",
    "5b": "5",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DOWNSTREAM.exists():
        sys.exit(f"missing {DOWNSTREAM}")
    down = pd.read_csv(DOWNSTREAM)
    arms = pd.read_csv(ARMS) if ARMS.exists() else pd.DataFrame(columns=["variant", "arm", "seed"])
    return down, arms


def rows_for(down: pd.DataFrame, arms: pd.DataFrame, source: str, sel) -> pd.DataFrame:
    variant, arm = sel
    if source == "arms":
        return arms[(arms.variant == variant) & (arms.arm == arm)]
    return down[down.variant == variant]


def build(down: pd.DataFrame, arms: pd.DataFrame, include_extra: bool) -> pd.DataFrame:
    out = []
    for key, label, source, sel, note in CONFIGS + (EXTRA if include_extra else []):
        r = rows_for(down, arms, source, sel)
        row = {"#": key, "configuration": label, "n": len(r)}
        for m in METRICS:
            if r.empty:
                row[m] = "—"
            elif len(r) == 1:
                row[m] = f"{r[m].iloc[0]:.5f}"
            else:
                row[m] = f"{r[m].mean():.5f} ± {r[m].std(ddof=1):.5f}"
        row["_r20"] = float(r["recall@20"].mean()) if not r.empty else float("nan")
        row["_seeds"] = ",".join(str(s) for s in sorted(r["seed"])) if not r.empty else "—"
        row["_note"] = note
        out.append(row)
    df = pd.DataFrame(out)

    r20 = dict(zip(df["#"], df["_r20"]))

    def parent_of(k: str) -> str:
        return PARENT.get(k, REFERENCE)

    def base_of(k: str) -> float:
        p = r20.get(parent_of(k), float("nan"))
        return float(p) if pd.notna(p) else float("nan")

    df["vs"] = ["—" if k == REFERENCE else parent_of(k) for k in df["#"]]
    df["ΔR@20"] = [
        "ref" if k == REFERENCE else
        ("—" if pd.isna(v) or pd.isna(base_of(k)) else
         f"{v - base_of(k):+.5f} ({100 * (v - base_of(k)) / base_of(k):+.1f}%)")
        for k, v in zip(df["#"], df["_r20"])
    ]
    df["verdict"] = [
        "—" if k == REFERENCE else
        ("no runs" if pd.isna(v) or pd.isna(base_of(k)) else
         "flat" if abs(v - base_of(k)) < RESOLUTION else
         "better" if v > base_of(k) else "worse")
        for k, v in zip(df["#"], df["_r20"])
    ]
    return df


def to_tex(df: pd.DataFrame) -> str:
    head = (r"\begin{tabular}{clccc}" "\n" r"\toprule" "\n"
            r"& Configuration & Recall@20 & Precision@20 & NDCG@20 \\" "\n" r"\midrule" "\n")
    body = ""
    for _, r in df.iterrows():
        cells = [str(r[m]).replace("±", r"$\pm$") for m in METRICS]
        body += f"{r['#']} & {r['configuration']} & " + " & ".join(cells) + r" \\" + "\n"
    return head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md", action="store_true", help="markdown instead of plain text")
    p.add_argument("--tex", action="store_true", help=f"also write {TEX_OUT.relative_to(ROOT)}")
    p.add_argument("--no-extra", action="store_true", help="headline five rows only")
    a = p.parse_args()

    down, arms = load()
    df = build(down, arms, include_extra=not a.no_extra)

    show = df[["#", "configuration", "n"] + METRICS + ["vs", "ΔR@20", "verdict"]]
    print(f"\nFive-configuration evaluation (mean ± std over seeds), default reference = config "
          f"{REFERENCE}; rows in PARENT compare to their own parent step instead (see 'vs' column)")
    print(f"resolution = ±{RESOLUTION:.4f} on Recall@20; smaller differences are flat, not gains\n")
    print(show.to_markdown(index=False) if a.md else show.to_string(index=False))

    print("\nseeds and provenance:")
    for _, r in df.iterrows():
        print(f"  {r['#']:>2}  seeds {r['_seeds']:<8} {r['_note']}")

    missing = df[df.n == 0]
    if not missing.empty:
        print("\nno runs yet for: " + ", ".join(f"config {k}" for k in missing["#"]))

    if a.tex:
        TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
        TEX_OUT.write_text(to_tex(df))
        print(f"\nwrote {TEX_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
