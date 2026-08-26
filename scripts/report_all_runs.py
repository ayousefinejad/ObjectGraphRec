#!/usr/bin/env python3
"""Inventory of every experiment in this project, in one table.

    python scripts/report_all_runs.py            # inventory + all downstream runs
    python scripts/report_all_runs.py --encoder  # also expand the encoder sweep stages

Two studies live here and they are not comparable, so they are never merged into one ranking:

  * the **encoder** study (data/graph-embeddings/sweeps/results.csv) optimises an *intrinsic*
    link-prediction AUC on the object co-occurrence graph. 447 runs, minutes each.
  * the **downstream** study (data/lattice-runs/*.csv) measures LATTICE recommendation metrics on
    home_v2-2. 41 runs, ~25 min each.

A high intrinsic AUC has never been shown to buy downstream Recall@20 in this project -- config 2
vs 3 is the clean test and it went the wrong way -- so the two tables are reported side by side and
the reader is left to draw that conclusion, rather than a single sorted list implying a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "lattice-runs"
SWEEPS = ROOT / "data" / "graph-embeddings" / "sweeps" / "results.csv"

# What each downstream CSV was collected for. The three files have different schemas because they
# were written by different drivers at different times; nothing is recomputed or back-filled here.
STUDIES = {
    "downstream.csv": "encoder variants, object graph on, official kNN",
    "fusion_arms.csv": "graph/fusion ablations at fixed encoder",
    "tuning.csv": "LATTICE hyperparameter sweep, image+text only",
}


def load(name: str) -> pd.DataFrame:
    p = RUNS / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["study"] = name.removesuffix(".csv")
    return df


def downstream_table() -> pd.DataFrame:
    """One row per LATTICE run, across all three schemas, with a common set of columns."""
    rows = []
    for name in STUDIES:
        df = load(name)
        for _, r in df.iterrows():
            # The identity of a run is spelled differently in each file: (variant), (variant, arm),
            # or (cell). Collapse to one label so the runs can sit in a single table.
            if name == "tuning.csv":
                label = r["cell"]
                cfg = f"{r['cf_model']} {r['weight_size']} lr={r['lr']}"
            elif name == "fusion_arms.csv":
                label = f"{r['variant']}/{r['arm']}"
                cfg = "mf [64,64] lr=0.0005"
            else:
                label = r["variant"]
                cfg = "mf [64,64] lr=0.0005"
            rows.append({
                "study": r["study"], "run": label, "seed": int(r["seed"]), "recipe": cfg,
                "R@10": r["recall@10"], "R@20": r["recall@20"],
                "N@10": r["ndcg@10"], "N@20": r["ndcg@20"],
                "P@20": r["precision@20"], "hit@20": r["hit@20"],
                "val_R@20": r.get("val_recall@20", float("nan")),
                "best_ep": int(r["best_epoch"]), "sec": r["wall_clock_s"],
            })
    return pd.DataFrame(rows)


def encoder_table() -> pd.DataFrame:
    """Encoder sweeps rolled up by stage. 447 individual runs is noise, not a table."""
    if not SWEEPS.exists():
        return pd.DataFrame()
    d = pd.read_csv(SWEEPS)
    axes = ["backbone", "num_layers", "hidden_dim", "lr", "temperature", "neg_mode",
            "edge_mode", "dropout", "min_cooc", "neg_ratio", "mask_rate"]
    out = []
    for s, g in d.groupby("sweep"):
        varied = [c for c in axes if c in g and g[c].nunique() > 1]
        out.append({
            "stage": s, "runs": len(g), "seeds": g.seed.nunique(),
            "backbones": ",".join(sorted(g.backbone.unique())),
            "varied": ", ".join(varied) or "—",
            "best test AUC": f"{g.test_auc.max():.4f}",
            "median": f"{g.test_auc.median():.4f}",
            "hours": f"{g.wall_clock_s.sum() / 3600:.2f}",
        })
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", action="store_true", help="expand encoder sweep stages")
    p.add_argument("--md", action="store_true", help="markdown instead of plain text")
    a = p.parse_args()

    down = downstream_table()
    if down.empty:
        sys.exit(f"no downstream runs found under {RUNS}")

    fmt = (lambda d: d.to_markdown(index=False)) if a.md else (lambda d: d.to_string(index=False))

    enc = encoder_table()
    n_enc = int(enc.runs.sum()) if not enc.empty else 0
    print(f"\nALL EXPERIMENTS: {n_enc} encoder runs + {len(down)} LATTICE runs = {n_enc + len(down)}")
    print(f"downstream GPU time: {down.sec.sum() / 3600:.1f} h\n")

    show = down.copy()
    for c in ("R@10", "R@20", "N@10", "N@20", "P@20", "hit@20", "val_R@20"):
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.5f}")
    show["sec"] = show.sec.map(lambda v: f"{v:.0f}")
    print("DOWNSTREAM -- LATTICE on home_v2-2, one row per run")
    print(fmt(show))

    print("\nstudies:")
    for k, v in STUDIES.items():
        print(f"  {k:<18} {v}")

    if not enc.empty:
        print(f"\nENCODER -- intrinsic link-prediction AUC on the object graph, by sweep stage")
        print(fmt(enc))
        if a.encoder:
            d = pd.read_csv(SWEEPS)
            print("\nper-stage best configuration:")
            for s, g in d.groupby("sweep"):
                b = g.loc[g.test_auc.idxmax()]
                print(f"  {s}: AUC {b.test_auc:.4f}  {b.backbone} {int(b.num_layers)}L "
                      f"h={int(b.hidden_dim)} lr={b.lr} tau={b.temperature} seed={int(b.seed)}")
        print("\nIntrinsic AUC is not a downstream proxy: config 2 (AUC 0.795) and config 3 "
              "(AUC 0.757)\nare flat against each other downstream, and the best encoder "
              "(config 4, AUC 0.825) is flat too.")


if __name__ == "__main__":
    main()
