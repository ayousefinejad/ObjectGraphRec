#!/usr/bin/env python3
"""The LATTICE hyperparameter sweep table, ranked on validation Recall@20.

    python scripts/report_tuning.py            # plain text
    python scripts/report_tuning.py --tex      # also write docs/tables/tuning.tex

The control is the published Amazon recipe (mf, lr 5e-4, weight_size [64,64]) on the same noobj
arm. Its rows live in fusion_arms.csv, which has no validation columns -- the encoder study never
needed them -- so its val statistics are recovered from its logs with the same parse_val() the
sweep uses. Ranking a sweep cell's val score against a control's *test* score would compare two
different splits, which is how a tuning study talks itself into a win.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_lattice_study import ROOT, parse  # noqa: E402
from scripts.tune_lattice import OUT as TUNING, parse_val  # noqa: E402

LOGS = ROOT / "data" / "lattice-runs" / "logs"
TEX_OUT = ROOT / "docs" / "tables" / "tuning.tex"

# Single-seed screening cannot resolve less than this on val Recall@20 (~5% relative). Anything
# smaller is a tie, not a win -- reported as "flat", never as a gain.
THRESHOLD = 0.002

RANK_BY = "val_recall@20"          # must be a val_ column; asserted in main()

CONTROL_LOGS = ["default_fixed_noobj_seed0", "default_fixed_noobj_seed1",
                "default_fixed_noobj_seed2"]


def control_rows() -> list[dict]:
    """The published recipe's val + test metrics, recovered from the existing noobj logs."""
    out = []
    for name in CONTROL_LOGS:
        p = LOGS / f"{name}.log"
        if not p.exists():
            continue
        log = p.read_text()
        test, val = parse(log), parse_val(log)
        if test is None or val is None:
            continue
        out.append({"cell": "control (published)", "seed": int(name[-1]),
                    "cf_model": "mf", "weight_size": "[64,64]", "lr": "0.0005",
                    "epoch_capped": 0} | test | val)
    return out


def recipe(r: pd.Series) -> str:
    """What actually ran, not what was typed on the command line.

    --regs [1e-5,1e-5,1e-2] is two-thirds fiction (main.py:38 reads only regs[0]), and
    --mess_dropout is never read unless cf_model is ngcf (dropout_list is built under that branch
    alone). Transcribing the command line would put a model in the writeup that did not run.
    """
    parts = [f"cf_model={r['cf_model']}", f"lr={r['lr']}", "weight_decay=1e-5",
             f"embed_size=64", f"topk=10", f"lambda_coeff=0.9", f"n_layers=1"]
    if r["cf_model"] == "mf":
        parts.insert(2, "(no user-item propagation)")
    else:
        parts.insert(2, f"cf_layers={len(eval(r['weight_size']))}")
        if r["cf_model"] == "ngcf":
            parts.append(f"mess_dropout={r.get('mess_dropout', '[0.1,0.1,0.1]')}")
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tex", action="store_true", help=f"also write {TEX_OUT.relative_to(ROOT)}")
    a = p.parse_args()

    assert RANK_BY.startswith("val"), "the sweep must never be ranked on a test metric"

    rows = control_rows()
    if TUNING.exists():
        with TUNING.open() as f:
            for r in csv.DictReader(f):
                rows.append({k: (float(v) if k.startswith(("val_", "recall", "precision", "hit",
                                                           "ndcg")) and v else v)
                             for k, v in r.items()} | {"seed": int(r["seed"]),
                                                       "epoch_capped": int(r["epoch_capped"])})
    if not rows:
        sys.exit(f"no rows: neither {TUNING} nor the control logs produced anything")

    df = pd.DataFrame(rows)
    base = df[df.cell.str.startswith("control")]
    b = float(base[RANK_BY].mean()) if not base.empty else float("nan")
    b_tail = float(base["val_tail5_recall@20"].mean()) if not base.empty else float("nan")

    # One row per (cell, seed); seed-0 screening first, then any confirmation seeds.
    df = df.sort_values([RANK_BY], ascending=False)
    df["Δval"] = df[RANK_BY] - b
    df["verdict"] = [
        "ref" if c.startswith("control") else
        "CAPPED" if cap else
        "flat" if abs(d) < THRESHOLD else
        # A cell must lead on the tail-5 mean too. The primary statistic is a max over 27-40 evals
        # and carries a measured +0.0003..+0.0007 best-of-N inflation; the tail mean does not.
        ("better" if d > 0 and t > b_tail else "noise (max only)" if d > 0 else "worse")
        for c, cap, d, t in zip(df.cell, df.epoch_capped, df["Δval"], df["val_tail5_recall@20"])
    ]

    # max - tail5. A cell that spikes once and decays has a large spread; a cell that genuinely
    # sits higher has a small one. lr 2e-3 tops the raw max while carrying 14x the spread of the
    # default lr, which is the whole reason the max alone must not pick the winner.
    df["spread"] = df[RANK_BY] - df["val_tail5_recall@20"]

    show = df[["cell", "seed", "cf_model", "weight_size", "lr", RANK_BY,
               "val_tail5_recall@20", "spread", "val_ndcg@20", "recall@20", "ndcg@20",
               "best_epoch", "Δval", "verdict"]].copy()
    for c in (RANK_BY, "val_tail5_recall@20", "spread", "val_ndcg@20", "recall@20", "ndcg@20"):
        show[c] = show[c].map(lambda v: f"{v:.5f}")
    show["Δval"] = df["Δval"].map(lambda v: f"{v:+.5f}")

    print(f"\nLATTICE hyperparameter sweep on home_v2 (noobj arm: image + text, official LATTICE)")
    print(f"ranked on {RANK_BY}; test columns are reported but never used for ranking")
    print(f"threshold = {THRESHOLD} val Recall@20 (~5% rel); smaller is a tie, not a gain\n")
    print(show.to_string(index=False))

    winners = df[df.verdict == "better"]
    print()
    if winners.empty:
        print(f"No cell beat the published recipe by more than {THRESHOLD} on validation Recall@20.")
        print(f"Published recipe stands: {recipe(base.iloc[0]) if not base.empty else 'n/a'}")
    else:
        # Cells within THRESHOLD of the top val score are tied on the primary statistic -- the
        # sweep cannot separate them at one seed. Break that tie on the tail-5 mean rather than on
        # the max, which is inflated by best-of-N and rewards an unstable curve that happened to
        # spike into a single lucky evaluation.
        top = float(winners[RANK_BY].max())
        tied = winners[winners[RANK_BY] > top - THRESHOLD]
        w = tied.sort_values("val_tail5_recall@20", ascending=False).iloc[0]
        if len(tied) > 1:
            print(f"{len(tied)} cells tie within {THRESHOLD} of the top val score "
                  f"({', '.join(tied.cell)}); broken on the tail-5 mean.")
        print(f"Best: {w['cell']} (seed {w['seed']}), {w['Δval']:+.5f} val R@20 vs control "
              f"({100 * w['Δval'] / b:+.1f}%), test R@20 {w['recall@20']:.5f}")
        print(f"Recipe as run: {recipe(w)}")

    # Aggregate over seeds. Only cells actually confirmed on more than one seed get an aggregate
    # row -- a single-seed cell printed as "mean" would invite the reader to compare a 1-run number
    # against the control's 3-run one as if the uncertainty were the same.
    multi = df.groupby("cell").filter(lambda g: len(g) > 1)
    if not multi.empty:
        print("\nconfirmed cells, aggregated over seeds (control is 3 seeds):")
        agg = multi.groupby("cell").agg(
            n=("seed", "count"),
            val=(RANK_BY, "mean"), val_sd=(RANK_BY, "std"),
            test=("recall@20", "mean"), test_sd=("recall@20", "std"),
            ndcg=("ndcg@20", "mean"),
        ).sort_values("val", ascending=False)
        ctrl_test = float(base["recall@20"].mean())
        for cell, r in agg.iterrows():
            tag = "  (ref)" if cell.startswith("control") else \
                  f"  test {100 * (r.test - ctrl_test) / ctrl_test:+.1f}% vs control"
            print(f"  {cell:<22} n={int(r.n)}  val {r.val:.5f} ± {r.val_sd:.5f}   "
                  f"test {r.test:.5f} ± {r.test_sd:.5f}   ndcg {r.ndcg:.5f}{tag}")

    capped = df[df.epoch_capped == 1]
    if not capped.empty:
        print("\nepoch-capped (score is a lower bound, cannot be declared a winner): "
              + ", ".join(f"{r.cell}/seed{r.seed}" for r in capped.itertuples()))

    if a.tex:
        TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
        head = (r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
                r"Configuration & CF model & Val R@20 & Test R@20 & Test NDCG@20 \\" "\n"
                r"\midrule" "\n")
        body = "".join(
            f"{r['cell']} & {r['cf_model']} & {r[RANK_BY]:.5f} & {r['recall@20']:.5f} & "
            f"{r['ndcg@20']:.5f} " + r"\\" + "\n" for _, r in df.iterrows())
        TEX_OUT.write_text(head + body + r"\bottomrule" "\n" r"\end{tabular}" "\n")
        print(f"\nwrote {TEX_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
