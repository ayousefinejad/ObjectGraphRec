#!/usr/bin/env python3
"""Read `data/lattice-runs/fusion_arms.csv` and say which arms actually moved Recall@20.

    python scripts/report_arms.py                 # screen (1 seed): deltas vs the control seed
    python scripts/report_arms.py --md            # same, as a markdown table to paste

Every arm is quoted against `control` **at the same seed**, never against a cross-seed mean: seed 0
is systematically the top of each triple (0.04344 / 0.04332 / 0.04331 / 0.04299 in the encoder
study), so mixing them manufactures a gain of about the size we are trying to detect.

The resolution line is the point of the report. The encoder study measured 0.0429 +/- 0.0005 over
3 seeds, so the smallest effect distinguishable from seed noise at 3 seeds is ~ +/-0.0012 (~2.8%).
A single-seed screen cannot resolve even that -- it can only reject arms that move nothing and
nominate the ones worth spending 3 seeds on. Anything inside the band is printed as `flat`, and
`flat` is a result, not a missing measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "lattice-runs" / "fusion_arms.csv"

# Smallest credible effect at 3 seeds, from the encoder study's seed spread (2 x 0.0005 std,
# rounded up). At 1 seed nothing inside this band means anything at all.
RESOLUTION = 0.0012

# Arms grouped by the question they answer, in the order the screen runs them. An arm's value is
# entirely in what it rules out, so the grouping is the argument.
BLOCKS = [
    ("Bound", "How much is the object modality worth at all?", ["noobj"]),
    ("Reference", "Reproduces downstream.csv; every delta below is against this.", ["control"]),
    ("Unfreeze fusion", "The fusion params get ~130 steps/run at --lr, so they never leave uniform.",
     ["lrfus", "lrfus_gated"]),
    ("Repair the object graph", "Break exact ties so the encoder's inter-label geometry is reachable.",
     ["tb_text", "tb_rand", "grp3_text", "res2_text"]),
    ("Confounds", "Tiebreak arms have self-loops on 100% of rows, the default graph on 22%.",
     ["tb_text_nosf", "selfloop"]),
    ("Item-graph propagation",
     "Replace the parameter-free item_adj @ h with learned attention over the same edges, so a "
     "neighbour can be down-weighted per item pair rather than only per modality.",
     ["itemgat"]),
    ("Other", "Symmetrisation, applied to all three modalities.", ["sym_max"]),
]

DIAG = ["zero_in_frac", "max_in_deg", "distinct_nbhd", "dup_slot_frac"]


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"{CSV} not found -- run scripts/run_lattice_study.py --arms ... first")
    return pd.read_csv(CSV)


def verdict(arm: str, d: float, n_seeds: int) -> str:
    if arm == "control":
        return "—"
    if arm == "noobj":
        # The sign means the opposite here: `noobj` deletes the object modality, so a *drop* is
        # the modality proving it carries signal. `flat` would be the bad outcome -- it would say
        # no object-graph repair can gain anything, because the channel is already worthless.
        if abs(d) < RESOLUTION:
            return "obj worthless"
        return "obj carries signal" if d < 0 else "obj actively hurts"
    if abs(d) < RESOLUTION:
        return "flat"
    if n_seeds < 2:
        return "nominate" if d > 0 else "worse"
    return "gain" if d > 0 else "worse"


def report(df: pd.DataFrame, variant: str, md: bool) -> None:
    v = df[df.variant == variant]
    if v.empty:
        return
    ctrl = v[v.arm == "control"]
    if ctrl.empty:
        sys.exit(f"{variant}: no control arm in {CSV.name} -- there is nothing to compare against")
    # Per-seed control, so an arm is always differenced against its own seed.
    base = ctrl.set_index("seed")["recall@20"]

    rows = []
    for block, why, arms in BLOCKS:
        first = True
        for arm in arms:
            a = v[v.arm == arm]
            if a.empty:
                continue
            common = a[a.seed.isin(base.index)]
            if common.empty:
                continue
            d = (common["recall@20"].values - base.loc[common.seed].values)
            n = len(d)
            rows.append({
                "block": block if first else "",
                "arm": arm,
                "n": n,
                "R@20": f"{common['recall@20'].mean():.5f}",
                "dR@20": f"{d.mean():+.5f}",
                "d%": f"{100 * d.mean() / base.loc[common.seed].values.mean():+.1f}%",
                "verdict": verdict(arm, d.mean(), n),
                **{k: (f"{a[k].mean():.4g}" if a[k].notna().any() else "—") for k in DIAG},
                "why": why if first else "",
            })
            first = False

    out = pd.DataFrame(rows)
    n_seeds = int(v.groupby("arm").size().max())
    print(f"\n{variant}: {len(v)} runs, {v.arm.nunique()} arms, up to {n_seeds} seed(s) each")
    print(f"control R@20 = {', '.join(f'seed{s}:{r:.5f}' for s, r in base.items())}")
    print(f"resolution   = +/-{RESOLUTION:.4f} at 3 seeds; a 1-seed screen resolves less than that, "
          f"so it nominates rather than concludes\n")

    show = out.drop(columns=["why"])
    if md:
        print(show.to_markdown(index=False))
    else:
        print(show.to_string(index=False))

    print("\nwhat each block rules out:")
    for block, why, _ in BLOCKS:
        if (out.block == block).any():
            print(f"  {block:<24} {why}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--md", action="store_true", help="Markdown table instead of plain text")
    args = p.parse_args()

    df = load()
    for variant in (args.variants or sorted(df.variant.unique())):
        report(df, variant, args.md)


if __name__ == "__main__":
    main()
