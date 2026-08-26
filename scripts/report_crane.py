#!/usr/bin/env python3
"""Report a CRANE grid on home_v2, selected on validation, next to the LATTICE baselines.

    python scripts/report_crane.py                       # newest log under the CRANE clone
    python scripts/report_crane.py --log path/to.log     # a specific run
    python scripts/report_crane.py --md

Two things this script exists to correct, both of which change the headline number:

1. **CRANE's own harness selects on test.** utils/quick_start.py:82-84 ranks the grid with
   `if best_test_upon_valid[val_metric] > best_test_value`, i.e. on *test* Recall@20. The row it
   prints under "BEST" is therefore a test-set selection. This script re-ranks the same rows on
   validation Recall@20 and reports that row's test metrics instead. Both are shown, so the size
   of the leak is visible rather than argued about.

2. **CRANE and LATTICE do not evaluate the same users.** 11,548 of home_v2's users have an empty
   train list but one val and one test interaction each. LATTICE scores every user with a
   non-empty test list (N_LATTICE below); MMRec's `filter_out_cod_start_users` drops the
   train-empty ones (N_CRANE), and cannot include them -- dataloader's `_get_pos_items_per_u`
   calls uid_freq.get_group(u), which raises for a user with no train rows. Those users are
   unreachable for any train-embedding model, so LATTICE scores them and gets ~0. LATTICE's
   Recall is thus diluted by N_CRANE/N_LATTICE relative to CRANE's.

   The `warm-equiv` column rescales LATTICE onto CRANE's population by that ratio. It is an
   UPPER BOUND on LATTICE, not a measurement: it assumes LATTICE scores exactly zero on all
   11,548 cold users. Any hit it does get there makes the true warm number lower. It is the
   conservative direction for the comparison being made (CRANE vs LATTICE), so it is the one
   reported -- but the raw column is kept alongside and neither is presented as "the" number.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT.parent / "CRANE" / "CRANE" / "src" / "log"
RUNS = ROOT / "data" / "lattice-runs"

# Counted from data/home_v2-2/5-core/*.json, not from either framework's own report:
# users with a non-empty test list, and of those the ones with a non-empty train list.
N_LATTICE, N_CRANE = 59251, 47703

# The grid summary block. `Parameters: [...]=(...)` then two metric lines. Matched on the
# bracketed name list rather than \S+ because the list contains spaces ('dropout', 'reg_weight').
BLOCK = re.compile(
    r"Parameters: \[(?P<names>[^\]]*)\]=\((?P<vals>[^)]*)\),\s*"
    r"best valid: (?P<valid>.*?),\s*"
    r"best test: (?P<test>.*?)(?=\n\w{3} \d|\n\n|\Z)",
    re.S,
)


def metrics(blob: str) -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(r"(\w+@\d+): ([\d.]+)", blob)}


def parse(log: Path) -> pd.DataFrame:
    text = log.read_text()
    if "============All Over" not in text:
        sys.exit(f"{log.name} has no 'All Over' summary -- the run did not finish the grid")
    tail = text.split("============All Over", 1)[1]
    rows = []
    for m in BLOCK.finditer(tail):
        names = [n.strip().strip("'\"") for n in m["names"].split(",")]
        vals = [v.strip() for v in m["vals"].split(",")]
        v, t = metrics(m["valid"]), metrics(m["test"])
        rows.append({**dict(zip(names, vals)),
                     "val R@20": v["recall@20"],
                     "test R@20": t["recall@20"], "test R@10": t["recall@10"],
                     "test N@20": t["ndcg@20"], "test P@20": t["precision@20"]})
    if not rows:
        sys.exit(f"{log.name}: 'All Over' present but no Parameters blocks matched")
    return pd.DataFrame(rows)


def lattice_rows() -> pd.DataFrame:
    """The three LATTICE reference points, each averaged over the seeds actually run."""
    tune = pd.read_csv(RUNS / "tuning.csv")
    fus = pd.read_csv(RUNS / "fusion_arms.csv")
    picks = [
        ("LATTICE mf, image+text", fus[(fus.arm == "noobj") & (fus.variant == "default_fixed")]),
        ("LATTICE lightgcn-3, image+text", tune[tune.cell == "lgn3"]),
        ("LATTICE lightgcn-3, +object graph", tune[tune.cell == "obj_lgn3"]),
    ]
    return pd.DataFrame([{
        "model": name, "seeds": len(g),
        "R@20": g["recall@20"].mean(), "R@10": g["recall@10"].mean(),
        "N@20": g["ndcg@20"].mean(), "P@20": g["precision@20"].mean(),
    } for name, g in picks])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, help="CRANE log file (default: newest under the clone)")
    p.add_argument("--md", action="store_true", help="markdown instead of plain text")
    a = p.parse_args()

    log = a.log or max(LOGS.glob("CRANE-home_v2-*.log"), key=lambda f: f.stat().st_mtime)
    grid = parse(log).sort_values("val R@20", ascending=False).reset_index(drop=True)
    fmt = (lambda d: d.to_markdown(index=False)) if a.md else (lambda d: d.to_string(index=False))

    print(f"\nCRANE on home_v2 -- {log.name}, {len(grid)} configs, {N_CRANE:,} users evaluated\n")
    show = grid.copy()
    for c in [c for c in show.columns if "@" in c]:
        show[c] = show[c].map("{:.4f}".format)
    print(fmt(show))

    val_pick = grid.iloc[0]
    test_pick = grid.loc[grid["test R@20"].idxmax()]
    hp = [c for c in grid.columns if "@" not in c]
    desc = lambda r: ", ".join(f"{c}={r[c]}" for c in hp)
    print(f"\nselected on VALIDATION : {desc(val_pick)}  -> test R@20 {val_pick['test R@20']:.4f}"
          f"  N@20 {val_pick['test N@20']:.4f}")
    print(f"selected on TEST (what quick_start.py prints as BEST):\n"
          f"                         {desc(test_pick)}  -> test R@20 {test_pick['test R@20']:.4f}"
          f"   [+{test_pick['test R@20'] - val_pick['test R@20']:.4f} of selection leak]")

    lat = lattice_rows()
    scale = N_CRANE / N_LATTICE
    cmp = pd.DataFrame([{"model": f"CRANE ({desc(val_pick)})", "seeds": 1,
                         "users": N_CRANE, "R@20": val_pick["test R@20"],
                         "N@20": val_pick["test N@20"], "warm-equiv R@20": val_pick["test R@20"]}]
                       + [{"model": r.model, "seeds": r.seeds, "users": N_LATTICE,
                           "R@20": r["R@20"], "N@20": r["N@20"],
                           "warm-equiv R@20": r["R@20"] / scale} for _, r in lat.iterrows()])
    for c in ("R@20", "N@20", "warm-equiv R@20"):
        cmp[c] = cmp[c].map("{:.4f}".format)
    print(f"\nvs LATTICE on the same interactions (test Recall@20):")
    print(fmt(cmp))
    print(f"\nwarm-equiv rescales LATTICE by {N_LATTICE}/{N_CRANE} = {1/scale:.3f} onto CRANE's"
          f"\nevaluated population. It is an upper bound on LATTICE, not a measurement -- see the"
          f"\nmodule docstring. The two frameworks agree on candidate masking (train items only)"
          f"\nand on the Recall/NDCG definitions; the population is the one real difference.")


if __name__ == "__main__":
    main()
