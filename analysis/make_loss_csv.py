#!/usr/bin/env python3
"""Extract LATTICE training-loss curves for plotting: with vs without the object modality.

    ~/hamedenv/bin/python make_loss_csv.py

Both arms are the `default_fixed` recipe (20-epoch encoder, repaired pipeline) under an
identical LATTICE protocol; the only difference is `--modalities image,text` vs the default
image,text,graph. So a difference between the curves is the object modality and nothing else.

Writes two files:
  loss_curves.csv       long format, one row per (arm, seed, epoch) -- for per-seed plots
  loss_curves_mean.csv  one row per (arm, epoch) with mean/std/n across seeds -- for a
                        mean +/- band plot

Reads only docs/lattice-logs/*.log. Writes nothing outside this directory.
"""
from __future__ import annotations

import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "object-graph" / "docs" / "lattice-logs"

# 'Epoch 12 [3.5s]: train==[98.23611=98.23424 + 0.00188]'
#                            total   = bpr      + emb(+reg)
# The eval epochs also emit 'Epoch N [a + b]: val==' / 'test==' lines; requiring the single
# bracket and 'train==' keeps this to the per-epoch training loss only.
LINE = re.compile(r"^Epoch (\d+) \[([\d.]+)s\]: train==\[([\d.]+)=([\d.]+) \+ ([\d.]+)\]")

ARMS = [("LATTICE (image+text)", "default_fixed_noobj_seed%d.log"),
        ("LATTICE + object graph", "default_fixed_seed%d.log")]
SEEDS = (0, 1, 2)


def main() -> None:
    rows = []
    for arm, pat in ARMS:
        for seed in SEEDS:
            p = LOGS / (pat % seed)
            if not p.exists():
                print(f"  ! missing {p.name}")
                continue
            n = 0
            for ln in p.read_text(errors="replace").splitlines():
                m = LINE.match(ln)
                if not m:
                    continue
                ep, secs, total, bpr, emb = m.groups()
                rows.append({"arm": arm, "seed": seed, "epoch": int(ep),
                             "loss": float(total), "bpr_loss": float(bpr),
                             "emb_loss": float(emb), "epoch_seconds": float(secs)})
                n += 1
            print(f"  {arm:26s} seed {seed}: {n} epochs")

    with (HERE / "loss_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "epoch", "loss", "bpr_loss",
                                           "emb_loss", "epoch_seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> loss_curves.csv       {len(rows)} rows")

    # Mean across seeds. Seeds early-stop at different epochs, so an epoch is only averaged
    # where every seed still has a value -- otherwise the tail of the mean curve would be
    # computed over a shrinking, self-selected subset of seeds and would bend for that reason
    # rather than for a real one.
    by = defaultdict(dict)
    for r in rows:
        by[r["arm"]].setdefault(r["epoch"], {})[r["seed"]] = r["loss"]
    out = []
    for arm, eps in by.items():
        full = max(len(v) for v in eps.values())
        for ep in sorted(eps):
            vals = list(eps[ep].values())
            if len(vals) < full:
                continue
            out.append({"arm": arm, "epoch": ep, "n_seeds": len(vals),
                        "loss_mean": round(st.mean(vals), 5),
                        "loss_std": round(st.stdev(vals), 5) if len(vals) > 1 else 0.0})
    with (HERE / "loss_curves_mean.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "epoch", "n_seeds", "loss_mean", "loss_std"])
        w.writeheader()
        w.writerows(out)
    print(f"  -> loss_curves_mean.csv  {len(out)} rows "
          f"({', '.join(f'{a}: {sum(1 for r in out if r[chr(97)+chr(114)+chr(109)] == a)} epochs' for a in by)})")


if __name__ == "__main__":
    main()
