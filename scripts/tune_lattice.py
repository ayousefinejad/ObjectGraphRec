#!/usr/bin/env python3
"""Hyperparameter sweep for official LATTICE (image + text) on home_v2.

    python scripts/tune_lattice.py --cells lgn1 lgn2 lgn3 ngcf1 ngcf2 --seeds 0
    python scripts/tune_lattice.py --cells lgn2 --seeds 0 --probe      # 3 epochs, no CSV write

Every LATTICE run in this project has used the published Amazon-dataset recipe, which was never
tuned for home_v2-2 (65,139 users x 14,503 items, 2.58 train interactions per user). This sweeps
the recommender itself on the `noobj` arm -- image + text only, no object modality -- so the recipe
is chosen without the modality the project is trying to evaluate ever being visible.

Why this is not an `--arms` entry in run_lattice_study.py: that runner has no per-run hyperparameter
axis (`BASE` is a module constant, `ARMS[arm]` a fixed flag list, `ARMS_FIELDS` has no lr/cf_model
columns) and its `done()` dedups on (variant, seed, fusion, arm). `fusion_arms.csv` already holds
`default_fixed,noobj,0`, so every cell here would print "already in fusion_arms.csv, skipping" and
run nothing. Dedup is on a hash of the actual flags instead, so editing a cell makes it a new run.

Selection is on **validation** Recall@20 and never on test -- the sweep_objectgraph.py convention.
Test metrics are recorded so the winner can be reported, but they must not order anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_lattice_study import (  # noqa: E402
    BASE, LOGS, ROOT, arm_dataset, check_provenance, parse, stale_caches,
)

OUT = ROOT / "data" / "lattice-runs" / "tuning.csv"

# main.py prints `val==` every --verbose epochs but `test==` only when val Recall@20 improves, so
# the val curve is the only complete record of a run. run_lattice_study.parse() reads the test line;
# this is the same shape against the val one. Note the two spaces after `]:` on val lines -- the
# regex starts at `val==` so it does not care, but a copy of the LINE regex would.
VAL = re.compile(
    r"val==\[.*?\], recall=\[([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+)\], "
    r"hit=\[([\d.]+), ([\d.]+)\], ndcg=\[([\d.]+), ([\d.]+)\]"
)
EPOCH = re.compile(r"Epoch (\d+)")

# Every cell is the noobj arm -- official LATTICE, image + text, object modality masked to weight 0.
# `--epoch 400` lifts the truncation cap: best epochs already reach 160 against the old 200 cap, and
# with patience 10 x verbose 5 = 50 epochs such a run needed epoch 215 to stop. --verbose and
# --early_stopping_patience are deliberately NOT touched, so the selection procedure stays identical
# to the one behind the existing 3-seed control and that control needs no re-run.
COMMON = ["--modalities", "image,text", "--epoch", "400"]

# --weight_size length IS the CF depth: Models.py:215-216 takes len() before prepending embed_dim,
# so [64] is 1 propagation layer and [64,64,64] is 3. --mess_dropout is read only by ngcf
# (dropout_list is built under `if cf_model == 'ngcf'`) and indexed to n_ui_layers, so it is pinned
# at length 3 on both ngcf cells rather than varied with the axis under test.
CELLS = {
    # mf/[64,64] is the published control and is already in fusion_arms.csv at 3 seeds -- free.
    # mf is also exactly lightgcn at depth 0 (with weight_size [] the layer loop never runs and
    # mean([ego]) == ego, the identical return to Models.py:581), so lgn1..3 extend a nested axis.
    "lgn1":  ["--cf_model", "lightgcn", "--weight_size", "[64]"],
    "lgn2":  ["--cf_model", "lightgcn", "--weight_size", "[64,64]"],
    "lgn3":  ["--cf_model", "lightgcn", "--weight_size", "[64,64,64]"],
    "ngcf1": ["--cf_model", "ngcf", "--weight_size", "[64]", "--mess_dropout", "[0.1,0.1,0.1]"],
    "ngcf2": ["--cf_model", "ngcf", "--weight_size", "[64,64]", "--mess_dropout", "[0.1,0.1,0.1]"],
    # Block B: lr around the Block A winner (lgn3, val R@20 0.04304). No lgn4 cell -- the depth
    # gain is +0.0021 from 1 to 2 layers but only +0.0004 from 2 to 3, i.e. already inside the
    # 0.002 decision band, so a fourth layer cannot be distinguished from noise at one seed.
    "lgn3_lr1e4": ["--cf_model", "lightgcn", "--weight_size", "[64,64,64]", "--lr", "0.0001"],
    "lgn3_lr2e3": ["--cf_model", "lightgcn", "--weight_size", "[64,64,64]", "--lr", "0.002"],
    "lgn3_lr5e3": ["--cf_model", "lightgcn", "--weight_size", "[64,64,64]", "--lr", "0.005"],

    # The object modality turned back ON, under the tuned backbone. These re-override the
    # `--modalities image,text` in COMMON (argparse last-wins), so they are the `default_fixed`
    # variant at full strength: image + text + the SAGE/MIT+NYU object graph out of
    # lattice-runs/default_fixed/5-core/graph_adj_10.pt.
    #
    # The object modality is spelled `graph`, not `object` (utility/parser.py:80, Models.py:272);
    # `--modalities image,text,object` would silently mask ALL THREE weights to 0 -- parser.py:143
    # only rejects an empty subset, and 'object' simply never matches, so `graph` would read 0.0
    # exactly like it does on the noobj arm and the run would look like a null result.
    #
    # Paired with lgn2/lgn3 at the same seeds this is the A/B the five-config table could not run:
    # modality on vs off with the backbone held at the tuned setting instead of at `mf`.
    "obj_lgn2": ["--modalities", "image,text,graph",
                 "--cf_model", "lightgcn", "--weight_size", "[64,64]"],
    "obj_lgn3": ["--modalities", "image,text,graph",
                 "--cf_model", "lightgcn", "--weight_size", "[64,64,64]"],
}

FIELDS = [
    "run_key", "cell", "seed", "cf_model", "weight_size", "lr", "mess_dropout", "modalities",
    "val_recall@20", "val_ndcg@20", "val_tail5_recall@20", "best_val_epoch", "n_evals",
    "recall@10", "recall@20", "precision@10", "precision@20",
    "hit@10", "hit@20", "ndcg@10", "ndcg@20",
    "best_epoch", "epoch_capped", "wall_clock_s", "flags",
]


def run_key(flags: list[str], seed: int) -> str:
    """Identity of a run is the flags it was launched with, not the name someone gave the cell.

    Renaming a cell must not re-run it; editing a cell's flags must. Hashing the flag list gets
    both for free, which a (cell, seed) key does not.
    """
    return hashlib.sha1(("|".join(flags) + f"|seed={seed}").encode()).hexdigest()[:12]


def flag_value(flags: list[str], name: str, default: str) -> str:
    """Last occurrence wins, matching argparse -- overrides are appended after BASE."""
    idx = [i for i, f in enumerate(flags) if f == name]
    return flags[idx[-1] + 1] if idx else default


def parse_val(log: str) -> dict | None:
    """Best validation Recall@20 over the run, plus the statistics used to sanity-check it.

    `val_tail5` is the mean of the last five evals. The primary statistic is a max over 27-40
    evaluations and that max carries a measured best-of-N inflation of +0.0003 to +0.0007 over the
    run's own plateau -- the same size as the effects this sweep is looking for. A cell that leads
    on the max but not on the tail mean has found evaluation noise, not a better model.
    """
    r20, n20, epochs = [], [], []
    for line in log.splitlines():
        m = VAL.search(line)
        if not m:
            continue
        r20.append(float(m.group(2)))
        n20.append(float(m.group(8)))
        e = EPOCH.match(line)
        epochs.append(int(e.group(1)) if e else -1)
    if not r20:
        return None
    best = max(range(len(r20)), key=lambda i: r20[i])
    return {
        "val_recall@20": r20[best],
        "val_ndcg@20": n20[best],
        "val_tail5_recall@20": round(sum(r20[-5:]) / len(r20[-5:]), 6),
        "best_val_epoch": epochs[best],
        "n_evals": len(r20),
    }


def run(cell: str, seed: int, eval_cores: int, probe: bool) -> dict | None:
    LOGS.mkdir(parents=True, exist_ok=True)
    flags = COMMON + CELLS[cell]
    # --modalities is not in OBJ_FLAGS, so nothing is stale and the run reads the shared 841 MB
    # adjacency caches straight out of the variant directory. No arm symlink tree is created.
    omit = stale_caches(flags)
    dataset = arm_dataset("default_fixed", cell, omit)
    if probe:
        flags = flags + ["--epoch", "3", "--verbose", "100"]
    log_path = LOGS / (f"probe_{cell}.log" if probe else f"tune_{cell}_seed{seed}.log")
    env = {**os.environ, "LATTICE_EVAL_CORES": str(eval_cores)}
    cmd = [sys.executable, "main.py", "--dataset", dataset, "--seed", str(seed), *BASE, *flags]
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    log = log_path.read_text()

    if probe:
        # --epoch 3 never reaches an eval at --verbose 100, so main.py dies on `print(test_ret)`.
        # That is after the model is built, which is the whole point: construction and per-epoch
        # cost are measured, and an ngcf mess_dropout IndexError surfaces for ~90s instead of 30min.
        per = re.findall(r"Epoch \d+ \[([\d.]+)s", log)
        bad = [l for l in log.splitlines() if "Error" in l and "test_ret" not in l]
        print(f"  {cell}: {'/'.join(per) or 'no epoch line'} s/epoch, {wall:.0f}s wall"
              + (f"\n  CONSTRUCTION FAILED: {bad[0]}" if bad else ""))
        return None

    if proc.returncode != 0 or "ERROR: loss is nan." in log:
        print(f"  FAILED rc={proc.returncode}; see {log_path}")
        print("  " + "\n  ".join(log.splitlines()[-8:]))
        return None
    test, val = parse(log), parse_val(log)
    if test is None or val is None:
        print(f"  no {'test' if test is None else 'val'}== line in {log_path}")
        return None
    check_provenance(log, "control", omit)

    full = BASE + flags
    patience = int(flag_value(full, "--early_stopping_patience", "10"))
    verbose = int(flag_value(full, "--verbose", "5"))
    cap = int(flag_value(full, "--epoch", "400"))
    row = {
        "run_key": run_key(flags, seed), "cell": cell, "seed": seed,
        "cf_model": flag_value(full, "--cf_model", "mf"),
        "weight_size": flag_value(full, "--weight_size", "[64,64]"),
        "lr": flag_value(full, "--lr", "0.0005"),
        "mess_dropout": flag_value(full, "--mess_dropout", "[0.1, 0.1]"),
        # Recorded explicitly: obj_* and their noobj twins are otherwise identical in every other
        # config column, so without this the two arms are distinguishable only by the cell name.
        "modalities": flag_value(full, "--modalities", "image,text,graph"),
        # A run whose best epoch is within one patience window of the cap never got the chance to
        # stop on its own, so its score is a lower bound and it cannot be declared a winner.
        "epoch_capped": int(test["best_epoch"] + patience * verbose >= cap),
        "wall_clock_s": round(wall, 1), "flags": " ".join(flags),
    }
    return row | test | val


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", required=True, choices=sorted(CELLS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--eval-cores", type=int, default=10)
    p.add_argument("--probe", action="store_true",
                   help="3 epochs, no eval, nothing written -- times a backbone and smoke-tests it")
    a = p.parse_args()

    if a.probe:
        for cell in a.cells:
            run(cell, a.seeds[0], a.eval_cores, probe=True)
        return

    seen = set()
    if OUT.exists():
        with OUT.open() as f:
            seen = {r["run_key"] for r in csv.DictReader(f)}
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for cell in a.cells:
            for seed in a.seeds:
                key = run_key(COMMON + CELLS[cell], seed)
                if key in seen:
                    print(f"{cell} seed={seed}: already in {OUT.name} ({key}), skipping")
                    continue
                print(f"{cell} seed={seed}: running...", flush=True)
                try:
                    row = run(cell, seed, a.eval_cores, probe=False)
                except RuntimeError as e:
                    print(f"  REJECTED: {e}")
                    continue
                if not row:
                    continue
                w.writerow(row)
                f.flush()
                seen.add(key)
                print(f"  val R@20={row['val_recall@20']:.5f} (tail5 {row['val_tail5_recall@20']:.5f}) "
                      f"test R@20={row['recall@20']:.5f} "
                      f"[{row['wall_clock_s']:.0f}s, best epoch {row['best_epoch']}"
                      + (", EPOCH-CAPPED" if row["epoch_capped"] else "") + "]")


if __name__ == "__main__":
    main()
