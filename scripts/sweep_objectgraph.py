#!/usr/bin/env python3
"""Hyperparameter sensitivity study for the ObjectGraph encoder.

Runs a named sweep stage, one row per (cell, seed), appending to a CSV that is the single
source of truth for the paper tables. Resumable: rows already present are skipped, so an
interrupted stage can be restarted with the same command.

    python scripts/sweep_objectgraph.py --sweep stageA --dry-run
    python scripts/sweep_objectgraph.py --sweep stageA

Nothing here writes outside --out (default data/graph-embeddings/sweeps/). The shipped
object_feat.npy and the LATTICE adjacency caches are never touched.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.core import prepare_study, train_eval

# Keys whose value changes the graph itself, and therefore the split, the negatives and the
# cached node features. Cells are grouped by these so StudyData is rebuilt only when needed.
GRAPH_KEYS = ("min_cooc", "edge_mode", "split_seed", "neg_seed", "val_frac", "test_frac", "neg_alpha")


def grid(**axes) -> list[dict]:
    """Full cartesian product of the named axes."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def ofat(base: dict | None = None, **axes) -> list[dict]:
    """One-factor-at-a-time: vary each axis alone from the base configuration.

    Deduplicated, so the base cell appears once even when it is a member of several axes.
    """
    cells, seen = [], set()
    for k, values in axes.items():
        for v in values:
            cell = {**(base or {}), k: v}
            key = json.dumps(cell, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                cells.append(cell)
    return cells


# --------------------------------------------------------------------------------------
# Stage definitions. Each returns (cells, seeds). Stages B/C/E take a --base override so a
# later stage can start from the winner of an earlier one.
# --------------------------------------------------------------------------------------

# Long-training defaults shared by every stage after A: 20 epochs is the shipped setting and
# is far too few to compare anything else against (measured: still improving at epoch 100).
LONG = {"epochs": 1000, "eval_every": 10, "patience": 20}


def stage_a(base: dict) -> tuple[list[dict], list[int]]:
    """Graph construction. First, because it changes the split and every downstream number."""
    cells = grid(min_cooc=[1, 2, 3, 5, 10], edge_mode=["dedup", "multiplicity", "weighted"])
    return [{**base, **LONG, **c} for c in cells], [0, 1, 2, 3, 4]


def stage_b(base: dict) -> tuple[list[dict], list[int]]:
    """Optimizer OFAT. Two passes over the same axes catch the epochs x lr coupling."""
    cells = ofat(
        {**base, **LONG},
        epochs=[20, 100, 300, 1000, 3000],
        lr=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        temperature=[0.05, 0.1, 0.2, 0.5, 1.0],
        neg_mode=["uniform", "degree"],
        neg_ratio=[0.5, 1.0, 2.0],
        dropout=[0.0, 0.1, 0.2, 0.5],
    )
    return cells, [0, 1, 2]


def stage_c(base: dict) -> tuple[list[dict], list[int]]:
    """Joint grid on the two axes most likely to interact."""
    cells = grid(lr=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2], temperature=[0.05, 0.1, 0.2, 0.5, 1.0])
    return [{**base, **LONG, **c} for c in cells], [0, 1, 2]


def stage_d(base: dict) -> tuple[list[dict], list[int]]:
    """Capacity and architecture. Supplies the numbers section 4.3.1 currently asserts."""
    cells = ofat(
        {**base, **LONG},
        hidden_dim=[16, 32, 64, 128, 256],
        num_layers=[1, 2, 3, 4],
        backbone=["sage", "wsage", "gat", "gcn"],
    )
    return cells, [0, 1, 2, 3, 4]


def stage_e(base: dict) -> tuple[list[dict], list[int]]:
    """GraphMAE stage ablation -- the second training stage that exists in code and not in
    the paper.

    The plan called for warm-starting this from stage C's joint optimum. Stage F showed that
    optimum does not transfer (validation 0.710 once combined with the other winners), so
    warm-starting here would have confounded 'stage 2 hurts' with 'that lr/tau cell hurts'.
    The base is instead the shipped configuration trained to convergence -- the `default_long`
    arm of Table F -- so the s1 row of this stage is a number the paper already reports and
    the ablation is anchored to it. `tuned_s1s2` re-runs the comparison at the tuned optimum
    to check the conclusion is not an artefact of the base.

    `remask` is swept rather than fixed because the shipped GraphMAE re-masks before a linear
    decoder, which drives the encoder's gradient to exactly zero (F14). Both variants are run:
    remask=True reproduces what the published pipeline did, remask=False answers the question
    the paper would actually be making a claim about.
    """
    cells = ofat(
        {**base, **LONG, "stage": "s1->s2", "remask": False},
        remask=[True, False],
        stage=["s1", "s2", "s1->s2"],
        mask_rate=[0.25, 0.5, 0.75, 0.9],
        mae_epochs=[100, 300, 1000, 3000],
        mae_alpha=[1.0, 2.0, 3.0, 5.0],
    )
    cells += [
        {**base, **LONG, "stage": "s2", "remask": True, "tag": "s2_shipped"},
        {**base, **LONG, **TUNED, "stage": "s1->s2", "remask": False, "tag": "tuned_s1s2"},
    ]
    return cells, [0, 1, 2, 3, 4]


# Selected on VALIDATION AUC from stages B/C/D, never on test. Axes where the shipped value
# already won (hidden_dim 64, dropout 0.0, neg_mode uniform, min_cooc 1, edge_mode
# multiplicity) are simply absent here.
TUNED = {"temperature": 0.2, "lr": 3e-4, "neg_ratio": 2.0, "num_layers": 1, "backbone": "gat"}

# Stage C found a strong lr x temperature interaction that one-factor-at-a-time cannot see:
# at tau=0.05 validation AUC runs from 0.749 (lr 3e-4) to 0.812 (lr 1e-2), while at tau=0.2 it
# is flat in lr. TUNED_C is that joint optimum, also selected on validation AUC.
#
# It does not transfer, and the stage F arms keep it to show why: Stage C measured
# tau=0.05/lr=1e-2 at num_layers=2 and neg_ratio=1.0, and combining it with the other OFAT
# winners drops validation AUC to 0.710 (vs 0.821 for TUNED). Coordinate-wise optima compose
# only when the axes are independent, and here they are not.
TUNED_C = {"temperature": 0.05, "lr": 1e-2, "neg_ratio": 2.0, "num_layers": 1}


def stage_f(base: dict) -> tuple[list[dict], list[int]]:
    """Final table, 10 seeds each.

    Four arms, because stacking one-factor-at-a-time winners silently assumes the axes do not
    interact. 'tuned_sage' isolates the optimizer gains from the backbone change, and
    'tuned_tau_only' shows how much of the total comes from temperature alone.
    """
    shipped = {k: DEFAULT_CFG[k] for k in ("epochs", "lr", "temperature", "hidden_dim", "eval_every")}
    return [
        {**base, **shipped, "patience": 10**9, "tag": "default"},
        {**base, **LONG, "tag": "default_long"},
        {**base, **LONG, "temperature": 0.2, "tag": "tuned_tau_only"},
        {**base, **LONG, **{k: v for k, v in TUNED.items() if k != "backbone"}, "tag": "tuned_sage"},
        {**base, **LONG, **TUNED, "tag": "tuned"},
        {**base, **LONG, **TUNED_C, "backbone": "sage", "tag": "tuned_joint_sage"},
        {**base, **LONG, **TUNED_C, "backbone": "gat", "tag": "tuned_joint_gat"},
    ], list(range(10))


STAGES = {"stageA": stage_a, "stageB": stage_b, "stageC": stage_c, "stageD": stage_d,
          "stageE": stage_e, "stageF": stage_f}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip() or "untracked"
    except Exception:
        return "untracked"


def run_key(cell: dict, seed: int) -> str:
    return json.dumps({**cell, "seed": seed}, sort_keys=True, default=str)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, choices=sorted(STAGES))
    p.add_argument("--out", type=Path, default=DEFAULT_CFG["log_dir"])
    p.add_argument("--base", type=str, default="{}", help="JSON cfg overrides applied to every cell")
    p.add_argument("--seeds", type=int, default=None, help="Override the stage's seed count")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base = json.loads(args.base)
    cells, seeds = STAGES[args.sweep](base)
    if args.seeds is not None:
        seeds = list(range(args.seeds))

    out_dir = args.out / args.sweep
    csv_path = args.out / "results.csv"

    done: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            done = {r["run_key"] for r in csv.DictReader(f) if r.get("run_key")}

    todo = [(c, s) for c in cells for s in seeds if run_key(c, s) not in done]
    print(f"{args.sweep}: {len(cells)} cells x {len(seeds)} seeds = {len(cells) * len(seeds)} runs")
    print(f"  already in {csv_path}: {len(cells) * len(seeds) - len(todo)}   to run: {len(todo)}")
    print(f"  estimated ~{len(todo) * 5 / 60:.0f} min at ~5 s/run")
    if args.dry_run:
        for c in cells[:8]:
            print("   ", {k: v for k, v in c.items() if k not in base})
        if len(cells) > 8:
            print(f"    ... and {len(cells) - 8} more cells")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = git_commit()
    versions = {"torch": torch.__version__, "python": sys.version.split()[0]}

    # Group by graph-construction settings so StudyData (split, negatives, MiniLM features) is
    # built once per distinct graph rather than once per run.
    def gkey(cell: dict) -> tuple:
        return tuple(cell.get(k, DEFAULT_CFG[k]) for k in GRAPH_KEYS)

    todo.sort(key=lambda cs: str(gkey(cs[0])))
    sd, cur = None, None
    t_start = time.time()

    for i, (cell, seed) in enumerate(todo, 1):
        cfg = {**DEFAULT_CFG, **cell, "seed": seed}
        if gkey(cell) != cur:
            cur = gkey(cell)
            sd = prepare_study(cfg, device=device)
            print(f"\n[graph] {dict(zip(GRAPH_KEYS, cur))} -> {sd.split.n_nodes} nodes, "
                  f"{len(sd.split.train_pairs)} train edges")

        res = train_eval(cfg, sd=sd, device=device)
        m = res["metrics"]

        row = {
            "run_key": run_key(cell, seed),
            "sweep": args.sweep,
            "tag": cell.get("tag", ""),
            "seed": seed,
            "git_commit": commit,
            **{k: cfg[k] for k in sorted(set(cell) | {"epochs", "lr", "temperature", "hidden_dim",
                                                       "num_layers", "dropout", "backbone", "neg_mode",
                                                       "neg_ratio", "min_cooc", "edge_mode"}) if k != "tag"},
            **m,
            **versions,
        }
        write_header = not csv_path.exists()
        # Stages contribute different columns; union them so late stages do not lose fields.
        if not write_header:
            with open(csv_path, newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f), [])
            if set(row) - set(existing):
                rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
                cols = existing + [c for c in row if c not in existing]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
                existing = cols
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=existing).writerow(
                    {k: row.get(k, "") for k in existing}
                )
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)

        stem = f"{args.sweep}_{abs(hash(row['run_key'])) % (10**10):010d}"
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"cfg": {k: str(v) for k, v in cfg.items()}, "seed": seed,
                 "metrics": m, "history": res["history"], "git_commit": commit},
                f, indent=2,
            )

        el = time.time() - t_start
        print(f"[{i}/{len(todo)}] seed={seed} test_auc={m['test_auc']:.4f} "
              f"val={m['val_auc']:.4f} best_ep={m['best_epoch']} "
              f"({el / i:.1f}s/run, {(len(todo) - i) * el / i / 60:.0f} min left) "
              f"{ {k: v for k, v in cell.items() if k not in base and k not in LONG} }")

    print(f"\nDone. {len(todo)} runs in {(time.time() - t_start) / 60:.1f} min -> {csv_path}")


if __name__ == "__main__":
    main()
