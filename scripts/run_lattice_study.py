#!/usr/bin/env python3
"""Run LATTICE once per (encoder variant, seed) and collect the downstream metrics.

    python scripts/run_lattice_study.py --variants shipped default_fixed converged tuned --seeds 0 1 2

Each run is a plain `main.py` subprocess against that variant's isolated dataset directory, so
the recommender code path is byte-identical to the published one. Metrics are parsed from the
final `test==` line, which `main.py` prints only when validation Recall@20 improves -- i.e. the
test scores of the best validation epoch, never a last-epoch or best-test number.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "lattice-runs" / "logs"
OUT = ROOT / "data" / "lattice-runs" / "downstream.csv"

# 'Epoch 30 [3.8s + 51.6s]: test==[...], recall=[0.028, 0.040], precision=[...], hit=[...], ndcg=[...]'
LINE = re.compile(
    r"test==\[.*?\], recall=\[([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+)\], "
    r"hit=\[([\d.]+), ([\d.]+)\], ndcg=\[([\d.]+), ([\d.]+)\]"
)
FIELDS = ["variant", "seed", "recall@10", "recall@20", "precision@10", "precision@20",
          "hit@10", "hit@20", "ndcg@10", "ndcg@20", "best_epoch", "wall_clock_s"]

# The fusion arm varies a second axis, so it gets its own file rather than a new column in
# downstream.csv -- that keeps the encoder study's schema and its already-written rows intact.
FUSION_OUT = ROOT / "data" / "lattice-runs" / "fusion.csv"
FUSION_FIELDS = FIELDS[:1] + ["fusion"] + FIELDS[1:]

# The graph-repair study varies a third axis and gets its own file again, so downstream.csv and
# fusion.csv keep the schemas their already-written rows were collected under.
ARMS_OUT = ROOT / "data" / "lattice-runs" / "fusion_arms.csv"
ARMS_FIELDS = FIELDS[:1] + ["arm"] + FIELDS[1:] + ["zero_in_frac", "max_in_deg",
                                                   "distinct_nbhd", "dup_slot_frac", "eps_used"]

# Named arms of the fusion / object-graph study. `control` must reproduce downstream.csv exactly;
# every other arm is defined by the flags it adds to BASE, so the log records the whole recipe.
ARMS = {
    # Bounds the entire study: if dropping the object modality costs nothing, no object-graph
    # repair can gain more than that gap.
    "noobj":        ["--modalities", "image,text"],
    "control":      [],
    # The fusion parameters see ~130 gradient steps a run at --lr, so they never leave uniform.
    "lrfus":        ["--lr_fusion", "0.05"],
    "lrfus_gated":  ["--lr_fusion", "0.05", "--fusion", "gated"],
    # Primary: break the exact ties so the encoder's inter-label geometry becomes reachable.
    "tb_text":      ["--obj_knn", "tiebreak", "--knn_tiebreak", "text"],
    # Control for tb_text -- de-hubs identically with zero cross-modal information, so a gain
    # here means "the hubs were the problem", not "text signal leaked into the object graph".
    "tb_rand":      ["--obj_knn", "tiebreak", "--knn_tiebreak", "random"],
    # Controls for the self-loop confound: every tiebreak arm has self-loops on 100% of rows,
    # the default graph on 22.1%. One removes them, one adds the residual without the tiebreak.
    "tb_text_nosf": ["--obj_knn", "tiebreak", "--knn_tiebreak", "text", "--knn_self", "drop"],
    "selfloop":     ["--knn_selfloop_alpha", "0.1"],
    "sym_max":      ["--knn_sym", "max"],
    "res2_text":    ["--obj_knn", "reserve", "--knn_reserve", "2", "--knn_tiebreak", "text"],
    "grp3_text":    ["--obj_knn", "group", "--knn_group_m", "3", "--knn_tiebreak", "text"],
    # Item-graph propagation, not graph *construction*: --item_prop leaves every adjacency and
    # every cache untouched, so unlike every arm above it needs no entry in OBJ_FLAGS and no
    # cache invalidation (omit stays empty and arm_dataset returns the plain variant dir).
    # Named itemgat, not gat: converged_gat and queue_gat_lattice.sh already mean the *encoder*
    # backbone, and the two axes have to stay distinguishable in filenames and CSVs.
    "itemgat":      ["--item_prop", "gat"],
    # Similarity threshold instead of top-k on the object graph: an item keeps as many neighbours
    # as it has similar ones. On the tuned encoder that is a mean out-degree of 269 against 10.
    "thr08":        ["--obj_knn", "threshold", "--knn_threshold", "0.8"],
    # Thresholding puts a self-loop on 100% of rows (the diagonal is 1.0) where the published
    # graph has one on 22.1%, the same confound tb_text/tb_text_nosf exists to separate.
    "thr08_nosf":   ["--obj_knn", "threshold", "--knn_threshold", "0.8", "--knn_self", "drop"],
}

# Flags that invalidate a cached adjacency. A stale cache and a working arm are indistinguishable
# downstream, so the arm runs against a dataset directory that simply does not have the file.
OBJ_FLAGS = {"--obj_knn", "--knn_tiebreak", "--knn_self", "--knn_selfloop_alpha"}


def stale_caches(flags):
    f = set(flags)
    if "--knn_sym" in f:
        return ["image", "text", "graph"]           # symmetrisation touches every modality
    return ["graph"] if f & OBJ_FLAGS else []


def arm_dataset(variant, arm, omit, core=5):
    """A dataset directory for one arm: the variant, minus the adjacencies the arm invalidates.

    Symlinks only, so an arm directory costs bytes rather than the 841 MB per cached tensor.
    """
    base = ROOT / "data" / "lattice-runs" / variant
    if not omit:
        return f"lattice-runs/{variant}"
    dst = ROOT / "data" / "lattice-runs" / "arms" / f"{variant}__{arm}"
    (dst / f"{core}-core").mkdir(parents=True, exist_ok=True)
    skip = {f"{m}_adj_10.pt" for m in omit}
    for src, out in ((base, dst), (base / f"{core}-core", dst / f"{core}-core")):
        for entry in sorted(src.iterdir()):
            if entry.is_dir() or entry.name in skip:
                continue
            link = out / entry.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(entry.resolve())
    for name in skip:
        stale = dst / f"{core}-core" / name
        if stale.exists() or stale.is_symlink():
            raise RuntimeError(f"{stale} must not exist: this arm rebuilds that adjacency and a "
                               f"cache hit would score the published graph under the arm's name.")
    return f"lattice-runs/arms/{variant}__{arm}"

BASE = [
    "--data_path", "data/", "--verbose", "5", "--epoch", "200", "--batch_size", "1024",
    "--regs", "[1e-5,1e-5,1e-2]", "--lr", "0.0005", "--model_name", "LATTICE",
    "--embed_size", "64", "--feat_embed_dim", "64", "--weight_size", "[64,64]",
    "--core", "5", "--topk", "10", "--lambda_coeff", "0.9", "--cf_model", "mf",
    "--n_layers", "1", "--mess_dropout", "[0.1, 0.1]", "--early_stopping_patience", "10",
    "--gpu_id", "0", "--Ks", "[10, 20]", "--test_flag", "part",
    # NOT --fast_laplacian. Its *forward* is bit-identical to the published diagflat+mm form
    # (torch.equal on real cached tensors and on a raw kNN adjacency), but its *backward* is not:
    # d(loss)/d(d_inv_sqrt) is a genuine ~10-term sum, and GEMM accumulates it in a different
    # order than the broadcast-mul reduction. Measured: fast=0 reproduces shipped_seed0.log
    # exactly over 4 epochs, fast=1 drifts from epoch 1 (100.83225 vs 100.83221). The speedup is
    # nil anyway -- epochs are 2.3 s either way, so the 22.7 s figure was a cold-start artifact.
]


def parse(log: str) -> dict | None:
    """Metrics from the last `test==` line, plus the epoch it came from."""
    best, epoch = None, None
    for line in log.splitlines():
        m = LINE.search(line)
        if m:
            best = m
            e = re.match(r"Epoch (\d+)", line)
            epoch = int(e.group(1)) if e else None
    if best is None:
        return None
    g = [float(x) for x in best.groups()]
    return {"recall@10": g[0], "recall@20": g[1], "precision@10": g[2], "precision@20": g[3],
            "hit@10": g[4], "hit@20": g[5], "ndcg@10": g[6], "ndcg@20": g[7],
            "best_epoch": epoch}


def done(out: Path, variant: str, seed: int, fusion: str, arm: str = "control") -> bool:
    if not out.exists():
        return False
    with out.open() as f:
        return any(r["variant"] == variant and int(r["seed"]) == seed
                   and r.get("fusion", "softmax") == fusion
                   and r.get("arm", "control") == arm
                   for r in csv.DictReader(f))


def check_provenance(log: str, arm: str, omit: list[str]) -> dict:
    """The arm's graph is what it claims to be, or the run does not count.

    Three ways an arm can silently become the control -- a flag argparse ignored, a cached
    adjacency that predates the flag, the wrong dataset directory -- and all three produce a
    clean run with published numbers. The diagnostics line catches all three at once.
    """
    diag = [l for l in log.splitlines() if l.startswith("LATTICE_DIAG ")]
    if not diag:
        raise RuntimeError(f"{arm}: no LATTICE_DIAG line -- cannot confirm which graph was used")
    d = json.loads(diag[-1][len("LATTICE_DIAG "):])
    for line in log.splitlines():
        if not line.startswith("LATTICE_KNN "):
            continue
        _, modality, source, _, spec = (line.split(None, 4) + [""])[:5]
        if modality in omit and source.endswith("=cache"):
            raise RuntimeError(f"{arm}: {modality} loaded from cache but the arm rebuilds it "
                               f"({spec})")
    if arm != "control" and ARMS[arm] and omit and not d.get("arm_spec"):
        raise RuntimeError(f"{arm}: LATTICE_DIAG reports the default spec -- flags were ignored")
    # The item_prop axis leaves no fingerprint in LATTICE_DIAG (it changes no adjacency), so the
    # arm_spec guard above is inert for it and a silently-ignored --item_prop would look exactly
    # like the control. LATTICE_PROP is that axis' fingerprint.
    flags = ARMS.get(arm, [])
    want = flags[flags.index("--item_prop") + 1] if "--item_prop" in flags else "dense"
    prop = [l for l in log.splitlines() if l.startswith("LATTICE_PROP ")]
    if not prop:
        raise RuntimeError(f"{arm}: no LATTICE_PROP line -- cannot confirm the propagation mode")
    got = dict(kv.split("=", 1) for kv in prop[-1].split()[1:] if "=" in kv).get("mode")
    if got != want:
        raise RuntimeError(f"{arm}: propagation mode is {got!r}, expected {want!r}")
    return d


def run(variant: str, seed: int, eval_cores: int, fusion: str, arm: str = "control") -> dict | None:
    LOGS.mkdir(parents=True, exist_ok=True)
    flags = ARMS[arm]
    omit = stale_caches(flags)
    dataset = arm_dataset(variant, arm, omit)
    parts = [variant] + ([] if fusion == "softmax" else [fusion]) + \
            ([] if arm == "control" else [arm])
    log_path = LOGS / ("_".join(parts) + f"_seed{seed}.log")
    env = {**os.environ, "LATTICE_EVAL_CORES": str(eval_cores)}
    cmd = [sys.executable, "main.py", "--dataset", dataset,
           "--seed", str(seed), "--fusion", fusion, *BASE, *flags]
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("# " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    log = log_path.read_text()
    # A NaN loss must fail the run regardless of exit code: main.py's own guard used to call bare
    # sys.exit() (return code 0), so `lrfus_gated` collapsed at epoch 2 and this only check would
    # have parsed the one test== line printed before the collapse and recorded it as real.
    if proc.returncode != 0 or "ERROR: loss is nan." in log:
        print(f"  FAILED rc={proc.returncode}; see {log_path}")
        print("  " + "\n  ".join(log.splitlines()[-8:]))
        return None
    row = parse(log)
    if row is None:
        print(f"  no test== line in {log_path}")
        return None
    d = check_provenance(log, arm, omit)
    row |= {"variant": variant, "seed": seed, "fusion": fusion, "arm": arm,
            "wall_clock_s": round(wall, 1)}
    row |= {k: d.get(k) for k in ("zero_in_frac", "max_in_deg", "distinct_nbhd",
                                  "dup_slot_frac", "eps_used")}
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eval-cores", type=int, default=10)
    p.add_argument("--fusions", nargs="+", default=["softmax"],
                   help="Modality fusion arms {softmax, sigmoid, gated, conf}. Anything other "
                        "than the lone default writes to fusion.csv instead of downstream.csv.")
    p.add_argument("--arms", nargs="+", default=None, choices=sorted(ARMS),
                   help="Named arms of the object-graph / fusion study. Writes to "
                        "fusion_arms.csv, leaving the encoder study's files untouched.")
    args = p.parse_args()

    arm_run = args.arms is not None
    fusion_arm = args.fusions != ["softmax"]
    out = ARMS_OUT if arm_run else (FUSION_OUT if fusion_arm else OUT)
    fields = ARMS_FIELDS if arm_run else (FUSION_FIELDS if fusion_arm else FIELDS)
    arms = args.arms if arm_run else ["control"]

    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for variant in args.variants:
            for arm in arms:
                for fusion in args.fusions:
                    for seed in args.seeds:
                        tag = f"{variant}/{arm}/{fusion} seed={seed}"
                        if done(out, variant, seed, fusion, arm):
                            print(f"{tag}: already in {out.name}, skipping")
                            continue
                        print(f"{tag}: running...", flush=True)
                        try:
                            row = run(variant, seed, args.eval_cores, fusion, arm)
                        except RuntimeError as e:
                            print(f"  REJECTED: {e}")
                            continue
                        if not row:
                            continue
                        w.writerow(row)
                        f.flush()
                        print(f"  R@20={row['recall@20']:.5f} NDCG@20={row['ndcg@20']:.5f} "
                              f"({row['wall_clock_s']:.0f}s, best epoch {row['best_epoch']})"
                              + (f"  zero_in={row['zero_in_frac']} max_in={row['max_in_deg']} "
                                 f"nbhd={row['distinct_nbhd']}" if arm_run else ""))


if __name__ == "__main__":
    main()
