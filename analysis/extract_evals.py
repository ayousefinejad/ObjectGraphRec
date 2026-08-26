#!/usr/bin/env python3
"""Extract ONLY the ObjectGraph object-graph-modality evaluations.

Scope: the intrinsic evaluation of the object co-occurrence graph encoder --
held-out co-occurrence link prediction and the embedding diagnostics that go
with it. Deliberately EXCLUDES every downstream recommender result
(LATTICE / MICRO / CRANE Recall@20, NDCG): those measure a recommender that
consumes the modality, not the modality itself.

Sources (read-only):
    docs/objectgraph-logs/results.csv        447 runs x 57 cols -- the spine
    docs/objectgraph-logs/stage*/*.json      same runs + per-epoch history
    docs/notion/eval_baselines.json          baselines, dataset, split, threshold

Outputs (this directory):
    og_runs.csv        447 rows, one per run: every config knob + every metric
    og_curves.csv      long format: run_key, epoch, loss, val_auc  (~17k rows)
    og_baselines.csv   5 non-neural / untrained reference methods
    og_dataset.json    corpus stats, split sizes, min_cooc threshold table
    og_evaluations.csv the evaluation registry (what each eval needs)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
LOGS = OG / "docs/objectgraph-logs"
BASELINES = OG / "docs/notion/eval_baselines.json"

# Config knobs that define an arm, in the order a reader wants them.
CFG = ["sweep", "tag", "seed", "stage", "backbone", "backbone_effective",
       "edge_mode", "min_cooc", "hidden_dim", "num_layers", "lr", "temperature",
       "epochs", "patience", "eval_every", "dropout", "neg_mode", "neg_ratio",
       "mask_rate", "mae_epochs", "mae_alpha", "remask"]

# Metrics, grouped by the evaluation they belong to.
PRIMARY = ["test_auc", "test_ap", "val_auc", "val_ap"]           # degree-matched
UNIFORM = ["test_auc_uniform", "test_ap_uniform"]                 # comparability
STRATA = ["auc_c=1", "n_c=1", "auc_c=2-4", "n_c=2-4", "auc_c>=5", "n_c>=5"]
DEGREE = ["auc_degq1", "auc_degq2", "auc_degq3", "auc_degq4"]
GEOMETRY = ["effective_rank", "alignment", "uniformity", "degree_bias_rho"]
DIAGNOSTIC = ["knn_room_purity", "n_labelled"]
TRIPWIRE = ["train_auc"]                                          # leakage check
DYNAMICS = ["best_epoch", "epochs_run", "final_loss"]
COST = ["wall_clock_s", "n_params"]
GRAPH = ["n_nodes", "n_edges"]
PROV = ["git_commit", "scenes_sha256", "torch", "python"]

METRICS = PRIMARY + UNIFORM + STRATA + DEGREE + GEOMETRY + DIAGNOSTIC + TRIPWIRE + DYNAMICS + COST + GRAPH


def clean(v):
    """Blank out NaN/empty so downstream plotting sees a real missing value."""
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    return s


def main() -> None:
    rows = list(csv.DictReader((LOGS / "results.csv").open()))
    print(f"  results.csv           {len(rows)} runs")

    # ---- join CSV rows to their per-run JSON -------------------------------
    # results.csv's `run_key` is a JSON config dict; the JSON files are named by
    # hash. Neither can address the other, so join on identity instead:
    # (sweep, seed, test_auc). test_auc is a 16-digit float, so this is exact.
    # 11 keys are shared by genuine repeat-runs (identical config AND identical
    # score); those are paired in stable filename order, which is safe because
    # such runs are interchangeable by construction.
    buckets: dict[tuple, list[Path]] = {}
    for p in sorted(LOGS.glob("stage*/*.json")):
        d = json.loads(p.read_text())
        key = (p.parent.name, str(d.get("seed")), round(float(d["metrics"]["test_auc"]), 10))
        buckets.setdefault(key, []).append(p)
    n_json = sum(len(v) for v in buckets.values())
    print(f"  stage*/*.json         {n_json} run files")

    out_rows, curves, unmatched = [], [], 0
    for r in rows:
        key = (r["sweep"], str(r["seed"]), round(float(r["test_auc"]), 10))
        pool = buckets.get(key)
        # a run_key that is a config dict is unwieldy as an id; mint a short
        # stable one instead and keep the original for traceability.
        rec = {"run_key": "", "config_key": r.get("run_key", ""), "json_file": ""}
        if pool:
            p = pool.pop(0)
            rec["run_key"] = p.stem            # e.g. stageA_0129366067
            rec["json_file"] = str(p.relative_to(LOGS))
            d = json.loads(p.read_text())
            cfg = d.get("cfg", {})
            # mae_lr and init_from live only in the JSON
            rec["mae_lr"] = clean(cfg.get("mae_lr"))
            rec["init_from"] = clean(cfg.get("init_from"))
            # results.csv only records the knob a sweep is *varying* and leaves
            # the rest blank (its run_key works the same way). The JSON cfg is
            # always complete, so it wins wherever both exist -- otherwise e.g.
            # stageE's mask_rate is blank on 50 of 70 repaired runs.
            json_cfg = {k: clean(cfg.get(k)) for k in CFG if cfg.get(k) is not None}
            for h in d.get("history", []):
                curves.append({"run_key": rec["run_key"], "sweep": r["sweep"],
                               "epoch": h.get("epoch"), "loss": h.get("loss"),
                               "val_auc": h.get("val_auc")})
        else:
            unmatched += 1
            rec["mae_lr"] = rec["init_from"] = ""
            json_cfg = {}
        for k in CFG:
            rec[k] = json_cfg.get(k) or clean(r.get(k))
        for k in METRICS:
            rec[k] = clean(r.get(k))
        for k in PROV:
            rec[k] = clean(r.get(k))
        out_rows.append(rec)

    cols = ["run_key", "config_key", "json_file"] + CFG + ["mae_lr", "init_from"] + METRICS + PROV
    with (HERE / "og_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"  -> og_runs.csv        {len(out_rows)} rows x {len(cols)} cols"
          + (f"   ({unmatched} without a matched JSON)" if unmatched else ""))

    with (HERE / "og_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "sweep", "epoch", "loss", "val_auc"])
        w.writeheader()
        w.writerows(curves)
    print(f"  -> og_curves.csv      {len(curves)} (run, epoch) points")

    # ---- baselines / dataset / split / threshold --------------------------
    b = json.loads(BASELINES.read_text())
    with (HERE / "og_baselines.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "auc", "ap", "auc_uniform", "auc_std"])
        w.writeheader()
        for name, m in b["baselines"].items():
            w.writerow({"method": name, "auc": m.get("auc"), "ap": m.get("ap"),
                        "auc_uniform": m.get("auc_uniform"), "auc_std": m.get("auc_std", "")})
    print(f"  -> og_baselines.csv   {len(b['baselines'])} reference methods")

    (HERE / "og_dataset.json").write_text(json.dumps(
        {"dataset": b["dataset"], "split": b["split"], "threshold": b["threshold"]}, indent=2))
    print("  -> og_dataset.json    corpus stats + split + min_cooc threshold table")


if __name__ == "__main__":
    main()
