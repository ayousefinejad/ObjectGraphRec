#!/usr/bin/env python3
"""Score the fixed evaluation protocol: dataset statistics, baselines, and any checkpoint.

    python scripts/eval_objectgraph.py --stats-only
    python scripts/eval_objectgraph.py --baselines
    python scripts/eval_objectgraph.py --checkpoint path/to/model.pt

Read-only with respect to every shipped artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ObjectGraph import eval as ev
from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.core import prepare_study
from ObjectGraph.encoder import GraphEncoder
from ObjectGraph.graph_data import build_cooccurrence, load_scenes


def dataset_stats(scenes: list[list[str]]) -> dict:
    nodes, pair_counts, node_counts = build_cooccurrence(scenes)
    sizes = np.array([len(s) for s in scenes])
    counts = np.array(list(pair_counts.values()))
    deg = Counter()
    for i, j in pair_counts:
        deg[i] += 1
        deg[j] += 1
    d = np.array([deg.get(i, 0) for i in range(len(nodes))])
    n = len(nodes)
    return {
        "n_scenes": len(scenes),
        "objects_per_scene_mean": float(sizes.mean()),
        "objects_per_scene_max": int(sizes.max()),
        "n_nodes": n,
        "n_edges": len(pair_counts),
        "density_pct": 100.0 * len(pair_counts) / (n * (n - 1) / 2),
        "degree_mean": float(d.mean()),
        "degree_median": float(np.median(d)),
        "degree_max": int(d.max()),
        "hub_label": nodes[int(d.argmax())],
        "singleton_edge_pct": 100.0 * float((counts == 1).mean()),
    }


def threshold_table(scenes: list[list[str]]) -> list[dict]:
    nodes, pair_counts, _ = build_cooccurrence(scenes)
    rows = []
    for t in (1, 2, 3, 5, 10):
        kept = {k: c for k, c in pair_counts.items() if c >= t}
        seen = {i for pair in kept for i in pair}
        rows.append({"min_cooc": t, "edges": len(kept), "non_isolated_nodes": len(seen),
                     "node_coverage_pct": 100.0 * len(seen) / len(nodes)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None, help="Also write results as JSON")
    args = p.parse_args()

    scenes = load_scenes()
    out: dict = {"dataset": dataset_stats(scenes), "threshold": threshold_table(scenes)}

    print("=== Dataset statistics ===")
    for k, v in out["dataset"].items():
        print(f"  {k:26s} {v if isinstance(v, (str, int)) else f'{v:.2f}'}")
    print("\n=== Co-occurrence threshold ===")
    print(f"  {'min_cooc':>8} {'edges':>8} {'nodes':>8} {'coverage':>9}")
    for r in out["threshold"]:
        print(f"  {r['min_cooc']:>8} {r['edges']:>8} {r['non_isolated_nodes']:>8} {r['node_coverage_pct']:>8.1f}%")

    if args.stats_only:
        if args.json:
            args.json.write_text(json.dumps(out, indent=2))
        return

    sd = prepare_study()
    split = sd.split
    print(f"\n=== Split (split_seed={DEFAULT_CFG['split_seed']}) ===")
    print(f"  train {len(split.train_pairs)}  val {len(split.val_pairs)}  test {len(split.test_pairs)}")
    print(f"  isolated nodes after split: {(split.train_degree == 0).sum()}")
    out["split"] = {"train": len(split.train_pairs), "val": len(split.val_pairs),
                    "test": len(split.test_pairs), "isolated": int((split.train_degree == 0).sum())}

    if args.baselines or args.checkpoint is None:
        print("\n=== Baselines (test set) ===")
        print(f"  {'method':28s} {'AUC_deg':>8} {'AP_deg':>8} {'AUC_unif':>9}")
        rows = {}
        bd = ev.baseline_scores(split, split.test_pairs, sd.neg_test)
        bu = ev.baseline_scores(split, split.test_pairs, sd.neg_test_uniform)
        for name in bd:
            rows[name] = {"auc": bd[name]["auc"], "ap": bd[name]["ap"], "auc_uniform": bu[name]["auc"]}
        fd = ev.feature_baseline(sd.x, split.test_pairs, sd.neg_test)
        fu = ev.feature_baseline(sd.x, split.test_pairs, sd.neg_test_uniform)
        rows["raw_minilm_cosine"] = {"auc": fd["auc"], "ap": fd["ap"], "auc_uniform": fu["auc"]}

        aucs, aps, aus = [], [], []
        for s in range(5):
            torch.manual_seed(s)
            m = GraphEncoder(sd.x.size(1), DEFAULT_CFG["hidden_dim"]).to(sd.x.device).eval()
            with torch.no_grad():
                z = m(sd.x, sd.edge_index)
            r = ev.link_pred_metrics(z, split.test_pairs, sd.neg_test)
            aucs.append(r["auc"])
            aps.append(r["ap"])
            aus.append(ev.link_pred_metrics(z, split.test_pairs, sd.neg_test_uniform)["auc"])
        rows["untrained_sage"] = {"auc": float(np.mean(aucs)), "ap": float(np.mean(aps)),
                                  "auc_uniform": float(np.mean(aus)), "auc_std": float(np.std(aucs))}
        for name, r in rows.items():
            print(f"  {name:28s} {r['auc']:>8.4f} {r['ap']:>8.4f} {r['auc_uniform']:>9.4f}")
        out["baselines"] = rows

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "final_embeddings" in ck:
            z = ck["final_embeddings"].to(sd.x.device)
        else:
            m = GraphEncoder(sd.x.size(1), DEFAULT_CFG["hidden_dim"]).to(sd.x.device)
            m.load_graphsage_state(ck.get("model_state_dict", ck))
            m.eval()
            with torch.no_grad():
                z = m(sd.x, sd.edge_index, sd.edge_weight)
        print(f"\n=== Checkpoint: {args.checkpoint} ===")
        print("  NOTE: core.train()/train_full() supervise on ALL edges and propagate all of")
        print("  them, so a checkpoint produced by that path has seen the val/test edges. The")
        print("  numbers below are leaked and are not comparable to sweep results, which use")
        print("  core.train_eval(). Use them only as a smoke test.")
        met = ev.evaluate_embeddings(z, split, sd.neg_val, sd.neg_test, sd.neg_test_uniform, scenes)
        for k, v in met.items():
            print(f"  {k:24s} {v:.4f}")
        out["checkpoint"] = {"path": str(args.checkpoint), "metrics": met}

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
