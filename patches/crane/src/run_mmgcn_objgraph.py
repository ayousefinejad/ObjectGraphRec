# coding: utf-8
"""MMGCN with vs without the object-graph modality, seed-replicated -- the same treatment
run_crane_objgraph.py gives CRANE, so the two are directly comparable in the same table.

Object features: data/lattice-runs/openai_mit/object_feat.npy (MIT-only OpenAI encoder, the
same file CRANE's object arm reads), so the object signal is held identical across both
architectures -- only the recommender differs.

    python run_mmgcn_objgraph.py --arm object    # use_object_graph=True
    python run_mmgcn_objgraph.py --arm noobject  # baseline, same 3 seeds
"""
import argparse
import os

os.environ["NUMEXPR_MAX_THREADS"] = "48"

from utils.quick_start import quick_start

SEEDS = [0, 1, 2]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["object", "noobject"])
    p.add_argument("--dataset", default="home_v2_openai")
    a = p.parse_args()

    cfg = {
        "gpu_id": 0,
        "use_object_graph": a.arm == "object",
        "object_feature_file": "object_feat.npy",
        "seed": SEEDS,
        "hyper_parameters": ["seed"],
    }
    print(f"[mmgcn-objgraph] arm={a.arm} dataset={a.dataset} "
          f"use_object_graph={cfg['use_object_graph']} seeds={SEEDS}", flush=True)
    quick_start(model="MMGCN", dataset=a.dataset, config_dict=cfg, save_model=False)
