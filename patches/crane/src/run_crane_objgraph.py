# coding: utf-8
"""CRANE with vs without the object-graph modality, seed-replicated.

The existing CRANE logs compare the two arms across dropout/reg_weight cells at a
single seed, which gives no replicate-level noise estimate -- the resulting
-0.7% R@20 sits at paired t = -0.29 and 3 of 8 cells move the other way. This
runs both arms at ONE fixed config over 3 seeds instead, so the delta can be
tested against seed spread the way the LATTICE and MICRO comparisons are.

Object features come from the MIT-only (OpenAI) encoder:
  data/lattice-runs/openai_mit/object_feat.npy  -- 510 unique vectors.

    python run_crane_objgraph.py --arm object    # use_object_graph=True
    python run_crane_objgraph.py --arm noobject  # baseline, same 3 seeds

main.py exposes only -m/-d, so configuration is passed through quick_start's
config_dict rather than by editing the shared YAML (which every other CRANE run
also reads).
"""
import argparse
import os

os.environ["NUMEXPR_MAX_THREADS"] = "48"

from utils.quick_start import quick_start

# Best cell from the earlier CRANE sweep (R@20 0.0424). Fixed here so the only
# thing varying within an arm is the seed.
FIXED = {"dropout": [0.9], "reg_weight": [0.001]}
SEEDS = [0, 1, 2]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["object", "noobject"])
    p.add_argument("--dataset", default="home_v2_openai")
    p.add_argument("--extra", default=None,
                   help='JSON dict merged into config_dict, e.g. \'{"obj_knn_sym": true}\'')
    a = p.parse_args()

    cfg = {
        "gpu_id": 0,
        "use_object_graph": a.arm == "object",
        "object_feature_file": "object_feat.npy",
        "seed": SEEDS,
        # sweep over seed only; dropout/reg_weight pinned to single values
        "hyper_parameters": ["dropout", "reg_weight", "seed"],
        # REQUIRED, and not the CRANE.yaml default (False = published 2-modality
        # behaviour). crane.py: "a third modality is only meaningful on the
        # batch_first=True axis... On the published axis it would only widen the
        # batch dimension from 2 to 3 and the object features would never meet
        # the image or text ones." With False the object arm is inert by
        # construction, and the earlier sweeps that scored 0.0398-0.0424 all ran
        # with True -- leaving it at the default also made the baseline arm
        # non-comparable to them (it scored ~0.012).
        "cross_modal_batch_first": True,
        "object_in_attention": True,
        **FIXED,
    }
    if a.extra:
        import json
        cfg.update(json.loads(a.extra))
    print(f"[crane-objgraph] arm={a.arm} dataset={a.dataset} "
          f"use_object_graph={cfg['use_object_graph']} "
          f"obj_knn_sym={cfg.get('obj_knn_sym', False)} seeds={SEEDS}", flush=True)
    quick_start(model="CRANE", dataset=a.dataset, config_dict=cfg, save_model=False)
