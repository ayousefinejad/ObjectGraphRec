#!/usr/bin/env python3
"""Train one encoder configuration and export its object_feat.npy into a variant directory.

    python scripts/export_lattice_feats.py --variant tuned --config tuned --seed 0

Configurations are the same dicts the sweep used, so the exported features come from the exact
recipe the intrinsic study scored. The shipped `data/home_v2-2/object_feat.npy` is never written;
`--config shipped` instead symlinks it, so the downstream baseline is the published artifact
itself rather than a re-trained approximation of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.core import build_object_feat, prepare_study, train_eval
from scripts.lattice_variant import SRC, setup

LONG = {"epochs": 1000, "eval_every": 10, "patience": 20}
TUNED = {"temperature": 0.2, "lr": 3e-4, "neg_ratio": 2.0, "num_layers": 1, "backbone": "gat"}

# Each arm names an intrinsic result from the study so the downstream table can be read against
# it. 'shipped' is the frozen artifact, not a re-run: it is the thing already published.
# 'default_fixed' exists to break a confound: exporting through the repaired pipeline recovers
# ~49 case-collided labels, so converged/tuned resolve 531 distinct object vectors across the
# 14,503 items where the shipped artifact resolves only 293. Without this arm -- same
# hyperparameters as shipped, only the pipeline repaired -- any downstream delta would confound
# the label fix with the training recipe.
CONFIGS = {
    "shipped": None,                                    # test AUC 0.752 (20 epochs, as published)
    "default_fixed": {},                                # test AUC 0.752, repaired pipeline
    "converged": {**LONG},                              # test AUC 0.790
    "tuned": {**LONG, **TUNED},                         # test AUC 0.824
    # Isolated backbone swap: identical to `converged` except backbone=gat, so
    # `converged` vs `converged_gat` is the only GAT-vs-SAGE pair downstream in
    # which nothing else moves. `tuned` cannot serve this purpose -- it changes
    # backbone AND epochs, temperature, lr and depth together. Intrinsically this
    # recipe is stageD's GAT cell: 0.8070 +/- 0.0027 vs converged's 0.7886.
    "converged_gat": {**LONG, "backbone": "gat"},       # test AUC 0.807

    # --- neighbourhood aggregation, all at the default_fixed recipe -------------------------
    # GraphSAGE's own ablation axis (Hamilton et al. Sec 3.3) plus the two convolutions that
    # fix a different aggregation. Everything except the aggregator is held at DEFAULT_CFG, so
    # `default_fixed` IS the mean arm and is not re-exported -- these five are the contrast.
    "agg_max":   {"aggr": "max"},                       # element-wise max over neighbours
    "agg_sum":   {"aggr": "add"},                       # unnormalised sum: degree leaks in
    "agg_gcn":   {"backbone": "gcn"},                   # symmetric-normalised sum, D^-1/2 A D^-1/2
    "agg_gat":   {"backbone": "gat"},                   # attention-weighted sum
    # Eq. (2)'s w_ab = c_ab / sqrt(c_a c_b). Two keys move together by necessity, not by
    # choice: SAGEConv drops edge_weight silently, so weighted edges need wsage to be read at
    # all, and wsage without them is just the mean arm. Flagged in the table as a 2-key change.
    "agg_wmean": {"backbone": "wsage", "edge_mode": "weighted"},

    # --- negative-sampling strategy for the encoder's link-prediction loss, all at the
    # default_fixed recipe. 'uniform' (neg_mode default) IS default_fixed -- not re-exported.
    "neg_pop":     {"neg_mode": "degree", "neg_alpha": 1.0},         # popularity-aware
    "neg_hard":    {"neg_mode": "hard"},                             # hardest non-edge per anchor
    "neg_semihard": {"neg_mode": "semihard", "semihard_band": (0.05, 0.30)},  # dynamic mid-band
}


def export(variant: str, config: str, seed: int, core: int = 5,
           scenes_path: str | None = None) -> Path:
    dst = setup(variant, core=core)
    out = dst / "object_feat.npy"

    if config == "shipped":
        # Symlink, so the downstream baseline is byte-identical to the published artifact.
        if not out.exists():
            out.symlink_to(SRC / "object_feat.npy")
        print(f"{variant}: symlinked shipped object_feat.npy (no training)")
        return out

    cfg = {**DEFAULT_CFG, **CONFIGS[config], "seed": seed}
    if scenes_path:
        # Training on a different corpus changes the node vocabulary, which changes how many of
        # the 14,503 item labels resolve by exact match rather than by nearest-node fallback --
        # so provenance.json records the corpus, not just the recipe.
        cfg["scenes_path"] = scenes_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = prepare_study(cfg, device=device)
    res = train_eval(cfg, sd=sd, device=device, verbose=True)
    m = res["metrics"]
    print(f"{variant}: intrinsic test AUC {m['test_auc']:.4f}  AP {m['test_ap']:.4f} "
          f"(best epoch {m['best_epoch']})")

    ckpt = dst / f"encoder_{config}_seed{seed}.pt"
    torch.save({"method": config, "model_state_dict": res["state_dict"],
                "final_embeddings": res["embeddings"], "nodes": sd.nodes, "cfg": cfg}, ckpt)

    feats = build_object_feat(dataset="home_v2-2", core=str(core),
                              model_path=ckpt, out_path=out)
    (dst / "provenance.json").write_text(json.dumps(
        {"variant": variant, "config": config, "seed": seed,
         # str(): when --scenes is omitted this is DEFAULT_CFG's PosixPath, which
         # json cannot serialise. Only ever hit by arms that use the default
         # corpus, which is why every --scenes arm exported fine.
         "scenes_path": str(cfg["scenes_path"]), "n_nodes": len(sd.nodes),
         "scenes_sha256": sd.scenes_sha256,
         "intrinsic": {k: m[k] for k in ("test_auc", "test_ap", "auc_degq1", "best_epoch")},
         "cfg": {k: str(v) for k, v in cfg.items()}}, indent=2))
    print(f"{variant}: object_feat.npy {feats.shape} -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--scenes", default=None,
                   help="Train on a different scenes corpus (default: cfg['scenes_path'], the "
                        "shipped data/scenes.json). Does not affect --config shipped.")
    args = p.parse_args()
    export(args.variant, args.config, args.seed, args.core, args.scenes)


if __name__ == "__main__":
    main()
