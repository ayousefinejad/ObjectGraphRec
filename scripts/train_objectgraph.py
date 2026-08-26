#!/usr/bin/env python3
"""Fine-tune ObjectGraph on data/scenes.json (contrastive + GraphMAE)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ObjectGraph import build_object_feat, train_full
from ObjectGraph.config import DEFAULT_CFG
from ObjectGraph.graph_data import load_scenes


def main() -> None:
    p = argparse.ArgumentParser(description="Train ObjectGraph on merged scenes.json")
    p.add_argument("--dataset", default="home_v2-2", help="LATTICE dataset for object_feat.npy")
    p.add_argument("--no-export", action="store_true", help="Skip build_object_feat")
    p.add_argument("--scenes", type=Path, default=None, help="Override scenes_path")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.scenes:
        cfg["scenes_path"] = args.scenes

    scenes = load_scenes(cfg)
    print(f"Scenes: {len(scenes)} from {cfg['scenes_path']}")
    print(f"Contrastive → {cfg['model_path']}")
    print(f"GraphMAE    → {cfg['graphmae_path']}")

    model_path = train_full(scenes, cfg)
    print(f"Saved: {model_path}")

    if not args.no_export:
        feats = build_object_feat(args.dataset, model_path=model_path)
        print(f"object_feat.npy shape: {feats.shape}")


if __name__ == "__main__":
    main()
