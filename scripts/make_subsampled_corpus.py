#!/usr/bin/env python3
"""Size-matched MIT corpus: MIT-Indoors randomly subsampled to NYU-Depth's scene count.

    ~/hamedenv/bin/python scripts/make_subsampled_corpus.py

MIT contributes 2,645 scenes and 1,007 graph nodes; NYU-Depth 579 scenes and 360 nodes. Any
MIT-vs-NYU difference downstream is therefore confounded with corpus size, and "MIT is the better
corpus" cannot be said without this control. Subsampling MIT to NYU's exact scene count holds
size fixed so the remaining difference is the imagery and its labels.

Matching is done on the count AFTER ObjectGraph.graph_data's own normalisation (case-folding,
per-scene dedup, and the min_objects=2 filter in _normalize_scene), because that is the number
of scenes the encoder actually trains on -- matching raw JSON entries would leave the two corpora
differently sized wherever the filter bites.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ObjectGraph.graph_data import build_cooccurrence, load_scenes  # noqa: E402

DATA = ROOT / "data"
MIT, NYU = DATA / "openai_mit.json", DATA / "nyu-depth.json"
OUT = DATA / "mit_sub579.json"
SEED = 20260816


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(MIT), help="corpus to subsample")
    ap.add_argument("--match", default=str(NYU), help="corpus whose scene count to match")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    mit_norm = load_scenes({"scenes_path": a.src})
    nyu_norm = load_scenes({"scenes_path": a.match})
    target = len(nyu_norm)
    out_path = Path(a.out)
    print(f"{Path(a.src).name} {len(mit_norm)} normalised scenes, "
          f"{Path(a.match).name} {target} -- subsampling to {target}")
    if len(mit_norm) < target:
        raise SystemExit(f"cannot subsample {len(mit_norm)} scenes down to {target}")

    # Sample from the NORMALISED scenes, then write those back out. Writing normalised rows is
    # safe: _normalize_scene is idempotent (already-folded labels fold to themselves, dedup is a
    # no-op, and every kept row already has >= 2 objects), so re-loading this file reproduces
    # exactly these scenes.
    rng = random.Random(SEED)
    sub = rng.sample(mit_norm, target)
    reloaded = [s for s in ([x for x in row] for row in sub) if len(s) >= 2]
    assert len(reloaded) == target, "normalisation is not idempotent on these rows"

    out_path.write_text(json.dumps(sub), encoding="utf-8")
    check = load_scenes({"scenes_path": str(out_path)})
    assert len(check) == target, f"re-loaded {len(check)} scenes, expected {target}"

    nodes_full, _, _ = build_cooccurrence(mit_norm)
    nodes_sub, pairs_sub, _ = build_cooccurrence(check)
    nodes_nyu, pairs_nyu, _ = build_cooccurrence(nyu_norm)
    print(f"-> {out_path}")
    print(f"   scenes  MIT {len(mit_norm):5d}  MIT-sub {len(check):5d}  NYU {len(nyu_norm):5d}")
    print(f"   nodes   MIT {len(nodes_full):5d}  MIT-sub {len(nodes_sub):5d}  NYU {len(nodes_nyu):5d}")
    print(f"   pairs   MIT-sub {len(pairs_sub):6d}  NYU {len(pairs_nyu):6d}")
    print(f"   objects/scene  MIT-sub {sum(len(s) for s in check) / len(check):.2f}  "
          f"NYU {sum(len(s) for s in nyu_norm) / len(nyu_norm):.2f}")
    print(f"\nnext: scripts/export_lattice_feats.py --variant {out_path.stem} "
          f"--config default_fixed --scenes data/{out_path.name}")


if __name__ == "__main__":
    main()
