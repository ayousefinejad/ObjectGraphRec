#!/usr/bin/env python3
"""Turn the NYU Depth v2 labeled subset into a scenes JSON the ObjectGraph pipeline can read.

    python data/prepare-objectgraph/ingest_nyu_v2.py nyu_depth_v2_labeled.mat -o data/scenes_nyu.json

The labeled subset ships *ground-truth* per-pixel class labels (1,449 images, 894 classes) in its
`labels` and `names` fields, so scenes come straight out of the annotations -- no detector, no
OpenAI Vision call, no API cost, and no detector error to account for in the paper. That makes it
a cleaner provenance story than the shipped MIT corpus, which was labelled by a vision-language
model.

Labels are normalised through the same `format_label` used to build `scenes.json`, so the two
corpora can be compared or merged without the case-collision bug that produced 49 phantom nodes.

Writes a new file. Never touches data/scenes.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The canonical normaliser lives next to this file, but `prepare-objectgraph` has a hyphen and so
# is not importable as a package -- load it by path. Using the same `format_label` that built
# scenes.json is what keeps the two corpora mergeable without case-collision duplicates.
_spec = importlib.util.spec_from_file_location(
    "scenes_format", Path(__file__).resolve().parent / "scenes_format.py")
_sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sf)
format_label = _sf.format_label


def read_mat(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (labels, names). Handles both the v7.3/HDF5 and the older scipy .mat layouts."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "labels" not in f:
                raise KeyError(f"no `labels` dataset; found {list(f.keys())[:10]}")
            labels = np.array(f["labels"])                      # (N, W, H) in v7.3 order
            names = ["".join(chr(c[0]) for c in f[ref]) for ref in f["names"][0]]
        return labels, names
    except OSError:
        pass  # not HDF5 -- fall through to scipy for the older format

    from scipy.io import loadmat
    m = loadmat(path)
    if "labels" not in m:
        raise KeyError(f"no `labels` in {path.name}; found {[k for k in m if not k.startswith('__')]}")
    labels = np.transpose(m["labels"], (2, 0, 1))               # (H, W, N) -> (N, H, W)
    names = [str(n[0]) for n in m["names"][0]]
    return labels, names


def to_scenes(labels: np.ndarray, names: list[str], min_objects: int) -> list[list[str]]:
    """One scene per image: the set of distinct annotated classes present, normalised."""
    scenes = []
    for i in range(labels.shape[0]):
        ids = np.unique(labels[i])
        ids = ids[ids > 0]                                      # 0 is unlabelled
        objs = {format_label(names[j - 1]) for j in ids if 0 < j <= len(names)}
        objs.discard("")
        if len(objs) >= min_objects:
            scenes.append(sorted(objs))
    return scenes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mat", type=Path, help="nyu_depth_v2_labeled.mat")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "scenes_nyu.json")
    p.add_argument("--min-objects", type=int, default=2,
                   help="drop images with fewer distinct classes (default 2, matching the "
                        "shipped pipeline)")
    args = p.parse_args()

    if args.out.resolve() == (ROOT / "data" / "scenes.json").resolve():
        sys.exit("refusing to overwrite the shipped data/scenes.json -- pick another --out")

    labels, names = read_mat(args.mat)
    print(f"{args.mat.name}: {labels.shape[0]} images, {len(names)} classes")

    scenes = to_scenes(labels, names, args.min_objects)
    sizes = [len(s) for s in scenes]
    vocab = Counter(o for s in scenes for o in s)
    print(f"scenes (>= {args.min_objects} objects) : {len(scenes)}")
    print(f"objects/scene mean, max      : {np.mean(sizes):.2f}, {max(sizes)}")
    print(f"distinct object labels       : {len(vocab)}")
    print(f"most common                  : {', '.join(o for o, _ in vocab.most_common(8))}")

    args.out.write_text(json.dumps(scenes, indent=1))
    print(f"\nwrote {len(scenes)} scenes -> {args.out}")
    print("next: python scripts/eval_objectgraph.py --stats-only --scenes " + str(args.out))


if __name__ == "__main__":
    main()
