#!/usr/bin/env python3
"""Placebo object features: the real object_feat.npy with its ROWS permuted.

    ~/hamedenv/bin/python scripts/export_shuffled_feats.py \
        --src default_fixed --dst default_fixed_shufobj --perm-seed 20260816

This is the negative control for the claim "the object modality helps because the detected-object
vocabulary genuinely covers the catalogue". A row permutation keeps everything a capacity- or
regularisation-based explanation would need -- identical feature distribution, identical number of
distinct vectors, and an item graph that is *isomorphic* to the real one (permuting rows permutes
the kNN graph's vertices, nothing else). The single thing it destroys is which item each object
embedding belongs to. So a gain that survives the shuffle is not about objects.

The item graph must be rebuilt from these features, never inherited: scripts/lattice_variant.py
refuses to symlink graph_adj_10.pt for exactly this reason (with lambda_coeff=0.9 an inherited
cache would carry 90% of the item-graph weight from the real encoder).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lattice_variant import setup  # noqa: E402

FROZEN = (ROOT / "data" / "home_v2-2" / "object_feat.npy").resolve()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="default_fixed", help="variant whose object_feat.npy is permuted")
    p.add_argument("--dst", default="default_fixed_shufobj")
    p.add_argument("--perm-seed", type=int, default=20260816)
    p.add_argument("--core", type=int, default=5)
    a = p.parse_args()

    src = ROOT / "data" / "lattice-runs" / a.src / "object_feat.npy"
    feat = np.load(src)
    print(f"source {src}  shape={feat.shape} dtype={feat.dtype}")

    rng = np.random.default_rng(a.perm_seed)
    perm = rng.permutation(feat.shape[0])
    moved = int((perm != np.arange(feat.shape[0])).sum())
    assert moved > 0.99 * feat.shape[0], f"permutation barely moves anything ({moved} rows)"
    out_arr = feat[perm]

    dst_dir = setup(a.dst, a.core)
    dst = dst_dir / "object_feat.npy"
    if dst.resolve() == FROZEN or dst.is_symlink():
        raise SystemExit(f"refusing to write {dst}: it is the frozen artifact or a symlink to it")

    np.save(dst, out_arr)
    np.save(dst_dir / "shuffle_perm.npy", perm)

    # Multiset equality: the placebo must be the same bag of vectors, only re-assigned. Comparing
    # sorted byte views catches an accidental resample or dtype change that a shape check misses.
    back = np.load(dst)
    assert back.shape == feat.shape and back.dtype == feat.dtype
    assert np.array_equal(np.sort(back.view(np.uint8), axis=None),
                          np.sort(feat.view(np.uint8), axis=None)), "not a pure permutation"
    assert not np.array_equal(back, feat), "permuted file is identical to the source"
    stale = dst_dir / f"{a.core}-core" / "graph_adj_10.pt"
    assert not stale.exists(), f"{stale} is present -- it would score the real encoder's graph"

    print(f"-> {dst}  ({moved}/{feat.shape[0]} rows moved, perm seed {a.perm_seed})")
    print(f"-> {dst_dir / 'shuffle_perm.npy'}  (the permutation, so the run is reproducible)")
    print(f"   {stale.name} absent as required; LATTICE will rebuild it from these features")


if __name__ == "__main__":
    main()
