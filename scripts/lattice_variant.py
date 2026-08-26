#!/usr/bin/env python3
"""Build an isolated LATTICE dataset directory for one object-graph encoder variant.

    python scripts/lattice_variant.py --variant tuned

Each variant gets its own `data/lattice-runs/<variant>/` tree so that `main.py --dataset
lattice-runs/<variant>` reads a different `object_feat.npy` while sharing every unchanged
artifact by symlink. Nothing under `data/home_v2-2/` is ever written.

Two things this script exists to get right:

1. **`graph_adj_10.pt` is never symlinked.** `Models.py` loads it from disk when present, and
   with `lambda_coeff=0.9` the cached graph carries 90% of the item-graph weight. Symlinking the
   shipped cache into a variant would make every variant silently score the *shipped* encoder.
   Leaving it absent forces `Models.py` to rebuild it from that variant's features, into that
   variant's own directory.
2. **`image_adj_10.pt` / `text_adj_10.pt` *are* symlinked.** They depend only on image/text
   features, which no variant changes, so rebuilding them would cost 2x841 MB and ~1 min per
   variant for a bit-identical result.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "home_v2-2"
RUNS = ROOT / "data" / "lattice-runs"

# Shared by every variant: features the object graph does not touch, the interaction split, and
# the two modality caches derived purely from those features.
LINK_TOP = ["image_feat.npy", "text_feat.npy"]
LINK_CORE = [
    "train.json", "val.json", "test.json", "user-item-dict.json",
    "item_list.txt", "user_list.txt", "raw_graph.txt", "raw_text.txt",
    "s_adj_mat.npz", "s_mean_adj_mat.npz", "s_norm_adj_mat.npz",
    "image_adj_10.pt", "text_adj_10.pt",
]
# Deliberately absent from LINK_CORE -- see the module docstring.
NEVER_LINK = "graph_adj_10.pt"


def setup(variant: str, core: int = 5, force: bool = False) -> Path:
    dst = RUNS / variant
    if dst.exists() and force:
        shutil.rmtree(dst)
    (dst / f"{core}-core").mkdir(parents=True, exist_ok=True)

    for name in LINK_TOP:
        link, target = dst / name, SRC / name
        if not target.exists():
            raise FileNotFoundError(target)
        if not link.exists():
            link.symlink_to(target)
    for name in LINK_CORE:
        link, target = dst / f"{core}-core" / name, SRC / f"{core}-core" / name
        if not target.exists():
            raise FileNotFoundError(target)
        if not link.exists():
            link.symlink_to(target)

    stale = dst / f"{core}-core" / NEVER_LINK
    # `exists()` follows symlinks, so a *broken* link would slip past it -- check both, and
    # reject any path that resolves outside this variant's own directory.
    if stale.is_symlink() or (stale.exists() and stale.resolve().parent != stale.parent):
        raise RuntimeError(
            f"{stale} resolves to {stale.resolve()}, outside this variant. With lambda_coeff=0.9 "
            f"this variant would score another encoder's graph. Remove it and re-run."
        )
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True)
    p.add_argument("--core", type=int, default=5)
    p.add_argument("--force", action="store_true", help="Recreate the directory from scratch")
    args = p.parse_args()
    dst = setup(args.variant, args.core, args.force)
    print(f"{dst}  (object_feat.npy still to be written; {NEVER_LINK} intentionally absent)")


if __name__ == "__main__":
    main()
