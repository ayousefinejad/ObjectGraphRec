#!/usr/bin/env python3
"""Build a scenes corpus from Visual Genome, in the same shape as scenes.json.

    ~/hamedenv/bin/python data/prepare-objectgraph/build_visual_genome.py --min-count 20

Visual Genome annotates 108,077 images with free-form object names. One image's object list is
exactly what this project calls a "scene", so VG drops into the existing pipeline as another
corpus alongside MIT-Indoors and NYU-Depth -- no encoder or recommender change.

Two normalisation decisions, both deferring to sources outside this script:

  * VG's own `object_alias.txt` maps its synonyms to canonical names (guy -> man). Using the
    dataset's alias table rather than inventing one keeps the canonicalisation VG's, not ours.
  * `ObjectGraph.graph_data._norm_label` is then applied -- the same function every other corpus
    goes through -- so VG nodes are case-folded identically to MIT/NYU nodes and the vocabularies
    are directly comparable.

Frequency filtering is NOT optional: VG's raw vocabulary is ~75k names, most of them singletons,
typos, or phrases. --min-count keeps names appearing in at least that many images. The
vocabulary-size curve is printed at several thresholds so the choice is visible rather than
buried in a default.

Only annotations are downloaded (53 MB). The VG images are ~100 GB and are never needed: this
pipeline consumes object *names*, not pixels.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ObjectGraph.graph_data import _norm_label, load_scenes  # noqa: E402

BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
CACHE = ROOT / "data" / "prepare-objectgraph" / "cache"
OUT = ROOT / "data" / "visual_genome.json"
CURVE = (5, 10, 20, 50, 100, 200)


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / name
    if dst.exists() and dst.stat().st_size:
        print(f"  cached {name} ({dst.stat().st_size / 1e6:.0f} MB)")
        return dst
    print(f"  downloading {name} ...", flush=True)
    urllib.request.urlretrieve(f"{BASE}/{name}", dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
    return dst


def load_aliases() -> dict[str, str]:
    """VG's own synonym table: each line is `alias,canonical`."""
    p = fetch("object_alias.txt")
    out = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [x.strip() for x in line.split(",") if x.strip()]
        if len(parts) >= 2:
            for a in parts[1:]:
                out[a.lower()] = parts[0].lower()
    print(f"  {len(out)} alias mappings")
    return out


def object_name(obj: dict) -> str | None:
    """VG objects carry `names` (a list) or sometimes `name`. Take the first, ignore synsets --
    synsets are WordNet ids, not the surface words the item labels are matched against."""
    n = obj.get("names") or ([obj["name"]] if obj.get("name") else [])
    return n[0] if n else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-count", type=int, default=20,
                   help="keep object names appearing in >= this many images")
    p.add_argument("--min-objects", type=int, default=2,
                   help="drop scenes left with fewer than this many distinct objects")
    a = p.parse_args()

    alias = load_aliases()
    zp = fetch("objects.json.zip")
    print("  reading objects.json from the zip (not extracting it to disk) ...", flush=True)
    with zipfile.ZipFile(zp) as z:
        inner = [n for n in z.namelist() if n.endswith("objects.json")][0]
        with z.open(inner) as f:
            data = json.load(f)
    print(f"  {len(data)} images")

    # Pass 1: normalised name per object, and how many IMAGES each name appears in.
    scenes_raw, doc_freq = [], collections.Counter()
    for img in data:
        names = set()
        for obj in img.get("objects", []):
            raw = object_name(obj)
            if not raw:
                continue
            lab = _norm_label(alias.get(raw.strip().lower(), raw))
            if lab:
                names.add(lab)
        scenes_raw.append(names)
        doc_freq.update(names)
    print(f"  {len(doc_freq)} distinct object names before filtering")

    print("\n  vocabulary-size curve (names kept at each threshold):")
    for t in CURVE:
        keep = sum(1 for c in doc_freq.values() if c >= t)
        cov = sum(c for c in doc_freq.values() if c >= t) / sum(doc_freq.values())
        print(f"    >= {t:4d} images: {keep:6d} names   ({100 * cov:.1f}% of all annotations)")

    keep = {n for n, c in doc_freq.items() if c >= a.min_count}
    scenes = [sorted(s & keep) for s in scenes_raw]
    scenes = [s for s in scenes if len(s) >= a.min_objects]
    print(f"\n  --min-count {a.min_count}: {len(keep)} names, {len(scenes)} scenes "
          f"(of {len(data)} images), {sum(len(s) for s in scenes) / len(scenes):.2f} objects/scene")

    OUT.write_text(json.dumps(scenes), encoding="utf-8")
    # Round-trip through the loader every other corpus uses, so what the encoder will see is
    # what is reported here -- the check make_subsampled_corpus.py does for the same reason.
    back = load_scenes({"scenes_path": str(OUT)})
    nodes = {x for s in back for x in s}
    print(f"-> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"   round-trip via load_scenes: {len(back)} scenes, {len(nodes)} nodes")
    if len(back) != len(scenes):
        print(f"   NOTE: loader dropped {len(scenes) - len(back)} scenes (its own min_objects=2)")


if __name__ == "__main__":
    main()
