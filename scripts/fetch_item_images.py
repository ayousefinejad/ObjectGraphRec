#!/usr/bin/env python3
"""Recover product image URLs for the 14,503 catalogue items from the Amazon metadata.

    ~/hamedenv/bin/python scripts/fetch_item_images.py

The pipeline kept only extracted 4096-d features (image_feat.npy); the images and their URLs were
discarded upstream. The SNAP 2014 metadata release -- the one the LATTICE paper cites
(jmcauley.ucsd.edu/data/amazon/links.html) -- carries `imUrl` per ASIN and joins on the ASINs in
5-core/item_list.txt.

Streams the gzip rather than extracting it (~100 MB compressed, several GB open) and keeps only
the rows whose ASIN is in our catalogue, so the output is a few hundred KB. No images are
downloaded here -- this produces URLs, nothing more.

The 2014 metadata is a Python dict literal per line, not JSON (single quotes), so it needs
ast.literal_eval. json.loads fails on every line.
"""
from __future__ import annotations
import ast, csv, gzip, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data/home_v2-2/5-core"
CACHE = ROOT / "data/prepare-objectgraph/cache"
URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz"
OUT = ROOT / "data/item_images.csv"

asins = [l.split("\t")[0] for l in (CORE / "item_list.txt").read_text().splitlines()]
labels = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
want = {a: i for i, a in enumerate(asins)}
print(f"catalogue: {len(want)} ASINs")

CACHE.mkdir(parents=True, exist_ok=True)
gz = CACHE / "meta_Home_and_Kitchen.json.gz"
if not (gz.exists() and gz.stat().st_size > 1_000_000):
    print("downloading SNAP 2014 metadata ...", flush=True)
    urllib.request.urlretrieve(URL, gz)
print(f"metadata: {gz.stat().st_size / 1e6:.0f} MB")

found, seen = {}, 0
with gzip.open(gz, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        seen += 1
        if seen % 200_000 == 0:
            print(f"  {seen:,} rows, {len(found):,} matched", flush=True)
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        a = d.get("asin")
        if a in want and a not in found:
            found[a] = {"imUrl": d.get("imUrl", ""), "title": (d.get("title") or "")[:160],
                        "price": d.get("price", ""), "brand": d.get("brand", "")}
            if len(found) == len(want):
                break
print(f"scanned {seen:,} metadata rows; matched {len(found):,} of {len(want):,} "
      f"({100 * len(found) / len(want):.1f}%)")
with_url = sum(1 for v in found.values() if v["imUrl"])
print(f"of those, {with_url:,} have an imUrl ({100 * with_url / len(want):.1f}% of the catalogue)")

with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "asin", "label", "imUrl", "title", "brand", "price"])
    for a, i in sorted(want.items(), key=lambda kv: kv[1]):
        r = found.get(a, {})
        w.writerow([i, a, labels[i], r.get("imUrl", ""), r.get("title", ""),
                    r.get("brand", ""), r.get("price", "")])
print(f"-> {OUT}")
