#!/usr/bin/env python3
"""Does the detected-object vocabulary appear in the Amazon items' own TEXT, not just in their
label field?

    ~/hamedenv/bin/python make_title_overlap.py

F17 measured coverage against `5-core/raw_graph.txt` -- one label per item. That field is
*derived* (it tracks the category leaf: item 7's path ends "...Teapots" and its label is
"teapot"), so coverage measured against it partly measures the labelling, not the catalogue.
Product text is raw, so it is an independent test of the same claim.

`raw_text.txt` is not a title: it is category-path + title + description concatenated, median 83
words and up to 1,744. Matching object names against that whole blob would count a hit anywhere
in a long description and inflate the number, so the fields are separated first.

Splitting them: a category path is SHARED by many items while a title is essentially unique, so
the longest leading token-prefix that also occurs in >= MIN_SHARED other items is the category
path. Everything after it is the product's own text. This is a heuristic and is reported as one
-- the per-field counts below let you see how much each field contributes rather than trusting a
single blended number.

Matching is exact on lowercased word n-grams (node names run 1-4 words), so 'towels' does NOT
match the node 'Towel'. That makes this a LOWER BOUND on true overlap, which is the direction an
overlap claim should err in.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
if str(OG) not in sys.path:
    sys.path.insert(0, str(OG))

from ObjectGraph.graph_data import load_scenes  # noqa: E402

CORE = OG / "data/home_v2-2/5-core"
CKPT = OG / "data/lattice-runs/default_fixed/encoder_default_fixed_seed0.pt"
CORPORA = {"mit": OG / "data/openai_mit.json", "nyu": OG / "data/nyu-depth.json"}
MIN_SHARED = 5          # a leading prefix this many items share is a category, not a title
MAX_CAT_TOKENS = 14
WORD = re.compile(r"[a-z0-9&]+")


def toks(s: str) -> list[str]:
    return WORD.findall(s.lower())


def split_fields(texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """(category tokens, product-text tokens) per item, via shared-prefix mining."""
    tok = [toks(t) for t in texts]
    counts = collections.Counter()
    for tk in tok:
        for k in range(1, min(MAX_CAT_TOKENS, len(tk)) + 1):
            counts[tuple(tk[:k])] += 1
    cats, rest = [], []
    for tk in tok:
        best = 0
        for k in range(1, min(MAX_CAT_TOKENS, len(tk)) + 1):
            if counts[tuple(tk[:k])] >= MIN_SHARED:
                best = k
        cats.append(tk[:best])
        rest.append(tk[best:])
    return cats, rest


def ngram_hits(tk: list[str], vocab: set[str], maxn: int) -> set[str]:
    out = set()
    for n in range(1, maxn + 1):
        for i in range(len(tk) - n + 1):
            g = " ".join(tk[i:i + n])
            if g in vocab:
                out.add(g)
    return out


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    nodes = ck["nodes"]
    maxn = max(len(n.split()) for n in nodes)
    vocab = {n.lower(): n for n in nodes}
    per_corpus = {k: {n.lower() for s in load_scenes({"scenes_path": str(p)}) for n in s}
                  for k, p in CORPORA.items()}

    texts = [l.rstrip("\n") for l in (CORE / "raw_text.txt").read_text(encoding="utf-8").splitlines()]
    labels = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
    assert len(texts) == len(labels) == 14503
    cats, rest = split_fields(texts)
    print(f"items {len(texts)}   object-graph nodes {len(nodes)} (up to {maxn} words)")
    print(f"category path: median {sorted(len(c) for c in cats)[len(cats) // 2]} tokens; "
          f"product text: median {sorted(len(r) for r in rest)[len(rest) // 2]} tokens")
    print(f"  e.g. CAT[{' '.join(cats[7])}]  TEXT[{' '.join(rest[7][:14])} ...]\n")

    # "Title-ish": the first 15 tokens of the product text. The title/description boundary is not
    # recoverable from this file, so this is a proxy and is labelled as one everywhere it appears.
    fields = {"category path": cats, "product text": rest,
              "product text, first 15 tokens": [r[:15] for r in rest]}

    rows, node_freq = [], collections.Counter()
    for fname, seq in fields.items():
        hits = [ngram_hits(tk, set(vocab), maxn) for tk in seq]
        n_any = sum(1 for h in hits if h)
        distinct = set().union(*hits) if hits else set()
        if fname == "product text":
            for h in hits:
                node_freq.update(h)
        row = {"field": fname, "items_with_>=1_object": n_any,
               "pct_items": round(100 * n_any / len(texts), 1),
               "distinct_nodes_seen": len(distinct),
               "pct_of_1068_nodes": round(100 * len(distinct) / len(nodes), 1),
               "mean_objects_per_item": round(sum(len(h) for h in hits) / len(texts), 2)}
        for c, cv in per_corpus.items():
            row[f"distinct_{c}_nodes"] = len({d for d in distinct if d in cv})
        rows.append(row)
        print(f"{fname:32s} items with >=1 object node: {n_any:6d} ({row['pct_items']:5.1f}%)   "
              f"distinct nodes {len(distinct):4d} ({row['pct_of_1068_nodes']:4.1f}%)   "
              f"mean/item {row['mean_objects_per_item']:.2f}")

    # Agreement with the label-field measurement, item by item: the two are independent views and
    # a claim is stronger where they agree than where either stands alone.
    lower_nodes = set(vocab)
    label_hit = [l.lower() in lower_nodes for l in labels]
    text_hit = [bool(ngram_hits(tk, lower_nodes, maxn)) for tk in rest]
    both = sum(1 for a, b in zip(label_hit, text_hit) if a and b)
    only_lab = sum(1 for a, b in zip(label_hit, text_hit) if a and not b)
    only_txt = sum(1 for a, b in zip(label_hit, text_hit) if b and not a)
    neither = sum(1 for a, b in zip(label_hit, text_hit) if not a and not b)
    print(f"\nagreement of the two independent views (n={len(labels)}):")
    print(f"  both label and text    {both:6d} ({100 * both / len(labels):.1f}%)")
    print(f"  label only             {only_lab:6d} ({100 * only_lab / len(labels):.1f}%)")
    print(f"  text only              {only_txt:6d} ({100 * only_txt / len(labels):.1f}%)")
    print(f"  neither                {neither:6d} ({100 * neither / len(labels):.1f}%)")
    print(f"  -> union reached by at least one view: "
          f"{100 * (both + only_lab + only_txt) / len(labels):.1f}%")

    print(f"\nmost frequent object nodes in product text (inflation check -- a generic node "
          f"topping this list means the number is carried by common words):")
    for name, c in node_freq.most_common(20):
        print(f"  {vocab[name]:24s} {c:6d} items")

    with (HERE / "title_overlap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (HERE / "title_overlap_nodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "items_in_product_text", "in_mit", "in_nyu"])
        for name, c in node_freq.most_common():
            w.writerow([vocab[name], c, int(name in per_corpus["mit"]),
                        int(name in per_corpus["nyu"])])
    (HERE / "title_overlap_summary.json").write_text(json.dumps(
        {"rows": rows, "agreement": {"both": both, "label_only": only_lab,
                                     "text_only": only_txt, "neither": neither}}, indent=2))
    print(f"\n-> title_overlap.csv, title_overlap_nodes.csv, title_overlap_summary.json")


if __name__ == "__main__":
    main()
