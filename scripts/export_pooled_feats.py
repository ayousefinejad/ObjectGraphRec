#!/usr/bin/env python3
"""Export object_feat.npy with soft multi-node pooling instead of hard argmax assignment.

    python scripts/export_pooled_feats.py --encoder data/lattice-runs/openai_mit/encoder_default_fixed_seed0.pt \
                                          --out /tmp/object_feat_pooled.npy --topk 3

Why this exists
---------------
`build_object_feat` maps each of the 14,503 items to exactly ONE object node -- an exact
label match when there is one, else the single nearest node by text similarity. Measured
consequence on the MIT (OpenAI) encoder:

    14,503 items  ->  1,815 distinct item labels  ->  510 distinct object vectors

The argmax discards 72% of the resolution the labels already carry, because many different
labels share a nearest node. The object graph then has only 510 distinct neighbourhoods
against 14,196 for image and 14,492 for text, and reaches 21.5% of items as kNN targets.
That is the binding constraint every downstream arm has run into.

This is Eq. (3)'s POOL in spirit: an item is a similarity-weighted blend of the object nodes
it is near, not one hard label. It uses only data already on disk -- no new detector, no new
corpus, no retraining. Measured: 1,815 unique vectors, 1,814 distinct neighbourhoods, 41.5%
of items reachable. topk=3 already saturates at the 1,815 ceiling (= the number of distinct
labels), so larger topk buys nothing.

Writes a NEW file. Never touches data/home_v2-2/object_feat.npy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="encoder checkpoint with nodes + embeddings")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default="data/home_v2-2/5-core/raw_graph.txt")
    p.add_argument("--topk", type=int, default=3, help="nodes pooled per item")
    p.add_argument("--tau", type=float, default=0.05,
                   help="softmax temperature over node similarity; lower = closer to argmax")
    p.add_argument("--exact-hard", type=int, default=1,
                   help="1: an exact label match still takes that node alone, as today")
    a = p.parse_args()

    out = Path(a.out)
    frozen = (ROOT / "data/home_v2-2/object_feat.npy").resolve()
    if out.resolve() == frozen:
        raise SystemExit("refusing to overwrite the frozen artifact %s" % frozen)

    from sentence_transformers import SentenceTransformer

    ck = torch.load(a.encoder, map_location="cpu", weights_only=False)
    nodes, emb = ck["nodes"], ck["final_embeddings"]
    emb = (emb if torch.is_tensor(emb) else torch.tensor(emb)).float()
    labels = [l.strip() for l in Path(a.labels).read_text(encoding="utf-8").splitlines()]
    uniq = sorted(set(labels))
    print(f"items={len(labels)}  distinct labels={len(uniq)}  nodes={len(nodes)}")

    enc = SentenceTransformer(ck.get("cfg", {}).get("text_encoder", "all-MiniLM-L6-v2"),
                              device="cpu")
    qv = F.normalize(torch.tensor(enc.encode(uniq)), dim=1)
    nv = F.normalize(torch.tensor(enc.encode(list(nodes))), dim=1)
    sim = qv @ nv.T
    lower = {n.lower(): i for i, n in enumerate(nodes)}

    per_label = {}
    n_exact = 0
    for r, u in enumerate(uniq):
        hit = lower.get(u.lower())
        if hit is not None and a.exact_hard:
            # Preserve today's behaviour where the label IS a node: pooling a known-correct
            # node with its neighbours would blur a match that is already exact.
            per_label[u] = emb[hit]
            n_exact += 1
        else:
            v, i = torch.topk(sim[r], a.topk)
            w = torch.softmax(v / a.tau, 0)
            per_label[u] = (emb[i] * w.unsqueeze(1)).sum(0)
    arr = F.normalize(torch.stack([per_label[l] for l in labels]), dim=1).numpy().astype(np.float32)

    uniq_vec = len(np.unique(arr.round(6), axis=0))
    print(f"exact-matched labels kept hard: {n_exact}")
    print(f"-> {out}  shape={arr.shape}  unique_vectors={uniq_vec}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)


if __name__ == "__main__":
    main()
