#!/usr/bin/env python3
"""Bidirectional semantic overlap between the object-graph nodes and the Amazon catalogue,
measured with all-MiniLM-L6-v2 -- the same sentence encoder the pipeline uses to build node
features and to resolve item labels (ObjectGraph/core.py:443).

Exact string matching answers "is this node literally an item label". It says nothing about the
537 nodes no label matches: `Countertop` scores 0 there, yet the catalogue is full of counter
accessories. Cosine in the encoder's own space is the measure the pipeline actually acts on, so
it is the fairer test of whether the two vocabularies describe the same world.

Both directions are reported because they answer different questions:
  node -> label   how much of the GRAPH is about things the catalogue sells
  label -> node   how much of the CATALOGUE the graph can describe   (this is F17's near tier)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
CORE = OG / "data/home_v2-2/5-core"
CKPT = OG / "data/lattice-runs/default_fixed/encoder_default_fixed_seed0.pt"
THR = (0.5, 0.6, 0.7, 0.8)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
nodes, enc_name = ck["nodes"], ck["cfg"]["text_encoder"]
labels_all = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
uniq = sorted(set(labels_all))
import collections
n_items = collections.Counter(labels_all)
print(f"encoder: {enc_name}   nodes: {len(nodes)}   distinct item labels: {len(uniq)}")

from sentence_transformers import SentenceTransformer
st = SentenceTransformer(enc_name, device="cpu")
NV = F.normalize(st.encode(nodes, convert_to_tensor=True, batch_size=256), dim=1)
LV = F.normalize(st.encode(uniq, convert_to_tensor=True, batch_size=256), dim=1)
sim = (NV @ LV.T).numpy()                       # [n_nodes, n_labels]

lower_lab = {u.lower(): i for i, u in enumerate(uniq)}
node_exact = np.array([lower_lab.get(n.lower(), -1) for n in nodes])
n2l = sim.max(1); n2l_arg = sim.argmax(1)
l2n = sim.max(0); l2n_arg = sim.argmax(0)
items = np.array([n_items[u] for u in uniq])

print("\n=== direction 1: node -> nearest Amazon label  (is the graph about sellable things?) ===")
print(f"  {'threshold':>12s} {'nodes':>7s} {'share':>7s}")
print(f"  {'exact match':>12s} {int((node_exact>=0).sum()):7d} {100*(node_exact>=0).mean():6.1f}%")
for t in THR:
    print(f"  {'cos >= '+str(t):>12s} {int((n2l>=t).sum()):7d} {100*(n2l>=t).mean():6.1f}%")
print(f"  median nearest-label cosine: {np.median(n2l):.3f}")

print("\n=== direction 2: Amazon label -> nearest node  (can the graph describe the catalogue?) ===")
lab_exact = np.array([1 if u.lower() in {n.lower() for n in nodes} else 0 for u in uniq]).astype(bool)
print(f"  {'threshold':>12s} {'labels':>7s} {'items':>8s} {'share of 14,503':>16s}")
print(f"  {'exact match':>12s} {int(lab_exact.sum()):7d} {int(items[lab_exact].sum()):8d} {100*items[lab_exact].sum()/items.sum():15.1f}%")
for t in THR:
    m = l2n >= t
    print(f"  {'cos >= '+str(t):>12s} {int(m.sum()):7d} {int(items[m].sum()):8d} {100*items[m].sum()/items.sum():15.1f}%")
print(f"  median nearest-node cosine: {np.median(l2n):.3f}   item-weighted: {np.median(np.repeat(l2n, items)):.3f}")

print("\n=== the 537 nodes no label matches exactly: how close are they semantically? ===")
un = np.where(node_exact < 0)[0]
for t in THR:
    print(f"  of {len(un)} unmatched nodes, {int((n2l[un]>=t).sum()):4d} ({100*(n2l[un]>=t).mean():4.1f}%) have a label within cos {t}")
order = un[np.argsort(-n2l[un])]
print("\n  closest (graph node -> nearest label, cosine):")
for j in order[:10]:
    print(f"    {nodes[j]:22s} -> {uniq[n2l_arg[j]]:22s} {n2l[j]:.3f}")
print("\n  most isolated from the catalogue:")
for j in order[-10:]:
    print(f"    {nodes[j]:22s} -> {uniq[n2l_arg[j]]:22s} {n2l[j]:.3f}")
np.savez_compressed(HERE/"semantic_overlap.npz", nodes=np.array(nodes), labels=np.array(uniq),
                    node_to_label_cos=n2l, node_to_label_idx=n2l_arg,
                    label_to_node_cos=l2n, label_to_node_idx=l2n_arg,
                    label_n_items=items, node_exact=node_exact)
print(f"\n-> {HERE/'semantic_overlap.npz'}")
