#!/usr/bin/env python3
"""Purely semantic assignment of Amazon Home & Kitchen item labels to MIT+NYU object-graph nodes.

    ~/hamedenv/bin/python semantic_assignment.py

Every Amazon label is assigned to its most similar graph node by cosine similarity in
sentence-transformers/all-MiniLM-L6-v2 -- the encoder the pipeline itself uses. Exact string
matching is NOT used as a criterion anywhere here: an exact pair simply scores ~1.0 on its own
merits, so the two are not mixed.
"""
from __future__ import annotations
import collections, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
CORE = OG / "data/home_v2-2/5-core"
CKPT = OG / "data/lattice-runs/default_fixed/encoder_default_fixed_seed0.pt"
THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ENC = "sentence-transformers/all-MiniLM-L6-v2"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
nodes = ck["nodes"]
labels_all = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
uniq = sorted(set(labels_all))
cnt = collections.Counter(labels_all)
items = np.array([cnt[u] for u in uniq])
N_NODES, N_LAB, N_ITEM = len(nodes), len(uniq), len(labels_all)

from sentence_transformers import SentenceTransformer
st = SentenceTransformer(ENC, device="cpu")
NV = F.normalize(st.encode(nodes, convert_to_tensor=True, batch_size=256), dim=1)
LV = F.normalize(st.encode(uniq, convert_to_tensor=True, batch_size=256), dim=1)
sim = (LV @ NV.T).numpy()                      # [labels, nodes]
best = sim.argmax(1)                           # most similar node per label
score = sim.max(1)                             # its similarity

print(f"encoder {ENC}")
print(f"graph nodes {N_NODES}   Amazon unique labels {N_LAB}   Amazon items {N_ITEM:,}\n")
print(f"{'thr':>5} | {'labels matched':>21} | {'items covered':>21} | {'nodes used':>19} | {'nodes unused':>19}")
print("-" * 98)
rows = []
for t in THR:
    m = score >= t
    used = len(set(best[m].tolist()))
    r = dict(threshold=t, labels=int(m.sum()), labels_pct=round(100*m.mean(), 1),
             items=int(items[m].sum()), items_pct=round(100*items[m].sum()/N_ITEM, 1),
             nodes_used=used, nodes_used_pct=round(100*used/N_NODES, 1),
             nodes_unused=N_NODES-used, nodes_unused_pct=round(100*(N_NODES-used)/N_NODES, 1))
    rows.append(r)
    print(f"{t:5.1f} | {r['labels']:6d} / {N_LAB}  {r['labels_pct']:5.1f}% | "
          f"{r['items']:7,} / {N_ITEM:,} {r['items_pct']:5.1f}% | "
          f"{r['nodes_used']:5d} {r['nodes_used_pct']:5.1f}% | {r['nodes_unused']:5d} {r['nodes_unused_pct']:5.1f}%")

print("\n=== below 0.5: labels whose nearest node is a poor match ===")
lo = score < 0.5
print(f"  labels {int(lo.sum())} / {N_LAB} ({100*lo.mean():.1f}%)   "
      f"items {int(items[lo].sum()):,} / {N_ITEM:,} ({100*items[lo].sum()/N_ITEM:.1f}%)")
for a, b in ((0.4, 0.5), (0.3, 0.4), (0.0, 0.3)):
    m = (score >= a) & (score < b)
    print(f"  {a:.1f} <= cos < {b:.1f}: {int(m.sum()):4d} labels, {int(items[m].sum()):5,} items "
          f"({100*items[m].sum()/N_ITEM:4.1f}%)")
worst = np.argsort(score)
print("\n  the 15 worst-matched labels, by cosine (these get an arbitrary object vector):")
print(f"    {'amazon label':24s} {'items':>6} {'assigned node':22s} {'cos':>6}")
for i in worst[:15]:
    print(f"    {uniq[i]:24s} {items[i]:6d} {nodes[best[i]]:22s} {score[i]:6.3f}")
print("\n  the 10 worst-matched labels weighted by item count (biggest blast radius):")
cand = [i for i in range(N_LAB) if score[i] < 0.5]
cand.sort(key=lambda i: -items[i])
for i in cand[:10]:
    print(f"    {uniq[i]:24s} {items[i]:6d} {nodes[best[i]]:22s} {score[i]:6.3f}")

mass = collections.Counter()
nlab = collections.Counter()
for i, b in enumerate(best):
    mass[int(b)] += int(items[i]); nlab[int(b)] += 1
print(f"\ntop matched nodes by Amazon items assigned (nearest-node assignment, no threshold):")
print(f"  {'#':>3} {'graph node':24s} {'items':>7} {'% cat':>6} {'labels':>7} {'mean cos':>9}  example labels")
for r, (j, m) in enumerate(mass.most_common(20), 1):
    idx = [i for i in range(N_LAB) if best[i] == j]
    ex = ", ".join(uniq[i] for i in sorted(idx, key=lambda i: -items[i])[:3])
    print(f"  {r:3d} {nodes[j]:24s} {m:7,} {100*m/N_ITEM:5.1f}% {nlab[j]:7d} "
          f"{np.mean([score[i] for i in idx]):9.3f}  {ex[:52]}")

with (HERE/"semantic_assignment.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["amazon_label","n_items","assigned_node","cosine"])
    for i in np.argsort(-items):
        w.writerow([uniq[i], int(items[i]), nodes[best[i]], f"{score[i]:.4f}"])
with (HERE/"semantic_assignment_nodes.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["graph_node","amazon_items","amazon_labels","mean_cosine"])
    for j, m in mass.most_common():
        idx=[i for i in range(N_LAB) if best[i]==j]
        w.writerow([nodes[j], m, nlab[j], f"{np.mean([score[i] for i in idx]):.4f}"])
print(f"\n-> semantic_assignment.csv ({N_LAB} labels)   semantic_assignment_nodes.csv ({len(mass)} nodes)")
print(f"   cosine distribution: min {score.min():.3f}  median {np.median(score):.3f}  "
      f"item-weighted median {np.median(np.repeat(score, items)):.3f}  max {score.max():.3f}")
