#!/usr/bin/env python3
"""Bundle everything needed for the Fig.6-style t-SNE (user stars + interacted-item dots) into
one .npz, comparing LATTICE's dense (GraphSAGE-consumed) vs GAT item-graph propagation.

    ~/hamedenv/bin/python make_tsne_recommender_data.py

No plot is drawn here and no t-SNE is fitted -- raw best-epoch user/item embeddings only, plus
the same 10 sampled users' train-interaction edges, so the two panels can be drawn identically
to the reference figure in whatever environment renders it.

Both checkpoints are the same LATTICE run (openai_mit, seed 0) differing ONLY in item_prop
(dense vs gat), so the recommender, dataset, and users are identical -- only the item-graph
propagation operator differs. R@20 at save: dense 0.04371 (epoch 120), gat 0.04306 (epoch 95).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent           # top-level objectgraph-eval/, this file's home
OG = HERE.parent / "object-graph"
if str(OG) not in sys.path:
    sys.path.insert(0, str(OG))

# The training runs wrote checkpoints relative to their cwd (object-graph/), landing them in a
# second, nested objectgraph-eval/ under object-graph/ -- distinct from this script's own
# directory. Read from there; write outputs to HERE, alongside every other artifact in this study.
EMB = OG / "objectgraph-eval" / "embeddings"
N_USERS = 10
MIN_ITEMS, MAX_ITEMS = 4, 10   # keeps the plot legible, matching the reference figure's density


def load_train_items():
    """uid -> [item ids], read directly off train.json -- avoids importing utility.load_data,
    which parses sys.argv at import time (Models.py / main.py convention) and would need a fake
    argv here for no benefit."""
    import json
    d = json.loads((OG / "data/lattice-runs/openai_mit/5-core/train.json").read_text())
    return {int(k): v for k, v in d.items()}


def main():
    dense = torch.load(EMB / "dense_seed0.pt", map_location="cpu", weights_only=False)
    gat = torch.load(EMB / "gat_seed0.pt", map_location="cpu", weights_only=False)
    assert dense["dataset"] == gat["dataset"] and dense["seed"] == gat["seed"], \
        "the two checkpoints are not the same recommender run -- panels would not be comparable"

    train_items = load_train_items()
    eligible = [u for u, items in train_items.items() if MIN_ITEMS <= len(items) <= MAX_ITEMS]
    print(f"users with {MIN_ITEMS}-{MAX_ITEMS} train items: {len(eligible)} of {len(train_items)}")

    random.seed(0)
    users = sorted(random.sample(eligible, N_USERS))
    edges_u, edges_i = [], []
    item_set = set()
    for u in users:
        for it in train_items[u]:
            edges_u.append(u)
            edges_i.append(it)
            item_set.add(it)
    items = sorted(item_set)
    print(f"sampled {len(users)} users, {len(items)} distinct interacted items, "
          f"{len(edges_u)} edges")

    ui = {u: i for i, u in enumerate(users)}
    ii = {it: i for i, it in enumerate(items)}
    edge_u_idx = np.array([ui[u] for u in edges_u], dtype=np.int32)
    edge_i_idx = np.array([ii[it] for it in edges_i], dtype=np.int32)

    out = {
        "user_ids": np.array(users, dtype=np.int64),
        "item_ids": np.array(items, dtype=np.int64),
        "edge_user_idx": edge_u_idx,     # index into user_ids
        "edge_item_idx": edge_i_idx,     # index into item_ids, paired with edge_user_idx
        "ua_dense": dense["ua"].numpy()[users].astype(np.float32),
        "ia_dense": dense["ia"].numpy()[items].astype(np.float32),
        "ua_gat": gat["ua"].numpy()[users].astype(np.float32),
        "ia_gat": gat["ia"].numpy()[items].astype(np.float32),
        "meta_json": np.array(
            f'{{"dataset": "{dense["dataset"]}", "seed": {dense["seed"]}, '
            f'"dense_epoch": {dense["epoch"]}, "gat_epoch": {gat["epoch"]}, '
            f'"dense_r20": 0.04371, "gat_r20": 0.04306, '
            f'"n_users": {len(users)}, "n_items": {len(items)}}}'),
    }
    dest = HERE / "tsne_lattice_users_dense_vs_gat.npz"
    np.savez_compressed(dest, **out)
    print(f"-> {dest}  ({dest.stat().st_size/1024:.0f} KB)")
    print("   keys: user_ids, item_ids, edge_user_idx, edge_item_idx,")
    print("         ua_dense[10,64], ia_dense[N,64], ua_gat[10,64], ia_gat[N,64], meta_json")


if __name__ == "__main__":
    main()
