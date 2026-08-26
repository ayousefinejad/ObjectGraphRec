#!/usr/bin/env python3
"""Bundle everything needed to draw a t-SNE comparing the GraphSAGE and GAT object-graph
encoders, into one .npz. No plot is produced and no t-SNE is fitted here -- the projection is
left to whatever environment draws the figure.

    ~/hamedenv/bin/python make_tsne_data.py

Both encoders are the `converged` recipe (1000 epochs, tau=0.5, lr=1e-3, 2 layers, same corpus,
same seed) and differ ONLY in the backbone, so a difference in the projection is attributable to
the encoder and nothing else. Intrinsic scores: SAGE 0.7886, GAT 0.8098 test AUC.

Node order is identical across the two arrays (verified, not assumed), so row i is the same
object in both -- required for a paired figure, and for drawing per-node connectors between
panels if wanted.

Room labels are the repo's own weak anchor rule (ObjectGraph/eval.py:ROOM_ANCHORS) and are
DIAGNOSTIC ONLY -- they are derived from object identity, so clustering by room is partly true
by construction. Use them to colour points, not to claim the encoder discovered rooms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
if str(OG) not in sys.path:
    sys.path.insert(0, str(OG))

ARMS = [("sage", OG / "data/lattice-runs/converged/encoder_converged_seed0.pt"),
        ("gat", OG / "data/lattice-runs/converged_gat/encoder_converged_gat_seed0.pt")]


def main() -> None:
    out, nodes_ref, meta = {}, None, {}
    for name, p in ARMS:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        emb = ck["final_embeddings"]
        emb = (emb if torch.is_tensor(emb) else torch.tensor(emb)).float().numpy()
        nodes = list(ck["nodes"])
        if nodes_ref is None:
            nodes_ref = nodes
        elif nodes != nodes_ref:
            raise SystemExit("node order differs between arms -- rows would not be comparable")
        out[f"emb_{name}"] = emb.astype(np.float32)
        meta[name] = {"backbone": ck["cfg"]["backbone"], "epochs": ck["cfg"]["epochs"],
                      "temperature": ck["cfg"]["temperature"], "lr": ck["cfg"]["lr"],
                      "num_layers": ck["cfg"]["num_layers"], "seed": ck["cfg"]["seed"]}
        print(f"  {name:5s} {emb.shape}  backbone={ck['cfg']['backbone']} "
              f"epochs={ck['cfg']['epochs']} layers={ck['cfg']['num_layers']}")

    # ---- weak room labels, for colouring ----------------------------------------------
    import json
    from ObjectGraph.eval import ROOM_ANCHORS, weak_room_labels
    scenes = json.loads((OG / "data/scenes.json").read_text())
    room_idx = weak_room_labels(scenes, nodes_ref)
    rooms = list(ROOM_ANCHORS)
    # weak_room_labels returns -1 (or similar) where a node never appeared in an unambiguous
    # scene; keep that as its own class rather than silently dropping those nodes.
    room_names = np.array([rooms[i] if 0 <= i < len(rooms) else "unlabelled" for i in room_idx])

    # ---- node degree in the co-occurrence graph, for sizing ---------------------------
    deg = np.zeros(len(nodes_ref), dtype=np.int64)
    n2i = {n: i for i, n in enumerate(nodes_ref)}
    for sc in scenes:
        for a in set(sc):
            if a in n2i:
                deg[n2i[a]] += 1

    out.update({
        "nodes": np.array(nodes_ref, dtype=object),
        "room": room_names,
        "scene_freq": deg,
        "meta_json": np.array(json.dumps(meta)),
    })
    dest = HERE / "tsne_encoder_sage_vs_gat.npz"
    np.savez_compressed(dest, **out)

    from collections import Counter
    print(f"\n  nodes={len(nodes_ref)}  rooms={dict(Counter(room_names))}")
    print(f"  -> {dest}  ({dest.stat().st_size/1024:.0f} KB)")
    print("     keys: emb_sage[N,64], emb_gat[N,64], nodes[N], room[N], scene_freq[N], meta_json")


if __name__ == "__main__":
    main()
