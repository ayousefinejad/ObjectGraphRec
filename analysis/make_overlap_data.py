#!/usr/bin/env python3
"""How much of the Amazon Home & Kitchen catalogue the MIT / NYU-Depth detected-object
vocabulary actually reaches.

    ~/hamedenv/bin/python make_overlap_data.py

No plot is drawn here -- this writes the measurement, plot_overlap.py renders it.

Why this measurement is the load-bearing one: `build_object_feat` (object-graph/ObjectGraph/
core.py:446-453) assigns every one of the 14,503 items an object embedding by matching the item's
label against the encoder's node list -- exactly by name if it can, otherwise by nearest MiniLM
neighbour. So "does the scene vocabulary overlap the catalogue" is not a rhetorical question
about two domains; it is a property of that assignment, item by item, and it is measurable here.

Both vocabularies are taken from the files the pipeline itself reads, never re-derived:

  object side  ObjectGraph.graph_data.load_scenes + build_cooccurrence, the same functions the
               encoder trains on, so node names and degrees are the trained graph's own. Verified
               against `nodes` in the default_fixed encoder checkpoint.
  item side    data/home_v2-2/5-core/raw_graph.txt -- one label per item, the exact file
               core.py:446 reads.

Tiers per item, mirroring core.py's own branch:
  exact  the label lower-cases to a node name  -> the item gets that object's trained embedding
  near   no exact hit, nearest-node cosine >= tau  -> a semantically close object's embedding
  far    nearest-node cosine < tau                 -> an arbitrary proxy; noise, labelled as such
tau is reported at 0.5/0.6/0.7 rather than fixed, so no headline rests on one threshold.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
if str(OG) not in sys.path:
    sys.path.insert(0, str(OG))

from ObjectGraph.graph_data import build_cooccurrence, load_scenes  # noqa: E402

DATA = OG / "data"
CKPT = DATA / "lattice-runs/default_fixed/encoder_default_fixed_seed0.pt"
RAW_GRAPH = DATA / "home_v2-2/5-core/raw_graph.txt"
TRAIN = DATA / "home_v2-2/5-core/train.json"
CORPORA = {"mit": DATA / "openai_mit.json", "nyu": DATA / "nyu-depth.json",
           "union": DATA / "scenes.json"}
# Visual Genome joins as a fourth corpus when it has been built. Optional so this script still
# runs on a checkout without it, and so the MIT/NYU/union numbers it prints stay comparable to
# the ones already in F17 -- if they move, the extension changed the measurement.
if (DATA / "visual_genome.json").exists():
    CORPORA["vg"] = DATA / "visual_genome.json"
TAUS = (0.5, 0.6, 0.7)


def corpus_nodes(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Node names, scene frequency, and distinct-neighbour degree for one scene corpus."""
    scenes = load_scenes({"scenes_path": str(path)})
    nodes, pair_counts, node_counts = build_cooccurrence(scenes)
    deg = np.zeros(len(nodes), dtype=np.int64)
    for (i, j) in pair_counts:
        deg[i] += 1
        deg[j] += 1
    return nodes, np.asarray(node_counts, dtype=np.int64), deg


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc_name = ckpt["cfg"]["text_encoder"]

    vocab = {}
    for name, path in CORPORA.items():
        nodes, freq, deg = corpus_nodes(path)
        vocab[name] = {"nodes": nodes, "freq": freq, "deg": deg}
        print(f"{name:6s} {path.name:22s} nodes={len(nodes):5d}")

    # The union corpus must BE the trained graph, or every number below describes a graph the
    # recommender never saw.
    assert set(vocab["union"]["nodes"]) == set(ckpt["nodes"]), \
        "scenes.json node set != default_fixed encoder checkpoint node set"
    print(f"       union node set == encoder checkpoint ({len(ckpt['nodes'])} nodes) OK")

    labels = [ln.strip() for ln in RAW_GRAPH.read_text(encoding="utf-8").splitlines()]
    assert len(labels) == 14503, len(labels)
    uniq = sorted(set(labels))
    lab_count = Counter(labels)
    print(f"items={len(labels)}  distinct labels={len(uniq)}")

    # Same encoder the fallback uses, on CPU: the GPU is busy with the training batch and this
    # is a few thousand short strings.
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(enc_name, device="cpu")
    lab_vec = F.normalize(st.encode(uniq, convert_to_tensor=True, batch_size=256), dim=1)

    per_corpus = {}
    for name, v in vocab.items():
        node_vec = F.normalize(st.encode(v["nodes"], convert_to_tensor=True, batch_size=256), dim=1)
        sim = lab_vec @ node_vec.T                      # [n_labels, n_nodes], cosine
        best = sim.max(dim=1)
        near_idx, near_cos = best.indices.numpy(), best.values.numpy()
        lower = {n.lower(): i for i, n in enumerate(v["nodes"])}
        exact_idx = np.array([lower.get(u.lower(), -1) for u in uniq], dtype=np.int64)
        # An exact hit is an exact hit regardless of cosine; core.py:449 short-circuits there.
        match_idx = np.where(exact_idx >= 0, exact_idx, near_idx)
        match_cos = np.where(exact_idx >= 0, 1.0, near_cos).astype(np.float32)
        per_corpus[name] = {"exact_idx": exact_idx, "near_idx": near_idx,
                            "near_cos": near_cos.astype(np.float32),
                            "match_idx": match_idx, "match_cos": match_cos}
        n_items = np.array([lab_count[u] for u in uniq])
        ex = exact_idx >= 0
        print(f"  {name:6s} exact labels={ex.sum():5d}  exact items={n_items[ex].sum():6d} "
              f"({100 * n_items[ex].sum() / len(labels):.1f}%)  "
              + "  ".join(f"near@{t}={n_items[~ex & (near_cos >= t)].sum():5d}" for t in TAUS))

    n_items_per_label = np.array([lab_count[u] for u in uniq], dtype=np.int64)

    # Item popularity = training interactions, the confound the stratified analysis has to
    # control for (covered labels are the catalogue's head).
    train = json.loads(TRAIN.read_text())
    pop = np.zeros(len(labels), dtype=np.int64)
    for items in train.values():
        for it in items:
            pop[it] += 1

    lab_to_row = {u: i for i, u in enumerate(uniq)}
    item_label_row = np.array([lab_to_row[l] for l in labels], dtype=np.int64)

    out = {
        "labels_uniq": np.array(uniq),
        "label_n_items": n_items_per_label,
        "item_label_row": item_label_row,          # 14503 -> row in labels_uniq
        "item_popularity": pop,
        "taus": np.array(TAUS, dtype=np.float32),
        "encoder": np.array(enc_name),
    }
    for name, v in vocab.items():
        out[f"{name}_nodes"] = np.array(v["nodes"])
        out[f"{name}_node_freq"] = v["freq"]
        out[f"{name}_node_deg"] = v["deg"]
        for k, arr in per_corpus[name].items():
            out[f"{name}_{k}"] = arr

    dest = HERE / "overlap_mit_nyu_amazon.npz"
    np.savez_compressed(dest, **out)
    print(f"-> {dest}  ({dest.stat().st_size / 1024:.0f} KB)")

    # Human-readable companion, union corpus, sorted by how many items ride on each label.
    u = per_corpus["union"]
    nodes = vocab["union"]["nodes"]
    rows = []
    for i in np.argsort(-n_items_per_label):
        ex = u["exact_idx"][i] >= 0
        rows.append({
            "label": uniq[i], "n_items": int(n_items_per_label[i]),
            "tier_tau0.6": "exact" if ex else ("near" if u["near_cos"][i] >= 0.6 else "far"),
            "matched_node": nodes[u["match_idx"][i]],
            "cosine": f"{u['match_cos'][i]:.4f}",
            "node_scene_freq": int(vocab["union"]["freq"][u["match_idx"][i]]),
            "node_degree": int(vocab["union"]["deg"][u["match_idx"][i]]),
            "in_mit": int(vocab["union"]["nodes"][u["match_idx"][i]] in set(vocab["mit"]["nodes"])),
            "in_nyu": int(vocab["union"]["nodes"][u["match_idx"][i]] in set(vocab["nyu"]["nodes"])),
        })
    csv_dest = HERE / "overlap_labels.csv"
    with csv_dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {csv_dest}  ({len(rows)} labels)")


if __name__ == "__main__":
    main()
