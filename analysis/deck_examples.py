#!/usr/bin/env python3
"""Real replacements for the deck's illustrative numbers on 8c (fused similarity) and 8d
(ranked list).

    ~/hamedenv/bin/python deck_examples.py

8c wants a worked per-pair example of modality fusion. This computes it the way Models.py does:
per-modality cosine on the SAME projected features the model fuses, combined with the alpha the
model actually learned -- so the arithmetic on the slide is the model's arithmetic.

8d wants a ranked list. This takes a real user, ranks with the dumped image+text embeddings and
again with the dumped image+text+object embeddings (same seed, same split, same epoch-selection
rule), and shows an item the object modality pulled into the top-20.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
CORE = OG / "data/home_v2-2/5-core"
VAR = OG / "data/lattice-runs/default_fixed"
EMB = OG / "objectgraph-eval/embeddings"

# Measured across 20 runs by deck_facts.py; the fusion parameters barely leave uniform.
ALPHA = np.array([0.3453, 0.3315, 0.3232])       # [image, text, object]


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def main():
    labels = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
    titles = [l.strip()[:110] for l in (CORE / "raw_text.txt").read_text(encoding="utf-8").splitlines()]
    img = norm(np.load(VAR / "image_feat.npy").astype(np.float32))
    txt = norm(np.load(VAR / "text_feat.npy").astype(np.float32))
    obj = norm(np.load(VAR / "object_feat.npy").astype(np.float32))

    print("=" * 78)
    print("8c  FUSED SIMILARITY -- real numbers")
    print("=" * 78)
    print(f"alpha (measured, softmax of modal_weight at end of training): "
          f"image {ALPHA[0]:.4f}  text {ALPHA[1]:.4f}  object {ALPHA[2]:.4f}\n")
    print("NOTE ON THE CURRENT SLIDE: 0.60 / 0.50 / 0.80 -> 0.675 implies weights")
    print("[0.25, 0.25, 0.50] (0.25*0.60 + 0.25*0.50 + 0.50*0.80 = 0.675). That says the model")
    print("learned to weight objects double. It does not: the measured alpha is essentially")
    print("uniform, and object is the SMALLEST of the three. Under the real alpha the same three")
    print(f"inputs fuse to {float(ALPHA @ [0.60, 0.50, 0.80]):.4f}, not 0.675.\n")

    # A pair the object modality is decisive for. Deliberately requires DIFFERENT labels: two
    # items sharing a label have object cosine exactly 1.0 by construction (argmax assignment
    # gives them the same vector), which makes a trivial slide -- "they matched because they are
    # the same word". Requiring different labels and cosine < 0.995 shows what the co-occurrence
    # graph actually contributes: two objects that appear in the same rooms, which neither the
    # product photo nor the title states.
    lab_arr = np.array(labels)
    rng = np.random.default_rng(1)
    best = []
    for i in rng.permutation(len(labels))[:600]:
        o, im = obj[i] @ obj.T, img[i] @ img.T
        m = (o > 0.75) & (o < 0.995) & (im < 0.35) & (lab_arr != labels[i])
        for j in np.where(m)[0][:40]:
            best.append((float(o[j] - im[j]), int(i), int(j)))
    best.sort(reverse=True)
    _, i, j = best[0]
    sims = np.array([float(img[i] @ img[j]), float(txt[i] @ txt[j]), float(obj[i] @ obj[j])])
    print(f"item {i}  label '{labels[i]}'\n    {titles[i]}")
    print(f"item {j}  label '{labels[j]}'\n    {titles[j]}")
    print(f"\n  cosine   image {sims[0]:.4f}   text {sims[1]:.4f}   object {sims[2]:.4f}")
    print(f"  fused    {ALPHA[0]:.4f}*{sims[0]:.4f} + {ALPHA[1]:.4f}*{sims[1]:.4f} + "
          f"{ALPHA[2]:.4f}*{sims[2]:.4f}  =  {float(ALPHA @ sims):.4f}")
    print(f"  without the object modality (image+text renormalised to 0.5/0.5): "
          f"{float(0.5 * sims[0] + 0.5 * sims[1]):.4f}")
    print(f"  -> the object modality moves this pair by "
          f"{float(ALPHA @ sims) - float(0.5 * sims[0] + 0.5 * sims[1]):+.4f}")

    print("\n" + "=" * 78)
    print("8d  RANKED LIST -- real, from the two dumped models")
    print("=" * 78)
    ctrl = torch.load(EMB / "cov_control_seed0.pt", map_location="cpu", weights_only=False)
    noobj = torch.load(EMB / "cov_noobj_seed0.pt", map_location="cpu", weights_only=False)
    train = {int(k): v for k, v in json.loads((CORE / "train.json").read_text()).items() if v}
    test = {int(k): v for k, v in json.loads((CORE / "test.json").read_text()).items() if v}

    def top20(ck, u):
        s = ck["ua"][u].numpy() @ ck["ia"].numpy().T
        s[train.get(u, [])] = -np.inf
        t = np.argpartition(-s, 20)[:20]
        return t[np.argsort(-s[t])]

    # A user whose held-out item the object modality rescued: out of top-20 without it, in with it.
    pick = None
    for u in list(test)[:6000]:
        if len(train.get(u, [])) < 3:
            continue
        gt = test[u][0]
        a, b = top20(noobj, u), top20(ctrl, u)
        if gt in b and gt not in a:
            pick = (u, gt, a, b)
            break
    if pick is None:
        print("no rescued example found in the first 6,000 users")
        return
    u, gt, a, b = pick
    print(f"user {u}   history ({len(train[u])} items): "
          + "; ".join(f"{labels[k]}" for k in train[u][:6]))
    print(f"held-out test item: {gt}  '{labels[gt]}'  --  {titles[gt][:80]}\n")
    for name, lst in (("image + text", a), ("image + text + object", b)):
        r = list(lst).index(gt) + 1 if gt in lst else None
        print(f"  {name:22s} top-8: " + ", ".join(f"{labels[k]}" for k in lst[:8]))
        print(f"  {'':22s} held-out item rank: {r if r else 'not in top-20'}")
    print("\nSame seed, same split, same early-stopping rule; the two runs differ only in whether")
    print("the object modality is present.")


if __name__ == "__main__":
    main()
