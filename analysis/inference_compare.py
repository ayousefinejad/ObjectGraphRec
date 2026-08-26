#!/usr/bin/env python3
"""Side-by-side inference: LATTICE (image+text) vs LATTICE + object graph.

    ~/hamedenv/bin/python inference_compare.py [--users 8] [--out INFERENCE_COMPARE.md]

Both models are the SAME run configuration on the same split, differing only in whether the
object modality is present -- `cov_noobj_seed*.pt` and `cov_control_seed*.pt`, the best-epoch
user/item embeddings dumped during the 9-run coverage batch. Scoring reproduces
utility/batch_test.py exactly (score = ua @ ia.T, candidates = all items minus that user's TRAIN
items), and the offline global Recall@20 was verified against every run's logged value to <5e-6.

Two halves, because either alone would mislead:
  * aggregate  -- what changes across all 59,251 users, so no cherry-picked example carries the
                  argument. Reported over 3 seeds, and separated into items the object
                  vocabulary covers vs does not (the F18 strata), since that is where the
                  mechanism predicts the difference should live.
  * qualitative-- a handful of real users' top-10 lists, chosen by a stated rule rather than by
                  eye, with the held-out item marked.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
EMB = OG / "objectgraph-eval" / "embeddings"
CORE = OG / "data/home_v2-2/5-core"
SEEDS = (0, 1, 2)
K = 10


def top_k(ua, ia, train, users, k):
    out = np.empty((len(users), k), dtype=np.int32)
    for s in range(0, len(users), 2048):
        ch = users[s:s + 2048]
        sc = ua[ch] @ ia.T
        for r, u in enumerate(ch):
            ti = train.get(u)
            if ti:
                sc[r, ti] = -np.inf
        part = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.take_along_axis(sc, part, 1).argsort(axis=1)[:, ::-1]
        out[s:s + 2048] = np.take_along_axis(part, order, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "INFERENCE_COMPARE.md"))
    a = ap.parse_args()

    labels = [l.strip() for l in (CORE / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
    titles = [l.strip() for l in (CORE / "raw_text.txt").read_text(encoding="utf-8").splitlines()]
    train = {int(k): v for k, v in json.loads((CORE / "train.json").read_text()).items() if v}
    test = {int(k): v for k, v in json.loads((CORE / "test.json").read_text()).items() if v}
    users = list(test.keys())

    ov = np.load(HERE / "overlap_mit_nyu_amazon.npz")
    row = ov["item_label_row"]
    exact = (ov["union_exact_idx"] >= 0)[row]
    near = (~exact) & (ov["union_near_cos"][row] >= 0.6)
    tier = np.where(exact, "exact", np.where(near, "near", "unreached"))

    agg, per_seed_top = [], {}
    for seed in SEEDS:
        res = {}
        for arm, tag in (("base", "cov_noobj"), ("obj", "cov_control")):
            ck = torch.load(EMB / f"{tag}_seed{seed}.pt", map_location="cpu", weights_only=False)
            res[arm] = top_k(ck["ua"].numpy(), ck["ia"].numpy(), train, users, 20)
        per_seed_top[seed] = res
        rescued = lost = both = neither = 0
        rank_b, rank_o = [], []
        jac = []
        by_tier = {t: [0, 0, 0] for t in ("exact", "near", "unreached")}   # [n, in_base, in_obj]
        for r, u in enumerate(users):
            gt = set(test[u])
            b20, o20 = res["base"][r], res["obj"][r]
            bs, os_ = set(b20.tolist()), set(o20.tolist())
            jac.append(len(set(b20[:K].tolist()) & set(o20[:K].tolist())) / K)
            for g in gt:
                inb, ino = g in bs, g in os_
                t = tier[g]
                by_tier[t][0] += 1
                by_tier[t][1] += inb
                by_tier[t][2] += ino
                if ino and not inb:
                    rescued += 1
                elif inb and not ino:
                    lost += 1
                elif inb and ino:
                    both += 1
                    rank_b.append(int(np.where(b20 == g)[0][0]) + 1)
                    rank_o.append(int(np.where(o20 == g)[0][0]) + 1)
                else:
                    neither += 1
        agg.append(dict(seed=seed, rescued=rescued, lost=lost, both=both, neither=neither,
                        jac=float(np.mean(jac)),
                        rank_b=float(np.mean(rank_b)), rank_o=float(np.mean(rank_o)),
                        by_tier={t: v[:] for t, v in by_tier.items()}))
        print(f"seed {seed}: rescued {rescued}  lost {lost}  both {both}  "
              f"top-{K} overlap {np.mean(jac):.3f}")

    L = ["# Inference comparison — LATTICE vs LATTICE + object graph", "",
         "Same recommender, same split, same seeds; the two models differ only in whether the "
         "object modality is present. Scoring reproduces `utility/batch_test.py` exactly "
         f"(59,251 users, 60,178 held-out items, top-20 candidates = all items minus the user's "
         "train set).", "", "## 1. What changes across all users", "",
         "| Seed | Held-out item retrieved by | | | | Top-10 list overlap |",
         "|---|---|---|---|---|---|",
         "| | **+object only** (rescued) | **image+text only** (lost) | both | neither | Jaccard |"]
    for r in agg:
        L.append(f"| {r['seed']} | **{r['rescued']}** | {r['lost']} | {r['both']} | "
                 f"{r['neither']:,} | {r['jac']:.3f} |")
    m = {k: np.mean([r[k] for r in agg]) for k in ("rescued", "lost", "both", "jac", "rank_b", "rank_o")}
    L += ["", f"Mean over 3 seeds: **{m['rescued']:.0f} items rescued** by the object modality "
              f"against **{m['lost']:.0f} lost** — a net gain of {m['rescued'] - m['lost']:.0f} "
              f"retrieved held-out items per seed.", "",
          f"The two models' top-10 lists overlap by only **{100 * m['jac']:.1f}%** on average, so "
          "they are not small perturbations of one another: adding the object modality reorders "
          "most users' recommendations, and the net metric gain is the residue of many "
          "compensating changes.", "",
          f"Among items both models retrieve, mean rank is {m['rank_b']:.2f} (image+text) vs "
          f"{m['rank_o']:.2f} (+object) — the object modality also *promotes* items it does not "
          "newly retrieve.", "",
          "## 2. Where the change lands (F18 coverage tiers)", "",
          "| Tier | Held-out items | Retrieved, image+text | Retrieved, +object | Δ |",
          "|---|---|---|---|---|"]
    for t in ("exact", "near", "unreached"):
        n = agg[0]["by_tier"][t][0]
        b = np.mean([r["by_tier"][t][1] for r in agg])
        o = np.mean([r["by_tier"][t][2] for r in agg])
        L.append(f"| {t} | {n:,} | {b:.0f} ({100 * b / n:.2f}%) | {o:.0f} ({100 * o / n:.2f}%) | "
                 f"**{o - b:+.0f}** ({100 * (o - b) / b:+.1f}%) |")

    # Qualitative: seed 0, users whose held-out item the object modality rescued, sampled under a
    # fixed seed from ALL such users rather than picked by eye.
    res = per_seed_top[0]
    cand = []
    for r, u in enumerate(users):
        if len(train.get(u, [])) < 3:
            continue
        gt = test[u][0]
        if gt in set(res["obj"][r].tolist()) and gt not in set(res["base"][r].tolist()):
            cand.append((r, u, gt))
    random.Random(0).shuffle(cand)
    L += ["", f"## 3. Worked examples", "",
          f"Seed 0. Sampled uniformly (seed 0) from the {len(cand):,} users whose held-out item "
          "the object modality rescued into the top-20 — a stated rule, not hand-picked. "
          "`>>` marks the held-out item.", ""]
    for r, u, gt in cand[:a.users]:
        hist = ", ".join(labels[i] for i in train[u][:6])
        truth = test[u]                      # the FULL held-out set, not just the sampled item
        L += [f"**User {u}**", "",
              f"| | |", "|---|---|",
              f"| Training history ({len(train[u])} items) | {hist} |",
              f"| **Ground truth** (held out, {len(truth)}) | "
              + "; ".join(f"**{labels[g]}** — {titles[g][:60]}" for g in truth) + " |", "",
              "| Rank | image + text | ✓ | image + text + object | ✓ |",
              "|---|---|---|---|---|"]
        tset = set(truth)
        for i in range(K):
            b, o = int(res["base"][r][i]), int(res["obj"][r][i])
            bm = "**GT**" if b in tset else ""
            om = "**GT**" if o in tset else ""
            L.append(f"| {i + 1} | {labels[b]} | {bm} | {labels[o]} | {om} |")
        rb = np.where(res["base"][r] == gt)[0]
        ro = np.where(res["obj"][r] == gt)[0]
        L += ["", f"  held-out item rank: image+text "
                  f"{int(rb[0]) + 1 if len(rb) else 'not in top-20'}, "
                  f"+object {int(ro[0]) + 1 if len(ro) else 'not in top-20'}", ""]
    Path(a.out).write_text("\n".join(L))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
