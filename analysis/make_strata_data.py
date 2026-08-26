#!/usr/bin/env python3
"""Decompose LATTICE's object-modality gain over the coverage tiers from overlap_labels.

    ~/hamedenv/bin/python make_strata_data.py

The global result -- +7.1% R@20 from adding the object modality -- is an average over 14,503
items. If the object graph helps *because its vocabulary covers the catalogue*, the gain has to
sit on the items that vocabulary actually reaches, and the placebo arm (same features, rows
permuted) has to show nothing anywhere. That is what this script measures.

Ranking is recomputed offline from the dumped best-epoch embeddings, reproducing
utility/batch_test.py exactly: score = ua @ ia.T, candidates = all items minus that user's
TRAIN items (val items stay in, as in test_one_user), top-20 by score.

VALIDITY GATE: before any stratum number is reported, the offline global recall@10/@20 must
reproduce each run's own logged `test==` line. A reconstruction that cannot reproduce the
published metric cannot be trusted to decompose it, so a mismatch aborts.

Metric per stratum is per-test-item hit@20 -- of all (user, test item) pairs whose item is in
this stratum, the fraction where the item made that user's top-20. R@20 is a per-user ratio and
does not decompose over items; this does, and its overall mean is a legitimate item-level view
of the same ranking.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
EMB = OG / "objectgraph-eval" / "embeddings"     # training cwd was object-graph/
CORE = OG / "data/home_v2-2/5-core"
ARMS = ("noobj", "control", "shufobj")
SEEDS = (0, 1, 2)
TAU = 0.6
KS = (10, 20)
# The logged recall is printed rounded to 5 dp, so up to 5e-6 of any gap is the log's own
# rounding. 3e-5 leaves room for that and still fails loudly on a real ranking discrepancy --
# validated at 2e-6 against the existing dense_seed0 dump.
TOL = 3e-5

LOGGED = re.compile(r"test==\[.*?\], recall=\[([\d.]+), ([\d.]+)\]")


def logged_recall(arm: str, seed: int) -> tuple[float, float]:
    """recall@10/@20 from the run's last `test==` line -- the best-validation-epoch scores,
    which is the same epoch --dump_embeddings wrote."""
    text = (EMB / f"cov_{arm}_seed{seed}.log").read_text()
    hits = LOGGED.findall(text)
    if not hits:
        raise SystemExit(f"{arm} seed{seed}: no test== line in the log")
    return float(hits[-1][0]), float(hits[-1][1])


def rank_hits(ua: np.ndarray, ia: np.ndarray, train_items: dict, test_users: list,
              test_set: dict, kmax: int = 20, batch: int = 2048):
    """Top-kmax item ids per test user, with that user's train items removed from the pool."""
    out = np.empty((len(test_users), kmax), dtype=np.int32)
    for s in range(0, len(test_users), batch):
        chunk = test_users[s:s + batch]
        sc = ua[chunk] @ ia.T
        for r, u in enumerate(chunk):
            ti = train_items.get(u)
            if ti:
                sc[r, ti] = -np.inf
        # argpartition then sort just the head: full argsort of 14,503 x 2,048 is ~30x slower.
        part = np.argpartition(-sc, kmax, axis=1)[:, :kmax]
        ord_ = np.take_along_axis(sc, part, 1).argsort(axis=1)[:, ::-1]
        out[s:s + batch] = np.take_along_axis(part, ord_, 1)
    return out


def main() -> None:
    train = {int(k): v for k, v in json.loads((CORE / "train.json").read_text()).items() if v}
    test = {int(k): v for k, v in json.loads((CORE / "test.json").read_text()).items() if v}
    test_users = list(test.keys())
    n_items = sum(1 for _ in (CORE / "item_list.txt").read_text().splitlines())
    n_pairs = sum(len(v) for v in test.values())
    print(f"{len(test_users)} test users, {n_pairs} (user, test item) pairs, {n_items} items")

    ov = np.load(HERE / "overlap_mit_nyu_amazon.npz")
    row = ov["item_label_row"]
    ex = (ov["union_exact_idx"] >= 0)[row]
    near = (~ex) & (ov["union_near_cos"][row] >= TAU)
    tier = np.where(ex, "exact", np.where(near, "near", "far"))
    pop = ov["item_popularity"]
    # Deciles of training interactions -- the popularity control. Ranked rather than cut on raw
    # counts because the distribution is long-tailed enough that fixed-width bins would be empty.
    decile = np.searchsorted(np.quantile(pop, np.linspace(0.1, 0.9, 9)), pop, side="right")
    print("tier item counts:", {t: int((tier == t).sum()) for t in ("exact", "near", "far")})

    per_item = {}          # (arm, seed) -> (exposures, hits) over items
    globals_ = []
    for arm in ARMS:
        for seed in SEEDS:
            f = EMB / f"cov_{arm}_seed{seed}.pt"
            if not f.exists():
                print(f"  missing {f.name} -- skipping")
                continue
            ck = torch.load(f, map_location="cpu", weights_only=False)
            ua, ia = ck["ua"].numpy(), ck["ia"].numpy()
            top = rank_hits(ua, ia, train, test_users, test, kmax=max(KS))

            rec = {k: 0.0 for k in KS}
            expo = np.zeros(n_items, dtype=np.int64)
            hit20 = np.zeros(n_items, dtype=np.int64)
            for r, u in enumerate(test_users):
                pos = test[u]
                pset = set(pos)
                relevant = np.fromiter((int(i in pset) for i in top[r]), dtype=np.int8, count=max(KS))
                for k in KS:
                    rec[k] += relevant[:k].sum() / len(pos)
                expo[pos] += 1
                t20 = set(top[r].tolist())
                for i in pos:
                    if i in t20:
                        hit20[i] += 1
            rec = {k: v / len(test_users) for k, v in rec.items()}

            want10, want20 = logged_recall(arm, seed)
            d10, d20 = abs(rec[10] - want10), abs(rec[20] - want20)
            ok = d10 < TOL and d20 < TOL
            print(f"  {arm:8s} seed{seed}  R@10 {rec[10]:.5f} (log {want10}, d={d10:.2e})  "
                  f"R@20 {rec[20]:.5f} (log {want20}, d={d20:.2e})  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(
                    f"{arm} seed{seed}: offline ranking does not reproduce the logged metric "
                    f"(|d| >= {TOL}). Every stratified number below would be unverifiable.")
            per_item[(arm, seed)] = (expo, hit20)
            globals_.append({"arm": arm, "seed": seed, "recall@10": round(rec[10], 6),
                             "recall@20": round(rec[20], 6), "epoch": ck["epoch"]})

    if not per_item:
        raise SystemExit("no dumps found yet")

    expo0 = next(iter(per_item.values()))[0]
    assert all(np.array_equal(v[0], expo0) for v in per_item.values()), \
        "exposure counts differ between arms -- the test split is not being held fixed"

    with (HERE / "strata_hits.csv").open("w", newline="") as f:
        cols = ["item", "tier", "pop_decile", "train_pop", "exposures"] + \
               [f"hits20_{a}_s{s}" for a in ARMS for s in SEEDS if (a, s) in per_item]
        w = csv.writer(f)
        w.writerow(cols)
        keys = [(a, s) for a in ARMS for s in SEEDS if (a, s) in per_item]
        for i in np.where(expo0 > 0)[0]:
            w.writerow([i, tier[i], int(decile[i]), int(pop[i]), int(expo0[i])] +
                       [int(per_item[k][1][i]) for k in keys])
    print(f"-> {HERE / 'strata_hits.csv'}  ({int((expo0 > 0).sum())} items with test exposure)")

    rows = []
    keys = [(a, s) for a in ARMS for s in SEEDS if (a, s) in per_item]
    for stratum, mask in ([(t, tier == t) for t in ("exact", "near", "far")] +
                          [("all", np.ones_like(ex))] +
                          [(f"decile{d}", decile == d) for d in range(10)] +
                          [(f"{t}|decile{d}", (tier == t) & (decile == d))
                           for t in ("exact", "far") for d in range(10)]):
        m = mask & (expo0 > 0)
        e = expo0[m].sum()
        if e == 0:
            continue
        r = {"stratum": stratum, "items": int(m.sum()), "exposures": int(e)}
        for a, s in keys:
            r[f"hit20_{a}_s{s}"] = round(float(per_item[(a, s)][1][m].sum() / e), 6)
        rows.append(r)
    with (HERE / "strata_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {HERE / 'strata_summary.csv'}  ({len(rows)} strata)")

    with (HERE / "strata_global.json").open("w") as f:
        json.dump(globals_, f, indent=2)
    print(f"-> {HERE / 'strata_global.json'}")

    # Headline, printed so a run of this script is self-describing.
    seeds_done = sorted({s for a, s in keys if a == "control"} & {s for a, s in keys if a == "noobj"})
    print(f"\npaired control - noobj, per-test-item hit@20 (seeds {seeds_done}):")
    for stratum in ("all", "exact", "near", "far"):
        m = (np.ones_like(ex) if stratum == "all" else tier == stratum) & (expo0 > 0)
        e = expo0[m].sum()
        ds = [float((per_item[("control", s)][1][m].sum() - per_item[("noobj", s)][1][m].sum()) / e)
              for s in seeds_done]
        base = np.mean([per_item[("noobj", s)][1][m].sum() / e for s in seeds_done])
        print(f"  {stratum:6s} n={int(e):6d}  base={base:.5f}  d={np.mean(ds):+.5f} "
              f"({100 * np.mean(ds) / base:+.1f}%)  per-seed {[round(x, 5) for x in ds]}")


if __name__ == "__main__":
    main()
