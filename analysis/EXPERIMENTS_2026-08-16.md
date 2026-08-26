# Object-graph experiments — 16 August 2026

LATTICE + object-graph modality, Amazon Home & Kitchen (`home_v2-2`, 5-core: 14,503 items,
59,251 users, 273,972 interactions). Object graph from MIT-Indoors + NYU-Depth v2 scenes.
Environment `~/hamedenv`, single RTX 4090 shared with another tenant.

---

## Headline

1. **The object modality's +7.1% on LATTICE is now explained, not just measured.** 51.8% of the
   catalogue carries a label that is verbatim a node of the object graph; the gain lands on
   exactly those items and vanishes on the ones the vocabulary misses.
2. **A placebo rules out the capacity/regularisation explanation.** Permuting *which item gets
   which object vector* — holding features, graph topology and parameter count fixed — drops
   performance *below* the image+text baseline.
3. **The +7.1% looks like a property of the MF backbone, not of LATTICE.** On a tuned LightGCN the
   same comparison gives +0.00032 (t = 0.99, n.s.).
4. **Corpus choice (MIT vs NYU) has not mattered anywhere yet** — every corpus sits inside the
   0.0012 resolution band on MF.

---

## 1. Vocabulary coverage — MIT/NYU objects vs the Amazon catalogue

**Question.** Indoor-scene objects and a Home & Kitchen catalogue are different domains. Is there
any real overlap for the modality to exploit?

Measured on the pipeline's own files: object side is the trained graph's node list
(verified equal to the `default_fixed` encoder checkpoint), item side is `5-core/raw_graph.txt` —
the exact file `build_object_feat` reads. Tiers follow that function's own branch: exact name
match, else nearest MiniLM node.

| Scene corpus | Graph nodes | Exact | Near (cos ≥ 0.6) | Unreached |
|---|---|---|---|---|
| MIT Indoor | 1,007 | 7,443 (51.3%) | 5,534 (38.2%) | 1,526 (10.5%) |
| NYU-Depth | 360 | 6,281 (43.3%) | 5,510 (38.0%) | 2,712 (18.7%) |
| **MIT + NYU (shipped)** | **1,068** | **7,506 (51.8%)** | **5,485 (37.8%)** | **1,512 (10.4%)** |

τ moves only the near/unreached split, never the exact share: 3.9% / 10.4% / 21.1% unreached at
τ = 0.5 / 0.6 / 0.7.

- The overlap sits on the catalogue's **head**: 20 of the 22 largest labels are exact nodes
  (`pillow` 340 items, `pan` 327, `blender` 260). The two exceptions are spelling, not meaning —
  `cookware set → Cookware` (0.88), `coffeemaker → Coffee maker` (0.94).
- **NYU adds only +63 items over MIT alone** (0.4pp), which independently explains the earlier
  MIT-vs-NYU corpus result: MIT carries the overlap.
- The honest miss list: `caddy → Taxidermy` (0.46), `chopper → Toy helicopter` (0.60).

→ `overlap_mit_nyu_amazon.npz`, `overlap_labels.csv`, **F17**

---

## 2. Where the gain lands — stratified evaluation

9 LATTICE runs (3 arms × 3 seeds) with best-epoch embedding dumps, ranking recomputed offline
reproducing `utility/batch_test.py` exactly.

**Validity gate:** offline global recall reproduces all nine runs' logged `test==` values,
max deviation **4.8e-6**. Three of the nine reproduced previously recorded numbers bit-exactly.

Metric is per-test-item hit@20 — R@20 is a per-user ratio and does not decompose over items.

| Stratum | (user, item) pairs | baseline | Δ (+object − image+text) | % | t(2) | min detectable Δ |
|---|---|---|---|---|---|---|
| all | 60,178 | 0.04064 | +0.00284 | +7.0% | +13.6 | 0.00088 |
| **exact** | 29,919 | 0.03601 | **+0.00265** | **+7.4%** | +7.6 | 0.00151 |
| **near** | 23,299 | 0.04419 | **+0.00379** | **+8.6%** | +8.2 | 0.00198 |
| **far** | 6,960 | 0.04866 | **+0.00043** | **+0.9%** | +0.8 n.s. | 0.00234 |

**Popularity confound, handled.** Covered labels are the catalogue's head and popular items are
easier to retrieve. Repeating the comparison *inside* each training-interaction decile, weighted
by the smaller tier's exposures: exact − far = **+0.00241, t(2) = +4.5**. The effect survives
matching.

Two things worth stating plainly rather than smoothing over:

- The **near** tier gains *more* than exact (+8.6% vs +7.4%) — the nearest-node fallback is not a
  degraded path.
- The **far** tier is underpowered: its minimum detectable Δ (0.00234) is close to the exact
  tier's effect (0.00265), so "no effect on unreached items" is directional at n = 3, not a tight
  bound.

→ `strata_hits.csv`, `strata_summary.csv`, **F18**

---

## 3. Placebo — is it the objects, or just a third feature matrix?

`default_fixed_shufobj`: the same `object_feat.npy` with its **rows permuted** (fixed seed).
Identical feature distribution, identical count of distinct vectors, and an item graph isomorphic
to the real one — `LATTICE_DIAG` confirms the same 531 tie groups and the same 0.7792
zero-in-degree fraction. The only thing destroyed is which item each object embedding belongs to.

| Arm | R@20 (3 seeds) | vs image+text | t(2) |
|---|---|---|---|
| image + text | 0.04007 ± 0.00032 | — | — |
| **+ shuffled object** | **0.03781 ± 0.00062** | **−0.00226 (−5.6%)** | **−11.1** |
| + object | 0.04291 ± 0.00051 | +0.00284 (+7.1%) | +13.3 |

A capacity or regularisation explanation predicts the shuffled arm matching the real one. It does
the opposite — it lands **below the baseline**, and the real arm beats it by **+13.5%
(t = +35.5)**. Mis-assigned object features are actively harmful, which is only possible if the
assignment carries real signal.

---

## 4. CF backbones — MF vs LightGCN vs NGCF

Following the LATTICE paper (arXiv:2104.09036, Table 3), which plugs LATTICE into MF, NGCF and
LightGCN. **Every previous result in this study used `--cf_model mf`, the paper's weakest
backbone.**

Tuned LightGCN (depth 3, lr 5e-4), 3 seeds, from `tuning.csv`:

| Backbone | image+text | + object | paired Δ | t(2) |
|---|---|---|---|---|
| **MF** | 0.04007 ± 0.00032 | 0.04291 ± 0.00051 | **+0.00284 (+7.1%)** | **+13.3** |
| **LightGCN (depth 3)** | 0.04340 ± 0.00069 | 0.04372 ± 0.00069 | **+0.00032 (+0.7%)** | **+0.99 n.s.** |
| NGCF | *in flight* | *in flight* | — | — |

**LightGCN's seed variance is the trap here.** At depth 2 the two seeds are 0.04455 and 0.04182 —
sd 0.00193, nearly 4× MF's 0.00051, and larger than the entire object effect on MF. A single seed
on this backbone means nothing; even 3 seeds only resolve ~0.0017.

If this holds, the paper's claim needs scoping: the +7.1% is an MF-backbone property, plausibly
because LightGCN's user-item propagation already supplies the item-item smoothing the object graph
was contributing. Plausible mechanism, not yet evidence.

---

## 5. Size-matched corpus — built, queued

MIT is 2,645 scenes / 1,007 nodes; NYU-Depth is 579 / 360 — 4.5× smaller. Any MIT-vs-NYU gap is
confounded with corpus size, so `mit_sub579` subsamples MIT to NYU's exact scene count (matched
*after* the encoder's own normalisation, since that is what it trains on):

| | scenes | nodes | objects/scene |
|---|---|---|---|
| MIT (full) | 2,645 | 1,007 | — |
| **MIT-sub579** | **579** | **490** | **7.47** |
| NYU-Depth | 579 | 360 | 5.54 |

Scene-matched but **not** object-matched — MIT scenes are simply richer. Reported rather than
engineered away.

Queued: tuned LightGCN × {MIT, NYU, MIT-sub579} × 3 seeds = 9 runs (~5 h).

---

## 6. Protocol facts, pulled from the artifacts

Extracted by `deck_facts.py` so they refresh rather than being re-typed.

**Split — leave-one-out, not a ratio.** 98.0% of users have exactly 1 validation and 1 test item;
the 55.8/22.2/22.0% interaction share is a consequence of short histories (median 4), not a rule.
Identical across all arms (variant dirs symlink `train/val/test.json`).

⚠️ **19.5% of evaluated users (11,548) have zero training interactions** — 2-item histories, one to
val, one to test. Scored but never trained on. Same block for every arm so comparisons are
unaffected, but it explains why absolute R@20 is ~0.043 rather than the paper's ~0.09 on its
datasets, and it makes cross-paper level comparisons invalid.

**Negative sampling.** 1 negative per positive, uniform over all 14,503 items rejecting the user's
train set, resampled every batch. BPR + L2 (1e-5), batch 1024.

**Early stopping.** Max 200 epochs; evaluate every 5; select on **validation** R@20; patience 10
evaluations (= 50 epochs). Reported test metrics come from the epoch that set the best validation
R@20 — never last-epoch, never best-test. Observed best epochs 100–120, median 115.

**Detector ablation.**

| Detector | Nodes | Intrinsic AUC | R@20 |
|---|---|---|---|
| YOLOv8x | 62 | 0.7202 | 0.04216 ± 0.00041 |
| DETR-ResNet101 | 72 | 0.6915 | 0.04212 ± 0.00047 |
| OpenAI (open-vocab) | 1,007 | 0.7405 | 0.04323 ± 0.00044 |

Paired DETR − YOLO = −0.00004: indistinguishable, not a DETR win. DETR has the **worst** intrinsic
AUC and still ties — intrinsic encoder quality does not predict recommendation quality, confirmed
independently across the detector sweep and the GAT-vs-GraphSAGE backbone swap.

**Seed variance on the headline.**

| Metric | image+text | + object | paired Δ | % | t(2) |
|---|---|---|---|---|---|
| R@10 | 0.02928 ± 0.00055 | 0.03059 ± 0.00020 | +0.00130 ± 0.00050 | +4.5% | 4.5 |
| **R@20** | 0.04007 ± 0.00031 | 0.04291 ± 0.00051 | **+0.00284 ± 0.00037** | **+7.1%** | **13.4** |
| P@20 | 0.00206 ± 0.00002 | 0.00221 ± 0.00003 | +0.00015 ± 0.00002 | +7.1% | 12.2 |
| NDCG@20 | 0.02005 ± 0.00025 | 0.02103 ± 0.00019 | +0.00098 ± 0.00018 | +4.9% | 9.4 |

Quote as **+7.1% ± 0.9%**. The paired sd (0.00037) is smaller than either arm's own sd — seed
noise is common to both arms and cancels. Reproducibility floor is ~0.0003; this effect is 9× it.

**Learned fusion weights α** (softmax of `modal_weight`, 26 object-bearing runs):
image 0.3449 ± 0.0044, text 0.3326 ± 0.0039, object 0.3225 ± 0.0057.

⚠️ Do **not** present α as modality importance. The weights barely leave uniform and object is the
*smallest* of the three; they receive a gradient on one batch per epoch (~130 steps/run). α
measures how little those parameters moved. The contribution is the ablation, not α.

---

## 7. Bugs found and fixed today

**CUDA OOM at the epoch boundary** (`Models.py:653`). The rebuild allocated this epoch's four
14,503² adjacencies while last epoch's four were still referenced — 3.4 GB of dead weight carried
into the peak, which OOMed at a 804 MiB request when the co-tenant grew to 9.8 GB. Fixed by
dropping the references before reallocating. Verified numerics-neutral: epoch 0 reproduces
bit-identically (`103.08196 = 103.08028 + 0.00168`, recall `[0.00689, 0.00950]`), and the
subsequent control runs reproduced `downstream.csv` exactly.

**α reported for a switched-off modality.** `fusion_state()` prints `softmax(modal_weight)`
*unmasked*, while `_modal_weights()` masks the logits to −inf before the softmax. On image+text
runs the object weight therefore printed as its untouched init (0.3333) when its effective value
was exactly 0. Three such runs had been averaged into the α table; now excluded.

**Two claims of my own I had to walk back**, both from single-seed reads: "LightGCN beats every MF
arm" (seed 0 is LightGCN's lucky seed) and quoting the paper's Sports R@20 next to our Home &
Kitchen numbers (different datasets — the paper never runs Home & Kitchen, so those figures set
the expected *ordering* of backbones, nothing more).

---

## Status and queue

### Done

| Experiment | Runs | Outcome |
|---|---|---|
| Vocabulary coverage MIT/NYU × Amazon | 0 (CPU) | 51.8% exact coverage; F17 |
| Stratified gain by coverage tier | 6 | gain concentrated on reached items |
| Shuffled-object placebo | 3 | −5.6% vs baseline; rules out capacity |
| Deck protocol facts + worked examples | 0 (CPU) | split, negatives, stopping, α, 8c/8d |
| Size-matched MIT corpus `mit_sub579` | 0 (CPU) | 579 scenes / 490 nodes |

### Running

**NGCF backbone**, depth 2, ±object, 3 seeds (6 runs, ~3 h remaining). Seed 0 +object landed at
R@20 0.03053 — well below both MF and LightGCN, matching the paper's ordering only partially
(the paper has NGCF above MF). Depth 2 may not suit NGCF here; `tuning.csv` already shows NGCF
depth 1 (0.03906) beating depth 2 (0.03698) on image+text, so **depth 1 is likely the fairer NGCF
setting** and this sweep may need repeating there before NGCF is claimable.

### Queued

| # | Experiment | Runs | Est. | Answers |
|---|---|---|---|---|
| 1 | Tuned LightGCN × **MIT-only** × 3 seeds | 3 | ~1.7 h | the direct request: MIT + strong backbone |
| 2 | Tuned LightGCN × **NYU-only** × 3 seeds | 3 | ~1.7 h | MIT vs NYU on a strong backbone |
| 3 | Tuned LightGCN × **MIT-sub579** × 3 seeds | 3 | ~1.7 h | is it the imagery or the corpus size? |
| 4 | MF × MIT-sub579 × 3 seeds *(droppable tail)* | 3 | ~1.2 h | size control for the existing MF corpus table |

All at `--cf_model lightgcn --weight_size [64,64,64] --lr 0.0005 --epoch 400`, matching
`tuning.csv`'s `obj_lgn3` row exactly so the new corpus rows are comparable to the existing
MIT+NYU one. Baselines for this table already exist (`lgn3` image+text and `obj_lgn3` MIT+NYU,
3 seeds each) and are **not** being re-run.

**Dropped from the queue:** the 4 remaining depth-2 LightGCN runs from this morning's sweep,
superseded by the depth-3 tuned rows.

### Still open, not queued

- **NGCF at depth 1** — likely the fairer setting; decide after the current NGCF runs land.
- **Corpus characterisation figure (F19)** — corpus size, vocabulary intersection, catalogue
  coverage, intrinsic AUC and downstream R@20 on both backbones, side by side.
- **`RESULTS.md` §7** — "which scene corpus, and does it matter on a strong backbone".
- **B7 product photo** — no Amazon product images on disk (only extracted features and ASINs);
  needs either a fetch by ASIN or substituting a real MIT-Indoors scene.

### Resolution floors, for reading any of the above

| Backbone | between-seed sd | resolvable at 3 seeds |
|---|---|---|
| MF | 0.00051 | ~0.0012 |
| LightGCN depth 3 | 0.00069 | ~0.0017 |
| LightGCN depth 2 | 0.00193 | ~0.0048 |

Reproducibility floor from two identical runs: **~0.0003**. Any delta below it is unattributable
regardless of seed count.

Frozen artifacts (`object_feat.npy`, the three `*_adj_10.pt`) verified unchanged after every
batch against `frozen_artifacts.md5`.

---

## Files

| Path | What |
|---|---|
| `make_overlap_data.py` → `overlap_mit_nyu_amazon.npz`, `overlap_labels.csv` | coverage measurement |
| `plot_overlap.py` → `figures/F17_vocabulary_overlap.*` | coverage figure |
| `make_strata_data.py` → `strata_hits.csv`, `strata_summary.csv`, `strata_global.json` | stratified evaluation |
| `plot_strata.py` → `figures/F18_coverage_strata.*` | strata + placebo figure |
| `deck_facts.py`, `deck_examples.py` | protocol numbers and worked examples |
| `object-graph/scripts/export_shuffled_feats.py` | placebo dataset |
| `object-graph/scripts/make_subsampled_corpus.py` | size-matched corpus |
| `object-graph/scripts/queue_coverage_proof.sh`, `queue_backbones.sh` | GPU batches |
| `frozen_artifacts.md5` | frozen-artifact baseline (now in-repo, was `/tmp`) |
