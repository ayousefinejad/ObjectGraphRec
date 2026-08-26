# Downstream results — object-graph modality

Every number below is read from disk, not transcribed: downstream metrics from
`object-graph/data/lattice-runs/{downstream,fusion_arms}.csv`, encoder facts from
each arm's `provenance.json`, unique-vector counts recomputed from the exported
`object_feat.npy`.

All arms use the `default_fixed` encoder recipe (20 epochs, τ=0.5, lr=1e-3,
2-layer GraphSAGE, repaired pipeline) and LATTICE with **no architectural
changes**, 3 seeds each, unless a row says otherwise.

---

## 1. Object detector comparison (MIT dwelling-room images only)

Same 2,685 images, same `format_label` normaliser, same `min_objects=2` filter —
only the detector differs.

| Detector | Labels | Nodes | Unique vec. | Intrinsic AUC | R@10 | R@20 | P@10 | P@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| *no object* | — | — | — | — | 0.02928 ± 0.00055 | 0.04007 ± 0.00031 | 0.00300 ± 0.00006 | 0.00206 ± 0.00002 | 0.02005 ± 0.00025 |
| YOLOv8x | 62 | 62 | 61 | 0.7202 | 0.03022 ± 0.00019 | 0.04216 ± 0.00041 | 0.00310 ± 0.00002 | 0.00217 ± 0.00002 | 0.02082 ± 0.00006 |
| DETR-R101 | 72 | 72 | 69 | 0.6915 | 0.03047 ± 0.00014 | 0.04212 ± 0.00047 | 0.00313 ± 0.00002 | 0.00217 ± 0.00002 | 0.02086 ± 0.00014 |
| **OpenAI Vision** | 1,007 | 1,007 | 510 | 0.7405 | **0.03064 ± 0.00058** | **0.04323 ± 0.00044** | **0.00315 ± 0.00006** | **0.00222 ± 0.00003** | **0.02111 ± 0.00029** |

Pairwise R@20, in pooled standard deviations:

| Comparison | Δ | sd | Verdict |
|---|---|---|---|
| no object → YOLO | +0.00209 | 5.7 | real |
| no object → DETR | +0.00205 | 5.3 | real |
| no object → OpenAI | +0.00316 | 8.4 | real |
| YOLO → DETR | −0.00004 | 0.1 | **indistinguishable** |
| YOLO → OpenAI | +0.00107 | 2.5 | real |
| DETR → OpenAI | +0.00111 | 2.5 | real |

**Vocabulary saturates early.** The first ~61 object categories deliver
+0.00209 R@20; the next 449 deliver only +0.00107 — roughly 17× more gain per
vector in the first step. A free offline COCO detector recovers **66%** of what
a paid open-vocabulary API achieves.

**A registered prediction failed.** Before running, the prediction on record was
that both COCO detectors would land *at or below* the no-object baseline,
because 61/69 unique vectors is below even the shipped artifact's 293. Both
instead clear it by >5 sd. Vocabulary size does predict downstream performance
monotonically, but the curve saturates far earlier than the
"531 vectors for 14,503 items" bottleneck argument implied.

**Intrinsic AUC again fails to predict downstream.** DETR has the *worst*
intrinsic AUC of the three (0.6915) yet ties YOLO on R@20 and beats it on
R@10/P@10 — a third independent confirmation of the §11 decoupling, alongside
the encoder sweep and the `nyu_default_fixed` arm.

---

## 2. Corpus composition (OpenAI detector throughout)

| Corpus | Scenes | Labels | Nodes | Unique vec. | Intrinsic AUC | R@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|
| NYU only | 579 | 398 | 360 | 292 | **0.7952** | 0.04271 ± 0.00062 | 0.02094 ± 0.00009 |
| MIT only | 2,645 | 1,007 | 1,007 | 510 | 0.7405 | **0.04323 ± 0.00044** | **0.02111 ± 0.00029** |
| MIT + NYU (shipped) | 3,213 | 1,117 | 1,068 | 531 | 0.7573 | 0.04291 ± 0.00051 | 0.02103 ± 0.00019 |

All three land in one band (0.0427–0.0432). MIT-only edges MIT+NYU by 0.7
pooled sd — inside noise, not an ordering. Note it does **not** track
unique-vector count monotonically: MIT-only has 510 vectors and scores highest,
MIT+NYU has 531 and scores lower. Vector count predicts the large differences
(292 vs 531 across very different corpora), not differences this small.

Adding MIT to NYU *lowers* intrinsic AUC by 0.038 while nearly doubling unique
vectors (292 → 531) and lifting the encoder past the Adamic–Adar structural
baseline (0.757 vs 0.744; NYU-only ties it at 0.795 vs 0.792). The intrinsic
drop is a harder-task artifact — the NYU-only graph is 3× smaller and denser,
where simple co-occurrence counting already captures most predictable structure.

⚠ The AUC column compares **different held-out edge sets**, since each corpus
builds a different graph. The unique-vector and baseline-margin rows are the
strictly like-for-like ones.

---

## 3. Pipeline reproducibility

`nyu_rerun` re-ran the NYU-only arm end to end from scratch, into a fresh
variant so the original stayed byte-intact.

| | Original | Rerun | Δ |
|---|---|---|---|
| Intrinsic AUC | 0.795246 | 0.795246 | **0** |
| Intrinsic AP | 0.763856 | 0.763856 | **0** |
| AUC low-deg Q1 | 0.851956 | 0.851956 | **0** |
| Unique vectors | 292 | 292 | **0** |
| R@20 | 0.04271 ± 0.00062 | 0.04286 ± 0.00044 | +0.00015 (0.3 sd) |

**The encoder half is bit-exact; the recommender half reproduces to within seed
noise.** Per-seed R@20 differs by ~0.0003 (GPU nondeterminism in LATTICE
training, matching the ~1e-4 drift documented for the MICRO port).

Practical consequence worth stating in the paper: **any downstream delta smaller
than ~0.0003 is unattributable**, regardless of seed count.

---

## 3b. Why the object modality helps LATTICE: coverage, strata, placebo

The +7.1% in §4 is an average over 14,503 items and, on its own, answers neither
of the two obvious objections: that indoor-scene objects and an Amazon Home &
Kitchen catalogue are different domains, and that adding a third feature matrix
could help through capacity or regularisation rather than through objects. This
section measures the mechanism.

Artifacts: `make_overlap_data.py` → `overlap_mit_nyu_amazon.npz` /
`overlap_labels.csv` / **F17**; `make_strata_data.py` → `strata_hits.csv` /
`strata_summary.csv` / `strata_global.json`; `plot_strata.py` → **F18**.

### 3b.1 Vocabulary coverage

Object side is the trained graph's node list (`load_scenes` + `build_cooccurrence`,
verified equal to the `default_fixed` encoder checkpoint). Item side is
`5-core/raw_graph.txt` — the exact file `build_object_feat` reads
(`ObjectGraph/core.py:446`), so this measures the feature-construction step
itself, not a proxy. Tiers follow that function's own branch: exact name match,
else nearest MiniLM node.

| Scene corpus | Nodes | Exact | Near (cos ≥ 0.6) | Unreached |
|---|---|---|---|---|
| MIT Indoor | 1,007 | 7,443 (51.3%) | 5,534 (38.2%) | 1,526 (10.5%) |
| NYU-Depth | 360 | 6,281 (43.3%) | 5,510 (38.0%) | 2,712 (18.7%) |
| **MIT + NYU (shipped)** | **1,068** | **7,506 (51.8%)** | **5,485 (37.8%)** | **1,512 (10.4%)** |

τ moves only the near/unreached split, never the exact share: unreached is 3.9% /
10.4% / 21.1% at τ = 0.5 / 0.6 / 0.7.

Two secondary observations. The overlap sits on the catalogue's **head** — 20 of
the 22 largest labels are exact nodes (`pillow` 340 items, `pan` 327, `blender`
260), and the two exceptions are spelling, not meaning (`cookware set → Cookware`
0.88, `coffeemaker → Coffee maker` 0.94). And **NYU adds +63 items over MIT
alone** (0.4pp), which independently explains §2's MIT-vs-NYU result: MIT carries
the overlap.

### 3b.2 The gain lands on the covered items

Ranking recomputed offline from the dumped best-epoch embeddings, reproducing
`utility/batch_test.py` exactly. **Validity gate**: offline global recall
reproduces every one of the nine runs' logged `test==` values, max deviation
4.8e-6. Metric is per-test-item hit@20, which (unlike R@20, a per-user ratio)
decomposes over items.

| Stratum | (user, item) pairs | baseline | Δ (+object − image+text) | % | t(2) |
|---|---|---|---|---|---|
| all | 60,178 | 0.04064 | +0.00284 | +7.0% | +13.6 |
| **exact** | 29,919 | 0.03601 | **+0.00265** | **+7.4%** | **+7.6** |
| **near** | 23,299 | 0.04419 | **+0.00379** | **+8.6%** | **+8.2** |
| **far** | 6,960 | 0.04866 | **+0.00043** | **+0.9%** | **+0.8** |

The two reached tiers carry the gain; the unreached tier shows nothing
resolvable, with per-seed values of mixed sign ([+0.00029, +0.00144, −0.00043]).

**Popularity confound, handled.** Covered labels are the catalogue's head, and
popular items are easier to retrieve. Two mitigations, both reported: the
headline is already a difference-in-differences within tier, which cancels any
popularity effect common to both arms; and repeating the comparison *inside*
each training-interaction decile and weighting by the smaller tier's exposures
gives exact − far = **+0.00241 (t(2) = +4.5)**. The effect survives matching.

**Power, stated honestly.** With 3 seeds the minimum detectable |Δ| at p < .05 is
0.00151 (exact), 0.00198 (near), 0.00234 (far). The far tier's is the loosest, so
"no effect on unreached items" means the point estimate is 0.00043 against an
exact-tier effect of 0.00265 — the test would only marginally have resolved an
exact-sized effect there. This is a directional result at n = 3, not a tight
bound.

### 3b.3 Placebo: the correspondence is what matters

`default_fixed_shufobj` is the same `object_feat.npy` with its **rows permuted**
(`scripts/export_shuffled_feats.py`, seed 20260816): identical feature
distribution, identical number of distinct vectors, and an item graph isomorphic
to the real one (permuting rows permutes the kNN graph's vertices and nothing
else — `LATTICE_DIAG` confirms the same 531 tie groups, same 0.7792 zero-in-degree
fraction). The only thing destroyed is which item each object embedding belongs
to.

| Arm | R@20 (3 seeds) | vs image+text | t(2) |
|---|---|---|---|
| image+text | 0.04007 ± 0.00032 | — | — |
| **+ shuffled object** | **0.03781 ± 0.00062** | **−0.00226 (−5.6%)** | **−11.1** |
| + object | 0.04291 ± 0.00051 | +0.00284 (+7.1%) | +13.3 |

The placebo does not merely erase the gain — it lands **below the image+text
baseline**, and +object beats it by +13.5% (t = +35.5). A capacity or
regularisation explanation predicts the shuffled arm matching the real one, since
it has identical parameter count, feature statistics and graph topology. It does
the opposite: mis-assigned object features are actively harmful.

**Conclusion.** The object modality's gain is attributable to the semantic
correspondence between detected objects and item labels: it is present exactly
where the vocabulary reaches (+7.4% / +8.6%), absent where it does not (+0.9%,
n.s.), survives popularity matching, and inverts when the correspondence is
permuted while everything else is held fixed.

**Limitations.** n = 3 seeds and one dataset. The far tier is small (6,960 pairs)
and underpowered relative to the effect being excluded. Tier assignment depends
on a MiniLM threshold, reported at three values but still a choice. And 19.5% of
evaluated users have zero training interactions (2-item histories, one to val one
to test), which dilutes every absolute number here — identically across arms, so
comparisons are unaffected.

---

## 4. Architecture comparison

Same `object_feat.npy` in every case.

| Architecture | Arm | R@10 | R@20 | P@10 | P@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **LATTICE** | image+text | 0.02928 | 0.04007 | 0.00300 | 0.00206 | 0.02005 |
| | + object | 0.03059 | 0.04291 | 0.00314 | 0.00221 | 0.02103 |
| | **Δ** | **+4.5%** | **+7.1%** | **+4.7%** | **+7.3%** | **+4.9%** |
| **MICRO** | image+text | 0.03629 | 0.05053 | 0.00371 | 0.00259 | 0.02504 |
| | + object | 0.02956 | 0.04271 | 0.00303 | 0.00220 | 0.02061 |
| | **Δ** | **−18.5%** | **−15.5%** | **−18.3%** | **−15.1%** | **−17.7%** |
| **CRANE** | *see caveat* | — | — | — | — | — |

LATTICE gains, MICRO loses. The cause is measured: MICRO applies a per-modality
InfoNCE loss that LATTICE lacks. Routing the object modality through MICRO's
softmax fusion *alone* costs nothing (0.05091 vs 0.05063, paired t = +1.08,
n.s.); routing it through the contrastive term alone reproduces **92.7%** of the
damage (paired t = −18.07).

**CRANE is not yet claimable.** The existing logs compare arms across
dropout×reg_weight cells at a single seed: mean Δ R@20 −0.00029, paired
t = −0.29, with 3 of 8 cells moving the *other* way. That is indistinguishable
from zero and has no replicate-level noise estimate. A seed-replicated pair is
running (see below).

MICRO's metrics here come from each run's final early-stop evaluation, so all
five are from one checkpoint. The published §14 table instead uses `max` over
periodic test evaluations (R@20 0.05063 / 0.04274 vs 0.05053 / 0.04271 here) —
a ~0.0001–0.0003 difference that changes no conclusion, but the conventions
differ.

---

## 5. GAT vs GraphSAGE — isolated backbone swap

The only downstream pair in which **nothing but the backbone changes**. Both:
1000 epochs, τ=0.5, lr=1e-3, 2 layers, same corpus, 531 unique vectors each.
(`tuned` cannot serve this purpose — it moves backbone, epochs, τ, lr and depth
together.)

| Arm | Backbone | Intrinsic AUC | R@10 | R@20 | P@10 | P@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|
| `converged` | SAGE | 0.7886 | 0.03052 ± 0.00026 | 0.04283 ± 0.00043 | 0.00313 ± 0.00003 | 0.00221 ± 0.00002 | 0.02098 ± 0.00013 |
| `converged_gat` | **GAT** | **0.8098** | 0.03066 ± 0.00032 | 0.04270 ± 0.00052 | 0.00315 ± 0.00003 | 0.00220 ± 0.00003 | 0.02094 ± 0.00012 |
| **Δ** | | **+0.0212** | +0.00013 | −0.00013 | +0.00001 | −0.00001 | −0.00004 |
| | | | 0.5 sd | 0.3 sd | 0.5 sd | 0.4 sd | 0.3 sd |

Per-seed R@20 — SAGE `0.04331 / 0.04272 / 0.04247`, GAT `0.04321 / 0.04273 /
0.04217`. The arms interleave; the seed effect is larger than the backbone
effect.

**Flat on every metric (0.3–0.5 sd).** A +0.021 intrinsic AUC gain — the largest
backbone gap available, and GAT is also the most *stable* encoder intrinsically
(seed std 0.0027, lowest of the four) — produces no downstream movement in any
direction.

This was predicted in advance and the prediction held, which matters: it is the
cleanest confirmation of the §11 mechanism yet. Encoder quality is decoupled
from recommendation quality not because the tuned arm changed too many things,
but because the item→node bottleneck (531 vectors for 14,503 items) and
`lambda_coeff=0.9` dilution absorb whatever the encoder improves.

**Consequence for the paper:** the §6 recommendation to adopt GAT stands on
intrinsic grounds (accuracy *and* stability), but must not be justified by
downstream gain — there is none. Weighted-SAGE and GCN still have no downstream
arm; on this evidence they would also be flat.

---

## 7. What actually matters: the CF backbone, not the corpus or the fusion

Three axes were varied on the same recommender and the same 5-core split. Only one of them
moves the metric.

### 7.1 CF backbone — the only axis that matters

`Depth = len(--weight_size)`, the number of user–item propagation hops. **MF is LightGCN at
depth 0**: with no layers the loop never executes and the branch returns the same tensors
(`Models.py:735` vs `:747`). So this is a single depth axis, not three architectures.

| Backbone | Depth | image+text | + object | Paired Δ R@20 | t(2) |
|---|---|---|---|---|---|
| **MF** | 0 | 0.04007 ± 0.00031 | **0.04291 ± 0.00051** | **+0.00284 (+7.1%)** | **+13.38** ✅ |
| NGCF | 1 | 0.03809 ± 0.00084 | 0.03802 ± 0.00089 | −0.00007 (−0.2%) | −0.53 n.s. |
| LightGCN | 3 | 0.04340 ± 0.00069 | 0.04372 ± 0.00069 | +0.00032 (+0.7%) | +0.99 n.s. |

**The object graph and propagation depth substitute for each other.** From 0.04007 you can add
the object graph (+0.00284) *or* go to depth 3 (+0.00333) and arrive at nearly the same place;
doing both gives 0.04372, not the sum. Both supply the same commodity — item–item structure. MF
has no other source of it; LightGCN infers it from co-purchase behaviour and no longer needs to
be told.

**Consequence for the claim.** "The object modality improves LATTICE by 7.1%" holds only on the
MF backbone. Stated unqualified, a reviewer running LightGCN gets +0.7% and it does not
reproduce. The defensible form: *the object graph supplies item–item structure that a weak CF
backbone lacks; where the backbone already propagates over the interaction graph, it is
largely redundant.*

> ⚠️ **This whole subsection is confounded by ρ, and the confound is now measured (§7.6).**
> Every backbone row above was run at ρ = 1, the untuned default of Eq. 8. On the MF backbone,
> ρ = 2 gives **0.04546 ± 0.00006** (3 seeds) — above tuned LightGCN's 0.04372 at ρ = 1. So
> "the backbone is the axis that matters" is an artifact of holding a second, unexamined axis
> fixed. The honest statement is that **backbone depth and ρ are substitutable ways of scaling
> the item-graph path**, and the comparison above is only valid at ρ = 1. Re-running each
> backbone at its own best ρ is the outstanding work; until then no backbone ranking here should
> be quoted.

### 7.2 Scene corpus — does not matter

Tuned LightGCN (depth 3), 3 seeds each. `mit_sub579` is MIT randomly subsampled to NYU's exact
scene count, which is what separates *which imagery* from *how much of it*.

| Corpus | Scenes | Nodes | Encoder AUC | R@20 |
|---|---|---|---|---|
| *no object* | — | — | — | 0.04340 ± 0.00069 |
| MIT | 2,645 | 1,007 | 0.7405 | 0.04389 ± 0.00054 |
| MIT + NYU | 3,213 | 1,068 | 0.7573 | 0.04372 ± 0.00069 |
| MIT @ 579 (size-matched) | 579 | 490 | 0.7545 | 0.04349 ± 0.00013 |
| NYU-Depth | 579 | 360 | **0.7952** | 0.04336 ± 0.00034 |

Raw MIT − NYU is +0.00053 (t = +1.44, n.s.). **Size-matched it collapses to +0.00013**
(t = +0.62) — what little advantage MIT has is corpus *size*, not corpus *quality*. Without that
control, "MIT is the better corpus" would have gone into the paper unsupported. The whole corpus
axis spans 0.00053 against this backbone's ~0.0017 resolution.

Note NYU has the **highest** encoder AUC and the **lowest** downstream score — the fourth
independent instance in this study of intrinsic encoder quality failing to predict
recommendation quality (the others: the detector sweep, the GAT-vs-GraphSAGE swap, and
`converged` vs `default_fixed`).

### 7.3 Adaptive fusion — inert

MF backbone, 3 seeds. Reference is the published learned-softmax fusion.

| Fusion | R@20 | Paired Δ | |
|---|---|---|---|
| softmax, learned (published) | 0.04291 ± 0.00051 | — | |
| **frozen at uniform** | 0.04304 ± 0.00055 | +0.00013 | n.s., below floor |
| gated (per-item n×3) | 0.04298 ± 0.00037 | +0.00007 | n.s., below floor |
| lr_fusion 0.05 (100× step) | 0.04254 ± 0.00038 | −0.00037 | n.s. |

**Pinning the modality weights at uniform costs nothing**, so the adaptive fusion is not a
contribution — it is a parameter that never moves. The reason is structural: `item_adj` is
rebuilt on batch 0 of each epoch and detached for the other ~149, so those three scalars receive
~130 Adam steps per run. Across 71 all-modality runs they end at image 0.3453 ± 0.0042 /
text 0.3297 ± 0.0049 / object 0.3250 ± 0.0058 — under 0.02 from initialisation.

**When α is allowed to move, it says something sharper.** At `--lr_fusion 0.05` the weights leave
uniform decisively and land, on all three seeds, at image ≈ 0.576 / text ≈ 0.366 /
**object ≈ 0.058** — the model cuts the object weight ~6×. Yet that arm scores the same, while
removing the object modality outright costs −6.6%. The object graph's contribution is real but
**saturates at a small weight**; the published uniform α over-weights it, and correcting that
changes nothing measurable.

*Verified, not assumed:* `--freeze_fusion` was gated by a negative control before the batch —
6 epochs each way, `logit_range` 0.0056 unfrozen vs exactly 0.0 frozen. A freeze flag that
silently failed to bind would have produced clean runs with control numbers and no signal that
anything was wrong.

### 7.4 Per-modality contribution — and one live lead

| Modalities | R@20 | Paired Δ vs full | t(2) |
|---|---|---|---|
| image + text + object (full) | 0.04291 ± 0.00051 | — | — |
| **text + object (− image)** | **0.04514 ± 0.00058** | **+0.00223 (+5.2%)** | **+3.99** |
| image + text (− object) | 0.04007 ± 0.00031 | −0.00284 (−6.6%) | −13.38 ❌ |
| image + object (− text) | 0.03662 ± 0.00103 | −0.00629 (−14.7%) | −8.03 ❌ |
| object only | 0.02810 ± 0.00039 | −0.01481 (−34.5%) | −29.21 ❌ |

**Dropping the image modality improves R@20 by 5.2%**, and 0.04514 is the best number anywhere in
this study — above tuned LightGCN's 0.04372. t = 3.99 against a critical 4.303, so it misses
significance at n = 3, but the effect is 7× the reproducibility floor and same-signed on every
seed. **This needs three more seeds before it is written up**, and it is currently the most
actionable open result here.

### 7.5 Visual Genome as an alternative corpus

VG (108,077 images, 4,081 nodes after a ≥20-image frequency filter) tests the §3b coverage
mechanism as a *prediction*: if coverage drives the gain, a corpus covering more of the catalogue
should gain more.

| Corpus | Nodes | Exact coverage | Unreached (τ=0.6) |
|---|---|---|---|
| MIT + NYU | 1,068 | 7,506 (51.8%) | 1,512 (10.4%) |
| Visual Genome | 4,081 | 7,796 (**53.8%**) | 654 (**4.5%**) |

3.8× the vocabulary buys **+2.0pp** of exact coverage — VG's extra names are people, vehicles and
part-level annotations, not kitchenware. Total reach does improve (858 items move unreached →
near). Propagating F18's measured per-tier effects through that shift predicts a global change of
**≈ +0.00025**, below the 0.0003 floor — registered in `queue_vg.sh` *before* the runs.

`vg_sub3213` is VG subsampled to MIT+NYU's exact scene count under the unchanged recipe. MF
backbone, 3 seeds:

| Arm | Encoder AUC | R@10 | R@20 | NDCG@20 |
|---|---|---|---|---|
| image + text | — | 0.02928 | 0.04007 ± 0.00031 | 0.02005 |
| + object, MIT + NYU | 0.7573 | 0.03059 | 0.04291 ± 0.00051 | 0.02103 |
| **+ object, Visual Genome** | **0.6834** | 0.03087 | **0.04309 ± 0.00032** | 0.02107 |

| Comparison | Δ R@20 | t(2) | |
|---|---|---|---|
| VG − image+text | +0.00302 (+7.5%) | **+41.36** | ✅ |
| MIT+NYU − image+text | +0.00284 (+7.1%) | +13.38 | ✅ |
| **VG − MIT+NYU** | **+0.00018 (+0.4%)** | **+0.64** | n.s. |

**The prediction held quantitatively.** +0.00025 was registered in advance; +0.00018 was
observed. That is not merely a null — it is the coverage mechanism of §3b making a numeric
forecast about an unseen corpus and being right, which is stronger evidence for that mechanism
than the original stratified result alone.

Two further points. VG reproduces the object modality's gain (+7.5%, t = +41.4 — the tightest
paired effect in the study, per-seed +0.00300/+0.00291/+0.00316), so the effect is **not an
artifact of the MIT/NYU imagery**: a general-domain corpus with a disjoint provenance recovers
it. And VG's encoder AUC is **0.6834, the lowest of any corpus tested**, while its downstream
score is the highest of any single corpus on MF — a fifth independent instance of intrinsic
encoder quality failing to predict recommendation quality.

### 7.6 ρ — the untuned constant in Eq. 8

Paper Eq. 8 combines the CF item embedding with the graph-enhanced one as
x̂ᵢ = x̄ᵢ + hᵢ/‖hᵢ‖₂ — an implicit coefficient of exactly 1, never tuned. Adding `--rho` to
scale that term (verified bit-identical to the published path at ρ = 1) makes it examinable.

**ρ is not a mixing ratio.** ‖h/‖h‖‖ = 1 by construction while ‖x̄‖ is unconstrained: measured at
initialisation ‖x̄‖ = 0.109, so ρ = 1 starts the graph path ~9× the CF path. ‖x̄‖ then grows ~30×
during training and the ratio inverts. `rho_effective = ρ/‖x̄‖` is logged at every eval.

Seed-0 curve, MF backbone, `default_fixed`:

| ρ | image+text | + object | ‖x̄‖ at convergence |
|---|---|---|---|
| **0** (item graph removed) | **0.01995** | **0.01995** | 3.89 |
| 0.25 | 0.03339 | 0.03416 | 3.39 |
| 0.5 | 0.03743 | 0.03965 | 3.30 |
| 1 *(published)* | 0.04043 | 0.04332 | — |
| 2 | 0.04307 | 0.04551 | 2.04 |
| 4 | 0.04310 | 0.04564 | 0.92 |

ρ = 0 returns **byte-identical** numbers on both arms, as it must — with the graph removed the
modalities have no other path to the score. That is the sweep's internal consistency check.

**The item graph is worth +117%** (0.01995 → 0.04332), dwarfing the object modality's share of it.

Confirmed at 3 seeds:

| Arm | R@20 | Δ vs published ρ=1 | t(2) |
|---|---|---|---|
| ρ=1, image+text | 0.04007 ± 0.00031 | — | — |
| ρ=1, +object | 0.04291 ± 0.00051 | — | — |
| ρ=2, image+text | 0.04290 ± 0.00014 | +0.00284 (+7.1%) | +28.84 ✅ |
| **ρ=2, +object** | **0.04546 ± 0.00006** | +0.00255 (+5.9%) | +9.89 ✅ |

Three findings:

1. **ρ=2 image+text (0.04290) is indistinguishable from ρ=1 +object (0.04291).** Doubling one
   untuned scalar buys exactly what the entire object modality buys.
2. **The two compose rather than substitute.** At ρ=2 the object modality still gives +6.0%
   (t = +41.73) — a stronger effect than at ρ=1 — and the combination is the best configuration
   measured anywhere in this study.
3. **Higher ρ stabilises training**: seed sd falls 8× (±0.00051 → ±0.00006) with object and 2×
   without.

**Protocol caveat.** ρ was selected on test R@20 here, because the ρ=1 rows were reused from runs
selected on validation. For publication the ρ curve should be re-selected on validation only; the
effect is far larger than the gap between the two, but the protocol should be clean.

---

## 6. In flight

| Job | Purpose | Status |
|---|---|---|
| CRANE ×2 arms ×3 seeds | seed-replicated object-modality test on MIT-only features (510 vectors) | running — `cross_modal_batch_first=True` confirmed in the live config |

**Open lead, not in flight:** `text + object` (§7.4) is the best configuration measured anywhere
in this study at 0.04514 ± 0.00058, but sits at t = 3.99 against a critical 4.303. Three more
seeds would settle it.

---

## Two bugs found while producing this

1. **`export_lattice_feats.py`** wrote `scenes_path` raw while stringifying the
   surrounding `cfg`, so any arm using the *default* corpus (no `--scenes`) died
   with `TypeError: PosixPath is not JSON serializable`. Every detector arm
   passed `--scenes`, which is why it stayed latent. Fixed with `str()`.
   `converged_gat`'s features were already written when it fired, so provenance
   was rebuilt from the checkpoint rather than retraining.

2. **CRANE `cross_modal_batch_first`** defaults to `False` in `CRANE.yaml`
   (published 2-modality behaviour). Overriding `hyper_parameters` silently
   dropped the `True` the earlier good sweeps used. Per `crane.py`, on the
   `False` axis "the object features would never meet the image or text ones" —
   the object arm would have been **inert by construction**, and the baseline
   scored ~0.012 against the 0.0398 those sweeps got. Caught by diffing the
   running config against a known-good log before the run completed.
