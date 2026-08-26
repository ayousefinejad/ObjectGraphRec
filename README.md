# ObjectGraphRec

**Enhancing Multimodal Recommender Systems with Object Graphs**

Arshia Yousefi Nezhad, Abolfazl Nadi — University of Tehran

Multimodal recommenders describe an item by what it *looks like* and what it *says about itself*.
Neither says much about what an item is normally used **with**. A sofa, a rug, and a coffee table
turn up in the same living rooms and complement one another, yet their photos and descriptions
are entirely unalike — so a similarity-based recommender that sees a sofa can find you another
sofa far more easily than it can find the rug that belongs beside it.

ObjectGraphRec adds a third modality carrying exactly that signal: a global weighted graph of
**object co-occurrence mined from real indoor scenes**. Its edges come neither from user behaviour
nor from a hand-built relation ontology, but from which objects genuinely appear together in
photographs of rooms.

The headline finding is deliberately two-sided. Adding this modality improves LATTICE by
**+7.1% Recall@20** and **+4.9% NDCG@20**. On CRANE and MICRO it does not reliably help. Object
context is complementary to text and vision, **but only when the host architecture's structure
learning and fusion can exploit it** — the results below report that plainly rather than
averaging it away.

---

## How it works

```
  MIT Indoor (15,620 imgs)          ┌──────────────────────────────────────────┐
  NYU Depth v2  (1,449 imgs)  ───►  │ 1. OBJECT-GRAPH CONSTRUCTION             │
                                    │    OpenAI vision-language model labels   │
                                    │    each scene (open-vocabulary, so       │
                                    │    long-tail indoor objects survive)     │
                                    │    c_ab = #scenes containing both        │
                                    │    w_ab = c_ab / sqrt(c_a · c_b)   (Eq.2)│
                                    │    → 1,068 nodes, 17,669 edges           │
                                    └────────────────────┬─────────────────────┘
                                                         ▼
                                    ┌──────────────────────────────────────────┐
  all-MiniLM-L6-v2 (384-d)   ───►   │ 2. NODE REPRESENTATIONS                  │
  label embeddings init             │    2-layer GraphSAGE, mean aggregation   │
                                    │    contrastive, τ=0.5 → 64-d per node    │
                                    └────────────────────┬─────────────────────┘
                                                         ▼
  Amazon item image                 ┌──────────────────────────────────────────┐
  ─► detected objects O_i    ───►   │ 3. PER-ITEM POOLING                      │
                                    │    X_i^g = POOL({z_o : o ∈ O_i})   (Eq.3)│
                                    └────────────────────┬─────────────────────┘
                                                         ▼
  text X^t ─┐                       ┌──────────────────────────────────────────┐
  image X^v ─┼──────────────────►   │ 4. LATENT STRUCTURE + FUSION             │
  object X^g ┘                      │    kNN graph per modality       (Eq.4-6) │
                                    │    learned refinement, skip λ   (Eq.7-8) │
                                    │    A = Σ α_m A^m, α = softmax(β) (Eq.9)  │
                                    └────────────────────┬─────────────────────┘
                                                         ▼
                                    ┌──────────────────────────────────────────┐
                                    │ 5. GRAPH CONV + CF          (Eq.10-13)   │
                                    │    H = Ã H, then                         │
                                    │    x_i = x̂_i + ρ · h_i/||h_i||           │
                                    │    ŷ_ui = x̂_uᵀ x_i,  BPR loss            │
                                    └──────────────────────────────────────────┘
```

The object graph is built **once, offline, independently of any recommender**. That separation is
what lets the same modality be dropped into different hosts, which is what the experiments test.

---

## Results

Amazon Home & Kitchen (`home_v2`): 65,139 users, 14,503 items, 273,972 interactions, density
2.90 × 10⁻⁴. Each architecture is compared against itself with every other setting held fixed —
only the object modality is added.

### Effect of the object modality

| Architecture | Arm | R@10 | R@20 | P@10 | P@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **LATTICE** | image+text | 0.02928 | 0.04007 | 0.00300 | 0.00206 | 0.02005 |
| **LATTICE** | **+ object** | **0.03059** | **0.04291** | **0.00314** | **0.00221** | **0.02103** |
| MICRO | image+text | 0.03629 | 0.05053 | 0.00371 | 0.00259 | 0.02504 |
| MICRO | + object † | 0.03503 | 0.05091 | 0.00369 | 0.00255 | 0.02444 |
| CRANE | image+text | 0.02620 | 0.03970 | 0.00270 | 0.00203 | 0.01757 |
| CRANE | + object | 0.02657 | 0.04070 | 0.00277 | 0.00210 | 0.01807 |

† For MICRO the object channel is used in **fusion only** and excluded from the per-modality
InfoNCE term. Routed through the contrastive term instead, it reproduces ~93% of a substantial
loss — the mechanism behind MICRO's non-result.

**LATTICE: +7.1% R@20, +4.9% NDCG@20.** Paired across three seeds this is
+0.00284 ± 0.00037 (t = 13.4), roughly 9× the measured run-to-run noise floor.

### Reading these numbers honestly

Downstream results are **means over 3 seeds**, and two identical runs of this pipeline differ by
about **0.0003 R@20**. Any difference below that floor is not attributable to the intervention.
With 3 seeds a paired t needs |t| ≥ 4.303 for p < .05.

<sub>**A note on the CRANE row.** The table above reproduces the paper, whose CRANE comparison
came from a single-seed run. A seed-replicated repeat (both arms, 3 seeds, one fixed config)
gives **0.03990 ± 0.00111 → 0.04067 ± 0.00197, paired t = +0.44, not significant**, with per-seed
deltas +0.0040 / +0.0002 / −0.0019 changing sign. The means agree closely with the paper; what
the replicate adds is a noise estimate, against which the CRANE gain is not distinguishable from
zero. We report both rather than only the favourable one.</sub>

### Ablations

**Object recognizer** — open-vocabulary labelling matters more than detector accuracy:

| Detector | Labels | Unique vectors | Intrinsic AUC | R@20 |
|---|---|---|---|---|
| no object | – | – | – | 0.04007 ± 0.00031 |
| YOLOv8x | 62 | 61 | 0.7202 | 0.04216 ± 0.00041 |
| DETR-R101 | 72 | 69 | 0.6915 | 0.04212 ± 0.00047 |
| **OpenAI Vision** | **1,007** | **510** | 0.7405 | **0.04323 ± 0.00044** |

DETR has a *lower* intrinsic AUC than YOLOv8x yet matches it downstream — the first of several
places where link-prediction quality fails to predict recommendation quality.

**Scene corpus** — more coverage is not automatically better:

| Corpus | Nodes | Unique vectors | R@20 |
|---|---|---|---|
| NYU only | 360 | 292 | 0.04271 ± 0.00062 |
| **MIT only** | 1,007 | 510 | **0.04323 ± 0.00044** |
| MIT + NYU | 1,068 | 531 | 0.04291 ± 0.00051 |

**Encoder backbone** — GAT wins intrinsically and loses downstream:

| Encoder | Intrinsic AUC | R@20 | NDCG@20 |
|---|---|---|---|
| **GraphSAGE** | 0.789 | **0.04283 ± 0.00043** | **0.02098** |
| GAT | **0.825** | 0.04272 ± 0.00025 | 0.02090 |

**Post-fusion propagation** — learned attention does not beat the parameter-free operator:

| Operator | R@20 | P@20 | NDCG@20 |
|---|---|---|---|
| **Simple message passing** | **0.04323 ± 0.00044** | **0.00222** | **0.02111 ± 0.00029** |
| GAT propagation | 0.04254 ± 0.00066 | 0.00219 | 0.02075 ± 0.00024 |

Longer write-ups, including the coverage/placebo analysis behind the mechanism claim, are in
`analysis/RESULTS.md`. The scripts that generated every table and figure are in `analysis/`; note
they expect the original working-tree layout and the (not-included) result directories — see
`analysis/README.md`.

---

## Repository layout

```
src/                 LATTICE-based recommender + the ObjectGraph encoder package
  Models.py          modality fusion, kNN item graphs, graph convolution
  main.py            training loop, BPR, evaluation
  ObjectGraph/       co-occurrence graph construction, GraphSAGE, feature export
  utility/           data loading, metrics, argument parsing
scripts/             feature export, seeded experiment runners, reporting
data/prepare/        scene-corpus builders (MIT Indoor, NYU Depth v2, Visual Genome)
analysis/            figure/table generation + written results
figures/             F01–F19 (png/pdf/svg)
patches/             our changes to CRANE and MICRO — see patches/README.md
```

**Data is not included.** The working tree behind this paper is ~28 GB (803 MB item-graph cache
per variant, a 2.5 GB scene tarball, 475 MB of extracted image features), far beyond what belongs
in git. `data/prepare/` holds the code that rebuilds it.

---

## Getting started

```bash
git clone https://github.com/<user>/ObjectGraphRec && cd ObjectGraphRec
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENAI_KEY to re-run scene labelling
```

**1. Build the scene corpus** (or skip — a prepared `scenes.json` is enough to go on):

```bash
python data/prepare/download_mit_indoor.py
python data/prepare/detect_mit_openai.py      # needs OPENAI_KEY
python data/prepare/ingest_nyu_v2.py
python data/prepare/build_scenes.py           # → data/scenes.json
```

**2. Train the object-graph encoder and export item features:**

```bash
python scripts/export_lattice_feats.py --variant default_fixed --config default_fixed
# → data/lattice-runs/default_fixed/object_feat.npy   (14,503 × 64)
```

**3. Run the recommender**, image+text versus +object, three seeds each:

```bash
python scripts/run_lattice_study.py --variants default_fixed --seeds 0 1 2
python scripts/run_lattice_study.py --variants default_fixed --arms noobj --seeds 0 1 2
```

Results land in `data/lattice-runs/downstream.csv` and `fusion_arms.csv`. The reporting scripts in
`analysis/` turn those into the tables above — they assume the original directory layout, so read
`analysis/README.md` first.

Amazon Home & Kitchen image/text features follow the standard preprocessing of the
[LATTICE](https://github.com/CRIPAC-DIG/LATTICE) and
[MMRec](https://github.com/enoche/MMRec) benchmarks (4,096-d visual, 384-d textual).

Experiments ran on a single NVIDIA RTX 4090 (24 GB). A LATTICE arm takes ~25 min; the object-graph
encoder trains in about a minute.

---

## Citation

```bibtex
@article{yousefinezhad2026objectgraphrec,
  title   = {ObjectGraphRec: Enhancing Multimodal Recommender Systems with Object Graphs},
  author  = {Yousefi Nezhad, Arshia and Nadi, Abolfazl},
  year    = {2026}
}
```

## License and attribution

This repository is MIT licensed (see `LICENSE`). It builds on, and gratefully acknowledges:

- **LATTICE** — Zhang et al., *Mining Latent Structures for Multimedia Recommendation*, MM'21
- **MICRO** — Zhang et al., *Latent Structure Mining with Contrastive Modality Fusion*, TKDE'23 (MIT)
- **CRANE** — Dai et al., *Cross-modal Attention Network with Dual Graph Learning*, TOMM'25
- **MMGCN** — Wei et al., MM'19, ported via [enoche/MMRec](https://github.com/enoche/MMRec)
- **MIT Indoor Scene Recognition** — Quattoni & Torralba, CVPR'09
- **NYU Depth v2** — Silberman et al., ECCV'12
- **Visual Genome** — Krishna et al., IJCV'17
- **all-MiniLM-L6-v2** — Wang et al., NeurIPS'20

CRANE and MICRO source is **not** redistributed here; `patches/` carries only our modifications,
with upstream commits pinned.
