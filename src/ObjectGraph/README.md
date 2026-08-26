# ObjectGraph

Object co-occurrence graph encoder for the multimodal LATTICE recommender.

**Full architecture (research):** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Training pipeline

```
YOLO scenes → co-occurrence graph → MiniLM node features
        → (1) Contrastive GraphSAGE
        → (2) GraphMAE fine-tuning  [KDD 2022]
        → object_feat.npy → LATTICE
```

## Quick start

```python
from ObjectGraph import train, train_mae, train_full, build_object_feat

train_full()                      # contrastive + GraphMAE (recommended)
build_object_feat("home_v2-2")    # export (n_items, 64)
```

Stage-by-stage:

```python
train()                           # Stage 1: contrastive
train_mae(cfg={"init_from": "data/graph-embeddings/v2/graphsage_model_v2.pt"})
build_object_feat("home_v2-2", model_path="data/graph-embeddings/v2/graphmae_model_v2.pt")
```

## Layout

```
ObjectGraph/
├── config.py
├── graph_data.py    # scenes + graph build
├── encoder.py       # GraphSAGE
├── graphmae.py      # GraphMAE (SCE loss)
├── core.py          # train / train_mae / train_full
├── docs/ARCHITECTURE.md
└── README.md
```

## GraphMAE defaults

| Parameter | Value |
|-----------|--------|
| `mask_rate` | 0.75 |
| `mae_alpha` | 3.0 (SCE) |
| `mae_epochs` | 100 |
| `hidden_dim` | 64 |

These are the GraphMAE paper's defaults, carried over unchanged.

**Stage 2 is not recommended on this graph.** As originally written it could not train the
encoder at all (the latent re-mask in front of a linear decoder zeroes every encoder gradient —
`decoder.bias` was the only parameter that moved), so `graphmae_model_v2.pt` held a stage-1
encoder. With that repaired (`remask=False`, now the default) it measurably *hurts*: held-out
co-occurrence AUC 0.790 ± 0.003 → 0.752 ± 0.014, at every masking rate, schedule length and
loss exponent tested. Reconstructing each node's own MiniLM embedding pulls the representation
back toward the text modality this graph is meant to complement. Prefer `train()` over
`train_full()`; see `docs/tables/stage_ablation.tex` and `docs/paper-code-audit.md` §5.

See `config.DEFAULT_CFG` for all options.

## Evaluation protocol

`ObjectGraph/eval.py` scores the encoder intrinsically, by **held-out co-occurrence link
prediction** — so hyperparameters can be selected without running LATTICE per configuration.

- **Split.** Unique undirected edges, 80/10/10, stratified by co-occurrence count
  (c=1 / 2–4 / ≥5), never orphaning a node. `split_seed` is fixed for the whole study, so every
  configuration is scored on identical held-out edges.
- **No leakage.** Held-out edges are excluded from *message passing*, not just from the loss:
  a 2-layer SAGE would otherwise average the answer into the node. The tripwire is
  `train_auc`, which must sit clearly above `test_auc`.
- **Negatives.** The primary metric is AUC/AP against **degree-matched** negatives
  (P(node) ∝ deg). This is not optional: against uniform negatives a bare degree product
  scores 0.92 AUC and Adamic–Adar 0.96, so every configuration lands in a narrow band and
  nothing is distinguishable. Uniform-negative AUC is reported alongside for comparability.
- **Selection.** Best validation AUC, early stopping with patience, best weights restored.
  Test metrics are computed once, from the restored weights.
- **Baselines.** Adamic–Adar, common neighbours, preferential attachment, raw MiniLM cosine
  (no message passing), and untrained random-init SAGE. The trained encoder is only meaningful
  if it clears all five.

```bash
python scripts/eval_objectgraph.py --baselines        # stats + baseline table
python scripts/sweep_objectgraph.py --sweep stageA    # a sensitivity stage (resumable)
python scripts/report_objectgraph.py --tables --figures
```

Sweep outputs go to `data/graph-embeddings/sweeps/`; the study never writes to
`data/home_v2-2/`. See [docs/paper-code-audit.md](../docs/paper-code-audit.md) for where the
paper's description and this code diverge.

## Prepare training data (MIT Indoor + NYU)

```bash
python data/prepare-objectgraph/run.py
# or: download_mit_indoor.py then build_scenes.py
```

See [data/prepare-objectgraph/README.md](../data/prepare-objectgraph/README.md).

## Notebook

`jupyter notebooks/Graph_Modality_V2_2.ipynb`
