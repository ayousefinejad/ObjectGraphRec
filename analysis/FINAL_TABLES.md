# Final tables — object-graph study

Amazon Home & Kitchen (`home_v2-2`, 5-core: 14,503 items, 59,251 users). Object graph from MIT-Indoors + NYU-Depth. Every Δ is paired by seed; t is a paired t-statistic on 3 seeds, so |t| ≥ 4.303 is p < .05. Deltas below 0.0003 are unattributable — that is the measured spread between two identical runs.

## 1. Collaborative-filtering backbone

| Backbone | Depth | Modalities | R@20 | NDCG@20 | Seeds | Paired Δ R@20 vs image+text |
|---|---|---|---|---|---|---|
| **MF** | 0 | image+text | 0.04007 ± 0.00031 | 0.02005 ± 0.00025 | 3 | — |
| **MF** | 0 | + object | 0.04291 ± 0.00051 | 0.02103 ± 0.00019 | 3 | +0.00284 (+7.1%) · t=+13.38 ✅ |
| **NGCF** | 1 | image+text | 0.03809 ± 0.00084 | 0.01753 ± 0.00028 | 3 | — |
| **NGCF** | 1 | + object | 0.03802 ± 0.00089 | 0.01780 ± 0.00031 | 3 | -0.00007 (-0.2%) · t=-0.53 n.s. · below the 0.0003 floor |
| **LightGCN** | 3 | image+text | 0.04340 ± 0.00069 | 0.02198 ± 0.00042 | 3 | — |
| **LightGCN** | 3 | + object | 0.04372 ± 0.00069 | 0.02199 ± 0.00037 | 3 | +0.00032 (+0.7%) · t=+0.99 n.s. |

Depth is `len(--weight_size)`, the number of user–item propagation hops. **MF is LightGCN at depth 0** — with no layers the loop never runs and the branch returns the same tensors (`Models.py:735` vs `:747`), so this is one depth axis, not three unrelated architectures.

NGCF is reported at depth 1, not 2: it scores *higher* at depth 1 than depth 2 on image+text (0.03906 vs 0.03698, single seeds in `tuning.csv`), the opposite of LightGCN — so depth 2, inherited from the MF recipe, would have handicapped it. Both NGCF arms here are run at depth 1, `--epoch 400`, all three seeds, so the pair is protocol-matched.

## 2. Fusion of the modalities

| Fusion of the three modalities | R@20 | NDCG@20 | Seeds | Paired Δ vs published |
|---|---|---|---|---|
| **softmax, learned** (published) | 0.04291 ± 0.00051 | 0.02103 ± 0.00019 | 3 | — (reference) |
| frozen (weights pinned at uniform) | 0.04304 ± 0.00055 | 0.02107 ± 0.00020 | 3 | +0.00013 (+0.3%) · t=+2.97 n.s. · below the 0.0003 floor |
| gated (per-item n×3 gate) | 0.04298 ± 0.00037 | 0.02101 ± 0.00010 | 3 | +0.00007 (+0.2%) · t=+0.54 n.s. · below the 0.0003 floor |
| softmax, lr_fusion 0.05 (100× step) | 0.04254 ± 0.00038 | 0.02091 ± 0.00013 | 3 | -0.00037 (-0.9%) · t=-0.99 n.s. |

Learned weights α = softmax(modal_weight) at end of training, across 71 object-bearing runs: **image 0.3453 ± 0.0042, text 0.3297 ± 0.0049, object 0.3250 ± 0.0058** — under 0.02 from the uniform 1/3 initialisation.

The reason is structural, not a tuning failure: `item_adj` is rebuilt on batch 0 of each epoch and detached for the other ~149, so these three scalars receive ~130 gradient steps in a whole run. **`frozen` is the arm that matters** — if pinning α at uniform costs nothing, the adaptive fusion is inert and should not be claimed as a contribution.

**What α does when it is allowed to move.** At `--lr_fusion 0.05` (100× step) the weights leave uniform decisively and land, across all three seeds, at **image ≈ 0.576, text ≈ 0.366, object ≈ 0.058** — the model drives the object modality's weight down roughly 6×. Yet that arm scores 0.04254 vs the published 0.04291 (n.s.), while removing the object modality outright costs −6.6%. So the object graph's contribution is real but saturates at a small weight: the published uniform α over-weights it, and correcting that neither helps nor hurts measurably.

α is reported only for runs where the object modality is on: `fusion_state()` prints softmax(modal_weight) *unmasked*, while `_modal_weights()` masks the logits before the softmax, so an image+text run prints 0.3333 for a modality whose effective weight is exactly 0.

### 2b. Per-modality contribution (leave-one-out)

| Modalities | R@20 | NDCG@20 | Seeds | Paired Δ vs full |
|---|---|---|---|---|
| image + text + object (full) | 0.04291 ± 0.00051 | 0.02103 ± 0.00019 | 3 | — (reference) |
| image + text  (− object) | 0.04007 ± 0.00031 | 0.02005 ± 0.00025 | 3 | -0.00284 (-6.6%) · t=-13.38 ❌ |
| image + object  (− text) | 0.03662 ± 0.00103 | 0.01727 ± 0.00036 | 3 | -0.00629 (-14.7%) · t=-8.03 ❌ |
| text + object  (− image) | 0.04514 ± 0.00058 | 0.02155 ± 0.00038 | 3 | +0.00223 (+5.2%) · t=+3.99 n.s. |
| object only | 0.02810 ± 0.00039 | 0.01246 ± 0.00019 | 3 | -0.01481 (-34.5%) · t=-29.21 ❌ |

## 3. Negative sampling

| Setting | Value | Source |
|---|---|---|
| Negatives per positive | **1** | `load_data.py:186-189` |
| Sampled from | all 14,503 items, uniformly | `sample_neg_items_for_u`, line 172 |
| Rejection | any item in that user's train set | line 177 |
| Resampled | every batch (a positive meets a new negative each epoch) | line 152 |
| Positives per step | 1 per sampled user, uniform from their train items | line 187 |
| Triples per step | 1,024 (batch size) | `--batch_size` |
| Loss | BPR: −log σ(score(u,i⁺) − score(u,i⁻)) | `main.py:224-231` |
| Regularisation | L2 on the three embedding tensors, 1e-5 | `--regs[0]` |
Reported, not ablated. Varying k would need `sample()` and `bpr_loss` reshaped for k negatives per positive; the study holds it at the published value of 1 throughout, so it is a constant across every arm in the tables above rather than a free variable.
