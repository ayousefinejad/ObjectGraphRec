# Inference comparison — LATTICE vs LATTICE + object graph

Same recommender, same split, same seeds; the two models differ only in whether the object modality is present. Scoring reproduces `utility/batch_test.py` exactly (59,251 users, 60,178 held-out items, top-20 candidates = all items minus the user's train set).

## 1. What changes across all users

| Seed | Held-out item retrieved by | | | | Top-10 list overlap |
|---|---|---|---|---|---|
| | **+object only** (rescued) | **image+text only** (lost) | both | neither | Jaccard |
| 0 | **723** | 556 | 1918 | 56,981 | 0.417 |
| 1 | **734** | 540 | 1893 | 57,011 | 0.401 |
| 2 | **695** | 544 | 1886 | 57,053 | 0.413 |

Mean over 3 seeds: **717 items rescued** by the object modality against **547 lost** — a net gain of 171 retrieved held-out items per seed.

The two models' top-10 lists overlap by only **41.1%** on average, so they are not small perturbations of one another: adding the object modality reorders most users' recommendations, and the net metric gain is the residue of many compensating changes.

Among items both models retrieve, mean rank is 5.77 (image+text) vs 5.79 (+object) — the object modality also *promotes* items it does not newly retrieve.

## 2. Where the change lands (F18 coverage tiers)

| Tier | Held-out items | Retrieved, image+text | Retrieved, +object | Δ |
|---|---|---|---|---|
| exact | 29,919 | 1077 (3.60%) | 1157 (3.87%) | **+79** (+7.4%) |
| near | 23,299 | 1030 (4.42%) | 1118 (4.80%) | **+88** (+8.6%) |
| unreached | 6,960 | 339 (4.87%) | 342 (4.91%) | **+3** (+0.9%) |

## 3. Worked examples

Seed 0. Sampled uniformly (seed 0) from the 358 users whose held-out item the object modality rescued into the top-20 — a stated rule, not hand-picked. `>>` marks the held-out item.

**User 15395**

| | |
|---|---|
| Training history (20 items) | knife bar, shower curtain rings, pot rack, dish, toilet brush, bed skirt |
| **Ground truth** (held out, 2) | **shower curtain** — Home & Kitchen Bath Bathroom Accessories Shower Curtains, Ho; **cabinet** — Home & Kitchen Furniture Other Furniture Cabinets Wall-Mount |

| Rank | image + text | ✓ | image + text + object | ✓ |
|---|---|---|---|---|
| 1 | shower curtain liner |  | shower curtain liner |  |
| 2 | keeper |  | hooks |  |
| 3 | shower curtain rod |  | shower curtain | **GT** |
| 4 | pillow |  | shower curtain |  |
| 5 | shower caddy |  | shower curtain |  |
| 6 | cloth |  | cloth |  |
| 7 | rack |  | shower curtain |  |
| 8 | shower rod |  | shower curtain liner |  |
| 9 | shower caddy |  | shower curtain liner |  |
| 10 | cabinet rack |  | shower liner |  |

  held-out item rank: image+text not in top-20, +object 3

**User 29734**

| | |
|---|---|
| Training history (3 items) | mattress topper, bed frame, pillows |
| **Ground truth** (held out, 1) | **mattress** — Home & Kitchen Furniture Bedroom Furniture Mattresses & Box  |

| Rank | image + text | ✓ | image + text + object | ✓ |
|---|---|---|---|---|
| 1 | bed frame |  | bed frame |  |
| 2 | caddy |  | bed frame |  |
| 3 | table |  | dispenser |  |
| 4 | mattress topper |  | mattress pad |  |
| 5 | pillow |  | blender |  |
| 6 | dispenser |  | mattress protector |  |
| 7 | mattress protector |  | pillow |  |
| 8 | lunch box |  | bed frame |  |
| 9 | table |  | bed frame |  |
| 10 | fan |  | pillow |  |

  held-out item rank: image+text not in top-20, +object 17

**User 25858**

| | |
|---|---|
| Training history (3 items) | organizer, recycle bags, bento box |
| **Ground truth** (held out, 1) | **bento box** — Home & Kitchen Kitchen & Dining Storage & Organization Lunch |

| Rank | image + text | ✓ | image + text + object | ✓ |
|---|---|---|---|---|
| 1 | kettle |  | kettle |  |
| 2 | cleaner |  | hooks |  |
| 3 | turntable |  | sink strainer |  |
| 4 | sink strainer |  | ice tray |  |
| 5 | rice cooker |  | bento box |  |
| 6 | organizer |  | shower hooks |  |
| 7 | jar |  | jar |  |
| 8 | bento box |  | rice cooker |  |
| 9 | organizer |  | mattress protector |  |
| 10 | containers |  | spice jar set |  |

  held-out item rank: image+text not in top-20, +object 13
