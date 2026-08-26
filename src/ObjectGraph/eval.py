"""Intrinsic evaluation for the object-graph encoder.

The encoder is trained self-supervised on co-occurrence edges, so the natural intrinsic task
is held-out co-occurrence link prediction. Nothing here touches the recommender: these
metrics score the graph encoder itself, so hyperparameters can be selected without running
LATTICE for every configuration.

Two protocol choices matter more than anything else in this file:

1. Leakage. The original train() supervises on *all* edges and evaluates nothing, so any AUC
   measured on those edges is meaningless. Held-out edges must be excluded from message
   passing too, not just from the loss -- a 2-layer SAGE would otherwise average the answer
   into the node representation.

2. Negative sampling. Against uniformly sampled negatives, a bare degree product scores
   ~0.92 AUC on this graph because degree is extremely skewed (max 532, median 11). Every
   configuration then lands in a narrow high band and the sweep cannot discriminate. Scoring
   against degree-matched negatives removes that shortcut. See sample_negatives().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# Co-occurrence strata used for stratified reporting: singletons, occasional, frequent.
STRATA: tuple[tuple[str, int, int], ...] = (("c=1", 1, 1), ("c=2-4", 2, 4), ("c>=5", 5, 10**9))


def stratum_of(count: int) -> str:
    for name, lo, hi in STRATA:
        if lo <= count <= hi:
            return name
    return STRATA[-1][0]


@dataclass
class EdgeSplit:
    """Disjoint train/val/test edge sets over unique undirected pairs.

    train_pairs doubles as the message-passing structure; val/test pairs are scored but never
    propagated. Built once per study and shared by every configuration so that all configs
    are compared on identical held-out edges.
    """

    n_nodes: int
    train_pairs: np.ndarray  # (E_train, 2), i < j
    val_pairs: np.ndarray
    test_pairs: np.ndarray
    train_counts: np.ndarray  # c_ab aligned with train_pairs
    val_counts: np.ndarray
    test_counts: np.ndarray
    node_counts: np.ndarray  # c_a, scenes containing each object
    nodes: list[str] = field(default_factory=list)

    @property
    def train_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(deg, self.train_pairs[:, 0], 1)
        np.add.at(deg, self.train_pairs[:, 1], 1)
        return deg

    def all_pairs_set(self) -> set[tuple[int, int]]:
        """Every real edge, used to reject false negatives when sampling."""
        out = set()
        for arr in (self.train_pairs, self.val_pairs, self.test_pairs):
            out.update(map(tuple, arr.tolist()))
        return out

    def message_passing(
        self, edge_mode: str = "multiplicity", device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """edge_index (and optional weights) built from TRAIN edges only."""
        src, dst, wts = [], [], []
        for (i, j), c in zip(map(tuple, self.train_pairs.tolist()), self.train_counts.tolist()):
            reps = c if edge_mode == "multiplicity" else 1
            for _ in range(reps):
                src += [i, j]
                dst += [j, i]
            if edge_mode == "weighted":
                w = c / float(np.sqrt(self.node_counts[i] * self.node_counts[j]))
                wts += [w, w]
        ei = torch.tensor([src, dst], dtype=torch.long, device=device)
        ew = torch.tensor(wts, dtype=torch.float, device=device) if edge_mode == "weighted" else None
        return ei, ew


def split_edges(
    pair_counts: dict[tuple[int, int], int],
    node_counts: list[int],
    n_nodes: int,
    nodes: list[str] | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 42,
    protect_degree: bool = True,
    stratify: bool = True,
) -> EdgeSplit:
    """Split unique undirected edges into train/val/test.

    protect_degree refuses to hold out an edge that would leave either endpoint with zero
    training edges: an isolated node receives no message passing, so scoring it measures the
    feature encoder rather than the graph encoder. On this graph the constraint is nearly
    free (a handful of edges are retained that otherwise would not be).

    stratify splits within co-occurrence strata so the held-out sets mirror the global mix
    (this graph is 62% singleton edges) and per-stratum AUC is reportable.
    """
    rng = np.random.default_rng(split_seed)
    pairs = np.array(sorted(pair_counts.keys()), dtype=np.int64)
    counts = np.array([pair_counts[tuple(p)] for p in pairs.tolist()], dtype=np.int64)

    groups = (
        [np.where([lo <= c <= hi for c in counts])[0] for _, lo, hi in STRATA]
        if stratify
        else [np.arange(len(pairs))]
    )

    deg = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(deg, pairs[:, 0], 1)
    np.add.at(deg, pairs[:, 1], 1)

    val_idx: list[int] = []
    test_idx: list[int] = []
    for g in groups:
        if len(g) == 0:
            continue
        order = rng.permutation(g)
        n_val, n_test = int(round(val_frac * len(g))), int(round(test_frac * len(g)))
        want = {"val": n_val, "test": n_test}
        for idx in order:
            target = "val" if want["val"] > 0 else ("test" if want["test"] > 0 else None)
            if target is None:
                break
            u, v = pairs[idx]
            if protect_degree and (deg[u] <= 1 or deg[v] <= 1):
                continue  # holding this out would orphan an endpoint
            deg[u] -= 1
            deg[v] -= 1
            (val_idx if target == "val" else test_idx).append(int(idx))
            want[target] -= 1

    held = np.zeros(len(pairs), dtype=bool)
    held[val_idx + test_idx] = True
    tr = np.where(~held)[0]
    va, te = np.array(sorted(val_idx), dtype=np.int64), np.array(sorted(test_idx), dtype=np.int64)
    return EdgeSplit(
        n_nodes=n_nodes,
        train_pairs=pairs[tr],
        val_pairs=pairs[va],
        test_pairs=pairs[te],
        train_counts=counts[tr],
        val_counts=counts[va],
        test_counts=counts[te],
        node_counts=np.asarray(node_counts, dtype=np.int64),
        nodes=list(nodes or []),
    )


def sample_negatives(
    n_nodes: int,
    n_samples: int,
    forbidden: set[tuple[int, int]],
    mode: str = "degree",
    degree: np.ndarray | None = None,
    seed: int = 7,
    alpha: float = 1.0,
) -> np.ndarray:
    """Sample non-edges.

    mode='uniform' picks endpoints uniformly. mode='degree' picks them with P(node) prop to
    deg^alpha, removing the degree-product shortcut that makes uniform negatives trivially
    separable on this graph (uniform: Adamic-Adar 0.954, preferential attachment 0.921).

    alpha=1.0 is the default because it reproduces the marginal endpoint-degree distribution
    of a real edge set exactly -- drawing P(node) prop to deg is the same law as "pick a
    random edge, then pick a random endpoint of it". Measured on this graph:

        alpha   neg deg mean   AA auc   PA auc
        0.00 (uniform)  24.0   0.9545   0.9213
        0.50            45.3   0.8746   0.8161
        0.75            59.2   0.8095   0.7355
        1.00            73.4   0.7438   0.6574
        1.25            86.3   0.6686   0.5707

    Positive endpoints average degree 109.0, still above alpha=1.0's 73.4: real edges are
    degree-assortative, while negatives draw their two endpoints independently. PA's residual
    0.657 is exactly that leftover assortativity, not a sampler defect -- alpha>1 would only
    hide it by over-correcting.
    """
    rng = np.random.default_rng(seed)
    if mode == "degree":
        if degree is None:
            raise ValueError("mode='degree' requires degree")
        p = np.power(degree.astype(np.float64), alpha)
        if p.sum() <= 0:
            raise ValueError("degenerate degree distribution")
        p /= p.sum()
    else:
        p = None

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(out) < n_samples:
        draw = max(n_samples - len(out), 1024) * 2
        u = rng.choice(n_nodes, size=draw, p=p)
        v = rng.choice(n_nodes, size=draw, p=p)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= n_samples:
                break
    return np.array(out, dtype=np.int64)


def _as_numpy(z: torch.Tensor | np.ndarray) -> np.ndarray:
    return z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)


def score_pairs(z: torch.Tensor | np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for each pair -- the same quantity the contrastive loss uses."""
    zz = _as_numpy(z)
    zz = zz / np.clip(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12, None)
    return np.sum(zz[pairs[:, 0]] * zz[pairs[:, 1]], axis=1)


def _auc_ap(pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def link_pred_metrics(
    z: torch.Tensor | np.ndarray, pos_pairs: np.ndarray, neg_pairs: np.ndarray
) -> dict[str, float]:
    return _auc_ap(score_pairs(z, pos_pairs), score_pairs(z, neg_pairs))


def stratified_metrics(
    z: torch.Tensor | np.ndarray,
    pos_pairs: np.ndarray,
    pos_counts: np.ndarray,
    neg_pairs: np.ndarray,
) -> dict[str, float]:
    """AUC per co-occurrence stratum against a shared negative pool.

    AUC only (not AP): AUC is invariant to class balance, so strata of different sizes stay
    comparable against one negative set.
    """
    neg = score_pairs(z, neg_pairs)
    out: dict[str, float] = {}
    for name, lo, hi in STRATA:
        m = (pos_counts >= lo) & (pos_counts <= hi)
        if m.sum() == 0:
            out[f"auc_{name}"] = float("nan")
            continue
        pos = score_pairs(z, pos_pairs[m])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        out[f"auc_{name}"] = float(roc_auc_score(y, np.concatenate([pos, neg])))
        out[f"n_{name}"] = int(m.sum())
    return out


def degree_bucket_metrics(
    z: torch.Tensor | np.ndarray,
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    degree: np.ndarray,
    n_buckets: int = 4,
) -> dict[str, float]:
    """AUC by endpoint degree quantile -- exposes hub bias in the embedding."""
    neg = score_pairs(z, neg_pairs)
    key = np.minimum(degree[pos_pairs[:, 0]], degree[pos_pairs[:, 1]])
    edges = np.quantile(key, np.linspace(0, 1, n_buckets + 1))
    out: dict[str, float] = {}
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        m = (key >= lo) & (key <= hi) if b == n_buckets - 1 else (key >= lo) & (key < hi)
        if m.sum() < 10:
            continue
        pos = score_pairs(z, pos_pairs[m])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        out[f"auc_degq{b + 1}"] = float(roc_auc_score(y, np.concatenate([pos, neg])))
    return out


def alignment_uniformity(
    z: torch.Tensor | np.ndarray, pos_pairs: np.ndarray, t: float = 2.0, max_nodes: int = 1500
) -> dict[str, float]:
    """Wang & Isola (2020). Low alignment = linked objects are close; low uniformity = the
    embedding spreads over the sphere rather than collapsing."""
    zz = _as_numpy(z)
    zz = zz / np.clip(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12, None)
    d = zz[pos_pairs[:, 0]] - zz[pos_pairs[:, 1]]
    align = float(np.mean(np.sum(d * d, axis=1)))
    sub = zz if len(zz) <= max_nodes else zz[np.random.default_rng(0).choice(len(zz), max_nodes, replace=False)]
    sq = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(len(sub), k=1)
    unif = float(np.log(np.mean(np.exp(-t * sq[iu]))))
    return {"alignment": align, "uniformity": unif}


def effective_rank(z: torch.Tensor | np.ndarray) -> float:
    """Participation ratio of the singular values: how many dimensions are actually used.
    If this is far below hidden_dim, the hidden_dim axis of the sweep is a no-op."""
    zz = _as_numpy(z)
    s = np.linalg.svd(zz - zz.mean(0, keepdims=True), compute_uv=False)
    return float((s.sum() ** 2) / max((s**2).sum(), 1e-12))


def degree_bias(z: torch.Tensor | np.ndarray, degree: np.ndarray) -> float:
    """Spearman rho between node degree and mean cosine to all other nodes. A large positive
    value means hubs sit near the centroid and look similar to everything."""
    from scipy.stats import spearmanr

    zz = _as_numpy(z)
    zz = zz / np.clip(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12, None)
    sim = zz @ zz.T
    np.fill_diagonal(sim, 0.0)
    return float(spearmanr(degree, sim.sum(1) / (len(zz) - 1)).statistic)


# --------------------------------------------------------------------------------------
# Baselines. These decide whether the study means anything: if a trained encoder cannot beat
# Adamic-Adar or raw text similarity, that is the finding.
# --------------------------------------------------------------------------------------


def _train_adjacency(split: EdgeSplit) -> np.ndarray:
    a = np.zeros((split.n_nodes, split.n_nodes), dtype=np.float64)
    a[split.train_pairs[:, 0], split.train_pairs[:, 1]] = 1.0
    a[split.train_pairs[:, 1], split.train_pairs[:, 0]] = 1.0
    return a


def baseline_scores(split: EdgeSplit, pos_pairs: np.ndarray, neg_pairs: np.ndarray) -> dict[str, dict[str, float]]:
    """Non-neural structural link predictors computed from TRAIN edges only."""
    a = _train_adjacency(split)
    deg = a.sum(1)
    cn = a @ a
    w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 1.0000001)), 0.0)
    aa = (a * w[None, :]) @ a

    def pick(m: np.ndarray, p: np.ndarray) -> np.ndarray:
        return m[p[:, 0], p[:, 1]]

    out = {}
    for name, mat in (("adamic_adar", aa), ("common_neighbors", cn)):
        out[name] = _auc_ap(pick(mat, pos_pairs), pick(mat, neg_pairs))
    pa_pos = deg[pos_pairs[:, 0]] * deg[pos_pairs[:, 1]]
    pa_neg = deg[neg_pairs[:, 0]] * deg[neg_pairs[:, 1]]
    out["preferential_attachment"] = _auc_ap(pa_pos, pa_neg)
    return out


def feature_baseline(x: torch.Tensor | np.ndarray, pos_pairs: np.ndarray, neg_pairs: np.ndarray) -> dict[str, float]:
    """Raw MiniLM cosine with no message passing. Node features are text embeddings, so this
    measures how much of the co-occurrence structure is already predictable from label text
    alone -- i.e. how much the graph actually adds."""
    return link_pred_metrics(x, pos_pairs, neg_pairs)


# --------------------------------------------------------------------------------------
# Weak room labels (diagnostic only)
# --------------------------------------------------------------------------------------

ROOM_ANCHORS: dict[str, tuple[str, ...]] = {
    "bathroom": ("Toilet", "Bathtub", "Shower curtain", "Shower", "Towel rack"),
    "bedroom": ("Bed", "Crib", "Nightstand", "Headboard"),
    "kitchen": ("Stove", "Oven", "Refrigerator", "Microwave", "Dishwasher"),
    "dining_room": ("Dining table", "Dining chair", "Buffet"),
    "livingroom": ("Sofa", "Television", "Coffee table", "Fireplace"),
}


def weak_room_labels(scenes: list[list[str]], nodes: list[str]) -> np.ndarray:
    """Assign each node a room by majority vote over scenes containing it, where a scene's
    room is decided by anchor objects.

    Diagnostic only, and partly circular: the labels are derived from object identity, so
    clustering by room is partly guaranteed by construction. The true scene->room mapping is
    unrecoverable because scenes.json is a flat list with no image keys.
    """
    rooms = list(ROOM_ANCHORS)
    n2i = {n: i for i, n in enumerate(nodes)}
    tally = np.zeros((len(nodes), len(rooms)), dtype=np.int64)
    for scene in scenes:
        s = set(scene)
        hits = [r for r, anchors in ROOM_ANCHORS.items() if s & set(anchors)]
        if len(hits) != 1:
            continue  # ambiguous or unlabelled scene
        r = rooms.index(hits[0])
        for obj in s:
            if obj in n2i:
                tally[n2i[obj], r] += 1
    labels = np.full(len(nodes), -1, dtype=np.int64)
    hit = tally.sum(1) > 0
    labels[hit] = tally[hit].argmax(1)
    return labels


def knn_room_purity(z: torch.Tensor | np.ndarray, labels: np.ndarray, k: int = 10) -> dict[str, float]:
    zz = _as_numpy(z)
    zz = zz / np.clip(np.linalg.norm(zz, axis=1, keepdims=True), 1e-12, None)
    m = labels >= 0
    if m.sum() < k + 1:
        return {"knn_room_purity": float("nan"), "n_labelled": int(m.sum())}
    sub, lab = zz[m], labels[m]
    sim = sub @ sub.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    purity = float(np.mean((lab[nn] == lab[:, None]).mean(1)))
    return {"knn_room_purity": purity, "n_labelled": int(m.sum())}


def evaluate_embeddings(
    z: torch.Tensor | np.ndarray,
    split: EdgeSplit,
    neg_val: np.ndarray,
    neg_test: np.ndarray,
    neg_test_uniform: np.ndarray | None = None,
    scenes: list[list[str]] | None = None,
) -> dict[str, float]:
    """Full metric bundle for one embedding matrix."""
    deg = split.train_degree
    out: dict[str, float] = {}
    for k, v in link_pred_metrics(z, split.val_pairs, neg_val).items():
        out[f"val_{k}"] = v
    for k, v in link_pred_metrics(z, split.test_pairs, neg_test).items():
        out[f"test_{k}"] = v
    if neg_test_uniform is not None:
        for k, v in link_pred_metrics(z, split.test_pairs, neg_test_uniform).items():
            out[f"test_{k}_uniform"] = v
    out.update(stratified_metrics(z, split.test_pairs, split.test_counts, neg_test))
    out.update(degree_bucket_metrics(z, split.test_pairs, neg_test, deg))
    out.update(alignment_uniformity(z, split.test_pairs))
    out["effective_rank"] = effective_rank(z)
    out["degree_bias_rho"] = degree_bias(z, deg)
    if scenes is not None and split.nodes:
        out.update(knn_room_purity(z, weak_room_labels(scenes, split.nodes)))
    return out
