"""Graph modality: contrastive pre-training, GraphMAE fine-tuning, item features."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from . import eval as ev
from .config import DEFAULT_CFG, ROOT
from .encoder import GraphEncoder
from .graph_data import build_cooccurrence, build_graph, load_graph_data, load_scenes, scenes_from_benchmark
from .graphmae import GraphMAE


def _contrastive(z, pos, neg, t):
    sim = lambda a, b: F.cosine_similarity(a, b)
    logits = torch.cat([sim(z[pos[0]], z[pos[1]]) / t, sim(z[neg[0]], z[neg[1]]) / t])
    labels = torch.cat(
        [torch.ones(pos.size(1), device=z.device), torch.zeros(neg.size(1), device=z.device)]
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def _save_checkpoint(path: Path, nodes, emb, cfg, state_dict, method: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "model_state_dict": state_dict,
            "final_embeddings": emb.cpu(),
            "nodes": nodes,
            "cfg": cfg,
        },
        path,
    )


def train(scenes=None, cfg: dict | None = None) -> Path:
    """Stage 1: contrastive GraphSAGE on co-occurrence edges."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes, data = load_graph_data(scenes, cfg, device)
    model = GraphEncoder(data.x.size(1), cfg["hidden_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for _ in tqdm(range(cfg["epochs"]), desc="Contrastive"):
        opt.zero_grad()
        z = model(data.x, data.edge_index)
        neg = negative_sampling(data.edge_index, data.num_nodes, data.edge_index.size(1), method="sparse")
        _contrastive(z, data.edge_index, neg, cfg["temperature"]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)
    out = Path(cfg["model_path"])
    _save_checkpoint(out, nodes, emb, cfg, model.state_dict(), "contrastive")
    return out


def train_mae(scenes=None, cfg: dict | None = None) -> Path:
    """Stage 2: GraphMAE masked feature reconstruction (fine-tune encoder)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes, data = load_graph_data(scenes, cfg, device)
    model = GraphMAE(
        data.x.size(1),
        cfg["hidden_dim"],
        mask_rate=cfg["mask_rate"],
        alpha=cfg["mae_alpha"],
        replace_rate=cfg["replace_rate"],
        remask=cfg["remask"],
    ).to(device)
    # init_from=None means NO warm start. Previously this silently fell back to
    # cfg["model_path"], so "stage-2 only" became "stage-1 -> stage-2" whenever a stage-1
    # checkpoint happened to exist on disk, making the stage ablation measure the wrong thing.
    init = cfg.get("init_from")
    if init and Path(init).exists():
        ckpt = torch.load(init, map_location=device, weights_only=False)
        model.load_encoder_checkpoint(ckpt["model_state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["mae_lr"])
    model.train()
    for _ in tqdm(range(cfg["mae_epochs"]), desc="GraphMAE"):
        opt.zero_grad()
        model(data.x, data.edge_index).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        emb = model.embed(data.x, data.edge_index)
    out = Path(cfg.get("graphmae_path", DEFAULT_CFG["graphmae_path"]))
    _save_checkpoint(out, nodes, emb, cfg, model.encoder.state_dict(), "graphmae")
    return out


def train_full(scenes=None, cfg: dict | None = None) -> Path:
    """Contrastive pre-train, then GraphMAE fine-tuning."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    p = train(scenes, cfg)
    return train_mae(scenes, {**cfg, "init_from": str(p)})


# ======================================================================================
# Split-aware training for the parameter study.
#
# train() above keeps its original contract -- it supervises on every edge and evaluates
# nothing -- because build_object_feat() and the shipped pipeline depend on it. Everything
# below is the study path: held-out edges are excluded from message passing as well as from
# the loss, every run is seeded, and selection is on validation AUC with best-weight restore.
# ======================================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_FEATURE_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def encode_nodes(nodes: list[str], text_encoder: str, device: torch.device) -> torch.Tensor:
    """MiniLM node features, cached by (encoder, node-list hash).

    Encoding 1,068 labels costs far more than a whole training run, and the node list is
    identical across every configuration that does not change min_cooc, so this is the
    difference between a 3-hour sweep and a 30-hour one.
    """
    key = (text_encoder, hashlib.sha256("\x00".join(nodes).encode()).hexdigest())
    if key not in _FEATURE_CACHE:
        enc = SentenceTransformer(text_encoder, device=str(device))
        with torch.no_grad():
            x = enc.encode(nodes, convert_to_tensor=True, device=device)
        # sentence-transformers 5.x encodes under inference_mode; those tensors cannot be
        # saved for backward.
        _FEATURE_CACHE[key] = x.clone().detach().cpu()
    return _FEATURE_CACHE[key].to(device)


@dataclass
class StudyData:
    """Everything a run needs that does not depend on model hyperparameters.

    Built once per graph-construction setting and reused across every optimizer/architecture
    cell, so all configurations are scored on identical held-out edges and identical negatives.
    """

    nodes: list[str]
    scenes: list[list[str]]
    x: torch.Tensor
    split: ev.EdgeSplit
    neg_val: np.ndarray
    neg_test: np.ndarray
    neg_val_uniform: np.ndarray
    neg_test_uniform: np.ndarray
    edge_index: torch.Tensor
    edge_weight: torch.Tensor | None
    train_pos: torch.Tensor  # (2, 2*E_train) both directions, for the contrastive loss
    scenes_sha256: str


def prepare_study(cfg: dict | None = None, scenes=None, device: torch.device | None = None) -> StudyData:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = scenes or load_scenes(cfg)
    sha = hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest()

    nodes, pair_counts, node_counts = build_cooccurrence(scenes)
    if cfg["min_cooc"] > 1:
        pair_counts = {k: c for k, c in pair_counts.items() if c >= cfg["min_cooc"]}
    split = ev.split_edges(
        pair_counts,
        node_counts,
        len(nodes),
        nodes=nodes,
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        split_seed=cfg["split_seed"],
    )

    deg = split.train_degree
    forb = split.all_pairs_set()
    ns = cfg["neg_seed"]
    n = len(nodes)
    nv, nt = len(split.val_pairs), len(split.test_pairs)
    neg_val = ev.sample_negatives(n, nv, forb, "degree", deg, seed=ns, alpha=cfg["neg_alpha"])
    neg_test = ev.sample_negatives(n, nt, forb, "degree", deg, seed=ns + 1, alpha=cfg["neg_alpha"])
    neg_val_u = ev.sample_negatives(n, nv, forb, "uniform", seed=ns + 2)
    neg_test_u = ev.sample_negatives(n, nt, forb, "uniform", seed=ns + 3)

    ei, ew = split.message_passing(cfg["edge_mode"], device=device)
    tp = torch.tensor(split.train_pairs.T, dtype=torch.long, device=device)
    tp = torch.cat([tp, tp.flip(0)], dim=1)

    return StudyData(
        nodes=nodes,
        scenes=scenes,
        x=encode_nodes(nodes, cfg["text_encoder"], device),
        split=split,
        neg_val=neg_val,
        neg_test=neg_test,
        neg_val_uniform=neg_val_u,
        neg_test_uniform=neg_test_u,
        edge_index=ei,
        edge_weight=ew,
        train_pos=tp,
        scenes_sha256=sha,
    )


def _mined_negatives(sd: StudyData, cfg: dict, n_samples: int, z: torch.Tensor,
                     device, band: tuple[float, float] | None) -> torch.Tensor:
    """Embedding-space negative mining: rank every non-edge pair by current cosine similarity
    and draw negatives from either the top of that ranking ('hard', band=None) or a percentile
    band within it ('semihard', band=(lo, hi)).

    Recomputed from the LIVE z every call -- unlike 'uniform'/'degree', which depend only on
    graph structure fixed at prepare_study time, this pool shifts as training moves the
    embeddings. That is what makes 'semihard' dynamic rather than a static hard-negative set.

    O(n^2) is deliberate and cheap at this graph's scale (<=4,081 nodes here, ~1,068 for
    MIT+NYU: a 1068x1068 similarity matrix is ~1 MB), recomputed once per epoch alongside the
    forward pass that already produces z -- there is no cheaper structure to exploit for a
    graph this small, and the alternative (an ANN index) would add a dependency to save
    microseconds.
    """
    n = z.size(0)
    forbidden = torch.zeros(n, n, dtype=torch.bool, device=device)
    forbidden[sd.train_pos[0], sd.train_pos[1]] = True
    forbidden.fill_diagonal_(True)                       # no self-pairs
    sim = (F.normalize(z, dim=1) @ F.normalize(z, dim=1).T).masked_fill(forbidden, float("-inf"))
    rank = sim.argsort(dim=1, descending=True)            # per-row, hardest (most similar) first
    valid = (~forbidden).sum(dim=1).clamp_min(1)          # candidates per anchor, self+edges excluded
    if band is None:
        col = torch.zeros(n, dtype=torch.long, device=device)               # rank 0: hardest
    else:
        lo, hi = band
        frac = lo + (hi - lo) * torch.rand(n, device=device)
        col = (frac * (valid - 1).clamp_min(0)).long()                      # within [lo, hi) of the ranking
    v_hard = rank.gather(1, col.unsqueeze(1)).squeeze(1)
    anchors = torch.randint(0, n, (n_samples,), device=device)
    return torch.stack([anchors, v_hard[anchors]])


def _train_negatives(sd: StudyData, cfg: dict, n_samples: int, device,
                     z: torch.Tensor | None = None) -> torch.Tensor:
    """Negatives for the contrastive loss. Resampled every epoch, unlike the evaluation
    negatives which are drawn once and frozen."""
    mode = cfg["neg_mode"]
    if mode == "degree":
        deg = sd.split.train_degree
        p = torch.tensor(np.power(deg, cfg["neg_alpha"]), dtype=torch.float, device=device)
        p = p / p.sum()
        u = torch.multinomial(p, n_samples, replacement=True)
        v = torch.multinomial(p, n_samples, replacement=True)
        keep = u != v
        return torch.stack([u[keep], v[keep]])
    if mode == "hard":
        assert z is not None, "neg_mode='hard' needs the live embedding z"
        return _mined_negatives(sd, cfg, n_samples, z, device, band=None)
    if mode == "semihard":
        assert z is not None, "neg_mode='semihard' needs the live embedding z"
        return _mined_negatives(sd, cfg, n_samples, z, device, band=cfg["semihard_band"])
    return negative_sampling(sd.train_pos, sd.split.n_nodes, n_samples, method="sparse")


def train_eval(
    cfg: dict | None = None,
    sd: StudyData | None = None,
    scenes=None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    """One seeded, split-aware, early-stopped run. Returns metrics + history + state.

    Selection is on validation AUC against degree-matched negatives; test metrics are computed
    once, from the restored best weights, and never used for selection.

    ``cfg["stage"]`` chooses the training recipe:

        's1'      contrastive only -- the shipped train() objective (default)
        's2'      GraphMAE masked feature reconstruction only, from random init
        's1->s2'  contrastive, then GraphMAE warm-started from it -- what train_full() does

    Both GraphMAE arms propagate E_train only, exactly like the contrastive arm, so the
    reconstruction target of a held-out edge's endpoint is never reachable through the graph.
    Stage 2 is early-stopped on validation link-prediction AUC rather than on reconstruction
    loss, so every arm in the study is selected by the same criterion.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = sd or prepare_study(cfg, scenes, device)
    set_seed(cfg["seed"])

    # SAGEConv accepts no edge weights and drops them without warning, so edge_mode='weighted'
    # paired with backbone='sage' silently degenerates to 'dedup' -- measured: byte-identical
    # AUC. Promote to the weighted-mean conv, which is numerically identical to SAGEConv when
    # the weights are uniform, and record what actually ran.
    backbone = cfg["backbone"]
    if sd.edge_weight is not None and backbone == "sage":
        backbone = "wsage"

    model = GraphEncoder(
        sd.x.size(1),
        cfg["hidden_dim"],
        normalize=cfg["normalize"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        backbone=backbone,
        heads=cfg["heads"],
        aggr=cfg.get("aggr", "mean"),
    ).to(device)
    history: list[dict] = []
    t0 = time.time()

    def run_phase(module, n_epochs, lr, step, embed, phase: str) -> dict:
        """Train `module` for at most `n_epochs`, selecting on validation AUC.

        Shared by both stages so that early stopping, best-weight restore and the recorded
        history are literally the same code path -- a stage ablation where the two arms stop
        by different rules measures the stopping rule as much as the objective.
        """
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        best = {"val_auc": -1.0, "epoch": -1, "state": None}
        stale = 0
        for epoch in range(1, n_epochs + 1):
            module.train()
            opt.zero_grad()
            loss = step()
            loss.backward()
            opt.step()

            if epoch % cfg["eval_every"] == 0 or epoch == n_epochs:
                module.eval()
                with torch.no_grad():
                    ze = embed()
                va = ev.link_pred_metrics(ze, sd.split.val_pairs, sd.neg_val)["auc"]
                history.append(
                    {"phase": phase, "epoch": epoch, "loss": float(loss.item()), "val_auc": va}
                )
                if verbose:
                    print(f"  [{phase}] epoch {epoch:5d}  loss {loss.item():.4f}  val_auc {va:.4f}")
                if va > best["val_auc"]:
                    best = {"val_auc": va, "epoch": epoch, "state": copy.deepcopy(module.state_dict())}
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg["patience"]:
                        break
        if best["state"] is not None:
            module.load_state_dict(best["state"])  # report the best model, not the last one
        module.eval()
        return best

    stage = cfg["stage"]
    if stage not in ("s1", "s2", "s1->s2"):
        raise ValueError(f"unknown stage: {stage!r} (expected s1, s2 or s1->s2)")

    n_neg = int(sd.train_pos.size(1) * cfg["neg_ratio"])

    def contrastive_step():
        z = model(sd.x, sd.edge_index, sd.edge_weight)
        # z is detached before mining: the mined *indices* should not carry a gradient path of
        # their own (only which pairs are chosen depends on z; the loss on those pairs still
        # backprops through the z used in _contrastive below, which is the same forward pass).
        neg = _train_negatives(sd, cfg, n_neg, device, z=z.detach())
        return _contrastive(z, sd.train_pos, neg, cfg["temperature"])

    best = {"val_auc": -1.0, "epoch": -1}
    if stage in ("s1", "s1->s2"):
        best = run_phase(
            model, cfg["epochs"], cfg["lr"], contrastive_step,
            lambda: model(sd.x, sd.edge_index, sd.edge_weight), "s1",
        )

    encoder = model
    if stage in ("s2", "s1->s2"):
        mae = GraphMAE(
            sd.x.size(1),
            cfg["hidden_dim"],
            mask_rate=cfg["mask_rate"],
            alpha=cfg["mae_alpha"],
            replace_rate=cfg["replace_rate"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            backbone=backbone,
            heads=cfg["heads"],
            remask=cfg["remask"],
        ).to(device)
        if stage == "s1->s2":
            mae.encoder.load_state_dict(model.state_dict())
        best = run_phase(
            mae, cfg["mae_epochs"], cfg["mae_lr"],
            lambda: mae(sd.x, sd.edge_index, sd.edge_weight),
            lambda: mae.embed(sd.x, sd.edge_index, sd.edge_weight), "s2",
        )
        encoder = mae.encoder

    with torch.no_grad():
        emb = (
            mae.embed(sd.x, sd.edge_index, sd.edge_weight)
            if stage in ("s2", "s1->s2")
            else model(sd.x, sd.edge_index, sd.edge_weight)
        )

    metrics = ev.evaluate_embeddings(
        emb, sd.split, sd.neg_val, sd.neg_test, sd.neg_test_uniform, sd.scenes
    )
    # Train-set AUC is the leakage tripwire: it must sit clearly above test AUC.
    metrics["train_auc"] = ev.link_pred_metrics(
        emb,
        sd.split.train_pairs,
        ev.sample_negatives(
            sd.split.n_nodes,
            len(sd.split.train_pairs),
            sd.split.all_pairs_set(),
            "degree",
            sd.split.train_degree,
            seed=cfg["neg_seed"] + 4,
            alpha=cfg["neg_alpha"],
        ),
    )["auc"]
    metrics.update(
        {
            "backbone_effective": backbone,
            "stage": stage,
            "best_epoch": best["epoch"],
            "epochs_run": history[-1]["epoch"] if history else 0,
            "final_loss": history[-1]["loss"] if history else float("nan"),
            "wall_clock_s": time.time() - t0,
            # The encoder is what ships; the MAE decoder and mask token are discarded, so
            # counting them would make the stage-2 arms look larger than the model they yield.
            "n_params": sum(p.numel() for p in encoder.parameters()),
            "n_nodes": sd.split.n_nodes,
            "n_edges": len(sd.split.train_pairs) + len(sd.split.val_pairs) + len(sd.split.test_pairs),
            "scenes_sha256": sd.scenes_sha256[:16],
        }
    )
    return {
        "metrics": metrics,
        "history": history,
        "state_dict": copy.deepcopy(encoder.state_dict()),  # the encoder, never the MAE head
        "embeddings": emb.detach().cpu(),
    }


def infer(query: str, model_path: Path | str | None = None) -> tuple[str, list[float]]:
    path = Path(model_path or DEFAULT_CFG["graphmae_path"])
    if not path.exists():  # fall back to the stage-1 checkpoint before loading, not after
        path = Path(DEFAULT_CFG["model_path"])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    nodes, emb = ckpt["nodes"], ckpt["final_embeddings"]
    enc = SentenceTransformer(ckpt.get("cfg", DEFAULT_CFG)["text_encoder"])
    q = enc.encode([query], convert_to_tensor=True)
    n = enc.encode(nodes, convert_to_tensor=True)
    i = F.cosine_similarity(q, n).argmax().item()
    return nodes[i], emb[i].tolist()


def _dataset_root(dataset: str) -> Path:
    for root in (ROOT / "data" / dataset, ROOT / "data/home_v2" / dataset):
        if root.is_dir():
            return root
    raise FileNotFoundError(f"dataset not found: {dataset}")


def build_object_feat(
    dataset: str = "home_v2-2",
    core: str = "5",
    model_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> np.ndarray:
    path = model_path or DEFAULT_CFG["graphmae_path"]
    if not Path(path).exists():
        path = DEFAULT_CFG["model_path"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    nodes, emb = ckpt["nodes"], ckpt["final_embeddings"]
    lower = {n.lower(): i for i, n in enumerate(nodes)}
    enc = SentenceTransformer(ckpt.get("cfg", DEFAULT_CFG)["text_encoder"])
    nvec = enc.encode(nodes, convert_to_tensor=True)
    base = _dataset_root(dataset) / f"{core}-core"
    labels = [ln.strip() for ln in (base / "raw_graph.txt").read_text(encoding="utf-8").splitlines()]
    feats = []
    for lab in labels:
        if lab.lower() in lower:
            feats.append(emb[lower[lab.lower()]].numpy())
        else:
            q = enc.encode([lab], convert_to_tensor=True)
            feats.append(emb[F.cosine_similarity(q, nvec).argmax()].numpy())
    arr = np.stack(feats).astype(np.float32)
    out = Path(out_path or _dataset_root(dataset) / "object_feat.npy")
    np.save(out, arr)
    return arr
