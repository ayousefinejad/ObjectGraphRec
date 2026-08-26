"""Scene lists and co-occurrence graph construction."""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

from .config import DEFAULT_CFG


def scenes_from_benchmark(path: Path | str | None = None) -> list[list[str]]:
    with open(path or DEFAULT_CFG["benchmark_path"], encoding="utf-8") as f:
        data = json.load(f)
    return _entries_to_scenes(data)


def _norm_label(name: str) -> str:
    """'potted plant' / 'Potted Plant' -> 'Potted plant' (matches scenes_format.format_label).

    Applied to both input branches so that labels differing only in case collapse to one
    node. Without this the list branch produced 49 phantom nodes (1117 vs 1068), e.g.
    'Air Conditioner' / 'Air conditioner', and build_object_feat's .lower() keying made
    all but the last duplicate unreachable.
    """
    s = str(name).strip().lower()
    return (s[0].upper() + s[1:]) if s else s


def _normalize_scene(labels, min_objects: int = 2) -> list[str] | None:
    seen: set[str] = set()
    row: list[str] = []
    for raw in labels:
        lab = _norm_label(raw)
        if lab and lab not in seen:
            seen.add(lab)
            row.append(lab)
    return row if len(row) >= min_objects else None


def _entries_to_scenes(data: dict | list, min_objects: int = 2) -> list[list[str]]:
    """Parse data/scenes.json (list) or benchmark_results.json (dict)."""
    rows = data if isinstance(data, list) else [e.get("yolo", []) for e in data.values()]
    scenes = []
    for lst in rows:
        row = _normalize_scene(lst, min_objects)
        if row:
            scenes.append(row)
    return scenes


def load_scenes(cfg: dict | None = None) -> list[list[str]]:
    """Primary: data/scenes.json (NYU Depth format). Fallback: benchmark_results.json."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    sp = Path(cfg.get("scenes_path", ""))
    if sp.exists():
        with open(sp, encoding="utf-8") as f:
            return _entries_to_scenes(json.load(f))
    bp = Path(cfg["benchmark_path"])
    if bp.exists():
        with open(bp, encoding="utf-8") as f:
            return _entries_to_scenes(json.load(f))
    raise FileNotFoundError("no scenes found; run data/prepare-objectgraph/build_scenes.py")


def build_cooccurrence(scenes: list[list[str]]) -> tuple[list[str], dict[tuple[int, int], int], list[int]]:
    """Global co-occurrence counts.

    Returns (nodes, pair_counts, node_counts) where pair_counts maps (i, j) with i < j to
    c_ab (number of scenes containing both) and node_counts[i] is c_a (number of scenes
    containing object i). These are the quantities in Eq. (1)-(2) of the paper.
    """
    nodes = sorted({x for lst in scenes for x in lst})
    n2i = {x: i for i, x in enumerate(nodes)}
    pair_counts: dict[tuple[int, int], int] = {}
    node_counts = [0] * len(nodes)
    for lst in scenes:
        uniq = set(lst)
        for a in uniq:
            node_counts[n2i[a]] += 1
        for a, b in combinations(uniq, 2):
            i, j = n2i[a], n2i[b]
            key = (i, j) if i < j else (j, i)
            pair_counts[key] = pair_counts.get(key, 0) + 1
    return nodes, pair_counts, node_counts


def edge_index_from_counts(
    pair_counts: dict[tuple[int, int], int],
    node_counts: list[int] | None = None,
    edge_mode: str = "multiplicity",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build a symmetric edge_index under one of three weighting conventions.

    dedup        - each pair contributes one edge per direction.
    multiplicity - a pair seen in c_ab scenes contributes c_ab edges per direction. This is
                   what build_graph has always produced (edges were never deduplicated across
                   scenes), so under SAGE mean aggregation it already weights by raw c_ab.
    weighted     - one edge per direction carrying w_ab = c_ab / sqrt(c_a * c_b)  [Eq. 2].
    """
    src: list[int] = []
    dst: list[int] = []
    weight: list[float] = []
    for (i, j), c in pair_counts.items():
        reps = c if edge_mode == "multiplicity" else 1
        for _ in range(reps):
            src += [i, j]
            dst += [j, i]
        if edge_mode == "weighted":
            if node_counts is None:
                raise ValueError("edge_mode='weighted' requires node_counts")
            w = c / math.sqrt(node_counts[i] * node_counts[j])
            weight += [w, w]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float) if edge_mode == "weighted" else None
    return edge_index, edge_weight


def build_graph(scenes: list[list[str]], *, min_cooc: int = 1, edge_mode: str = "multiplicity"):
    """Co-occurrence graph. Defaults reproduce the original behaviour exactly.

    min_cooc drops pairs seen in fewer than min_cooc scenes (Eq. 2's threshold).
    """
    nodes, pair_counts, node_counts = build_cooccurrence(scenes)
    if min_cooc > 1:
        pair_counts = {k: c for k, c in pair_counts.items() if c >= min_cooc}
    edge_index, _ = edge_index_from_counts(pair_counts, node_counts, edge_mode)
    return nodes, edge_index.contiguous()


def load_graph_data(
    scenes: list[list[str]] | None = None,
    cfg: dict | None = None,
    device: torch.device | None = None,
) -> tuple[list[str], Data]:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = scenes or load_scenes(cfg)
    nodes, edge_index = build_graph(scenes)
    enc = SentenceTransformer(cfg["text_encoder"], device=str(device))
    with torch.no_grad():
        x = enc.encode(nodes, convert_to_tensor=True, device=device)
    # sentence-transformers 5.x encodes under torch.inference_mode(), and inference tensors
    # cannot be saved for backward. Clone to a normal tensor before it reaches autograd.
    x = x.clone().detach()
    data = Data(x=x, edge_index=edge_index.to(device)).to(device)
    return nodes, data
