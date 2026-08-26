#!/usr/bin/env python
"""Figures for §14: the MICRO object-modality port and the --graph_mode diagnostic.

Reads the raw MICRO training logs directly -- MICRO writes no results CSV, so the logs are the
only record. Arms whose logs are absent or still running are dropped from the aggregate panel and
reported on stderr rather than silently averaged over fewer seeds.

    python scripts/figures_micro.py --logs docs/micro-logs --out docs/notion/images
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

# Same palette convention as figures_objectgraph.py: one colour per arm, learned once.
C_BASE, C_BOTH, C_FUSION, C_CONTRAST = "#7f8c8d", "#c0392b", "#27ae60", "#8e44ad"

# 'recall=[R@10, R@20]' on the eval epochs; index 1 is the Recall@20 that selection uses.
EPOCH_RE = re.compile(r"Epoch (\d+) \[.*?\]: train==\[.*?\], recall=\[([\d.]+), ([\d.]+)\]")
TEST_RE = re.compile(r"Test_Recall@20: ([\d.]+)")
# The queue prints one of these per launched run; a log without it never reached the trainer.
ARGS_RE = re.compile(r"Namespace\(.*?seed=(\d+).*?modalities='([^']*)'(?:.*?graph_mode='([^']*)')?")

ARMS = [
    ("image+text (published)", "micro_seed{s}.log", C_BASE),
    ("+object, both pathways", "micro_obj_seed{s}.log", C_BOTH),
    ("object in fusion only", "micro_obj_fusion_seed{s}.log", C_FUSION),
    ("object in contrastive only", "micro_obj_contrast_seed{s}.log", C_CONTRAST),
]
SEEDS = (0, 1, 2)


def parse(path: Path) -> dict | None:
    """Return the val curve and the selected test score, or None if the run never finished.

    'Finished' means the log ends with an early stop or the epoch cap, not merely that the file
    exists -- a log being appended to right now would otherwise contribute a truncated best-so-far
    that looks like a converged result.
    """
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    epochs = [(int(e), float(r20)) for e, _r10, r20 in EPOCH_RE.findall(text)]
    tests = [float(t) for t in TEST_RE.findall(text)]
    if not epochs or not tests:
        return None
    done = "#####Early stop! #####" in text or "'recall'" in text.rsplit("\n", 3)[-1]
    return {"epochs": [e for e, _ in epochs], "val": [v for _, v in epochs],
            "test": max(tests), "n_evals": len(epochs), "done": done}


def load(logs: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for label, pat, _c in ARMS:
        out[label] = {}
        for s in SEEDS:
            r = parse(logs / pat.format(s=s))
            if r is None:
                print(f"  [missing] {label} seed {s}", file=sys.stderr)
            elif not r["done"]:
                print(f"  [running] {label} seed {s} -- excluded (best so far {r['test']:.5f})",
                      file=sys.stderr)
            else:
                out[label][s] = r
    return out


def fig_micro(data: dict, out: Path, name: str = "fig14_micro_graph_mode") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (a) seed 0 only. Averaging val curves across seeds would blur the thing the panel exists to
    # show -- that the gap opens at the very first eval and is a level shift, not slow convergence.
    for label, _pat, colour in ARMS:
        r = data[label].get(0)
        if r is None:
            continue
        ax1.plot(r["epochs"], r["val"], color=colour, lw=1.8, label=label)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Recall@20")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.set_title("(a) Validation curve, seed 0. The two arms carrying the InfoNCE\n"
                  "term open ~0.010 lower at the first eval and never close it.", fontsize=10)

    # (b) per-seed points over the arm mean, so n is visible rather than asserted.
    labels, means, colours = [], [], []
    for i, (label, _pat, colour) in enumerate(ARMS):
        seeds = data[label]
        if not seeds:
            continue
        vals = [seeds[s]["test"] for s in sorted(seeds)]
        x = len(labels)
        labels.append(f"{label}\n(n={len(vals)})")
        means.append(float(np.mean(vals)))
        colours.append(colour)
        ax2.scatter([x] * len(vals), vals, color=colour, s=34, zorder=3, alpha=0.85)
        ax2.hlines(np.mean(vals), x - 0.28, x + 0.28, color=colour, lw=2.4, zorder=4)
        ax2.text(x, max(vals) + 0.0008, f"{np.mean(vals):.5f}", ha="center", fontsize=8.5)

    base = means[0] if means else 0.0
    ax2.axhline(base, color=C_BASE, ls="--", lw=1, alpha=0.6, zorder=1)
    # Headroom for the value labels, which otherwise run into the title.
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo, hi + 0.15 * (hi - lo))
    ax2.set_xticks(range(len(labels)), labels, fontsize=8)
    ax2.set_ylabel("Test Recall@20 (at best val)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("(b) Selected test score, one point per seed, bar at the mean.\n"
                  "Dashed line is the published two-modality baseline.", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg -> {out}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", type=Path, default=root / "docs" / "micro-logs")
    p.add_argument("--out", type=Path, default=root / "docs" / "notion" / "images")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load(args.logs)
    for label, _pat, _c in ARMS:
        seeds = data[label]
        if seeds:
            vals = [seeds[s]["test"] for s in sorted(seeds)]
            spread = f" ± {np.std(vals):.5f}" if len(vals) > 1 else ""
            print(f"{label:32s} n={len(vals)}  mean {np.mean(vals):.5f}{spread}  "
                  f"[{', '.join(f'{v:.5f}' for v in vals)}]")
        else:
            print(f"{label:32s} no completed runs")
    fig_micro(data, args.out)


if __name__ == "__main__":
    main()
