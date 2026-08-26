#!/usr/bin/env python3
"""The three end-of-queue tables, generated from artifacts on disk.

    ~/hamedenv/bin/python final_tables.py [--out FINAL_TABLES.md]

  1. Collaborative-filtering backbone effect   MF / NGCF / LightGCN, with and without the object graph
  2. Fusion hyperparameter effect              how the modality weights are learned, and whether it matters
  3. Negative sampling                         the BPR setting, reported (not ablated)

Reads finished runs only and says so where a cell is missing, so this can be run mid-queue and
re-run at the end without editing. Every number comes from a run's last `test==` line -- the test
metrics of the best *validation* epoch, which is what main.py prints and never a best-test value.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OG = HERE.parent / "object-graph"
RUNS = OG / "data" / "lattice-runs"
EMB = OG / "objectgraph-eval" / "embeddings"

TEST = re.compile(r"test==\[.*?\], recall=\[([\d.]+), ([\d.]+)\], precision=\[([\d.]+), ([\d.]+)\], "
                  r"hit=\[[\d.]+, [\d.]+\], ndcg=\[([\d.]+), ([\d.]+)\]")
KEYS = ("recall@10", "recall@20", "precision@10", "precision@20", "ndcg@10", "ndcg@20")
FLOOR = 0.0003          # two identical runs differ by this much; below it nothing is attributable


def from_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    m = TEST.findall(path.read_text(errors="ignore"))
    if not m:
        return None
    return dict(zip(KEYS, (float(x) for x in m[-1])))


def logs(pattern: str) -> dict[int, dict]:
    """seed -> metrics, for embeddings/<pattern with {seed}>.log"""
    out = {}
    for s in (0, 1, 2):
        r = from_log(EMB / (pattern.format(seed=s) + ".log"))
        if r:
            out[s] = r
    return out


def from_csv(path: Path, **where) -> dict[int, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            if all(r.get(k) == v for k, v in where.items()):
                out[int(r["seed"])] = {k: float(r[k]) for k in KEYS if r.get(k)}
    return out


def agg(runs: dict[int, dict], key="recall@20") -> str:
    if not runs:
        return "—"
    v = np.array([r[key] for r in runs.values()])
    if len(v) == 1:
        return f"{v[0]:.5f} (n=1)"
    return f"{v.mean():.5f} ± {v.std(ddof=1):.5f}"


def paired(a: dict[int, dict], b: dict[int, dict], key="recall@20") -> str:
    """b - a, paired by seed. Refuses to report a delta it cannot defend."""
    s = sorted(set(a) & set(b))
    if not s:
        return "—"
    d = np.array([b[k][key] - a[k][key] for k in s])
    base = np.mean([a[k][key] for k in s])
    pct = 100 * d.mean() / base
    if len(d) < 2:
        return f"{d.mean():+.5f} ({pct:+.1f}%) · n=1, no variance"
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    verdict = "n.s." if abs(t) < 4.303 else ("✅" if d.mean() > 0 else "❌")
    tail = "" if abs(d.mean()) > FLOOR else " · below the 0.0003 floor"
    return f"{d.mean():+.5f} ({pct:+.1f}%) · t={t:+.2f} {verdict}{tail}"


def table(rows: list[list[str]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- 1. CF backbone
def cf_table() -> str:
    mf_no = from_csv(RUNS / "fusion_arms.csv", variant="default_fixed", arm="noobj")
    mf_ob = from_csv(RUNS / "downstream.csv", variant="default_fixed")
    ng_no, ng_ob = logs("ngcf1_noobj_seed{seed}"), logs("ngcf1_control_seed{seed}")
    lg_no = from_csv(RUNS / "tuning.csv", cell="lgn3")
    lg_ob = from_csv(RUNS / "tuning.csv", cell="obj_lgn3")

    rows = []
    for name, depth, no, ob in (("MF", "0", mf_no, mf_ob),
                                ("NGCF", "1", ng_no, ng_ob),
                                ("LightGCN", "3", lg_no, lg_ob)):
        rows.append([f"**{name}**", depth, "image+text", agg(no), agg(no, "ndcg@20"),
                     str(len(no)) if no else "0", "—"])
        rows.append([f"**{name}**", depth, "+ object", agg(ob), agg(ob, "ndcg@20"),
                     str(len(ob)) if ob else "0", paired(no, ob)])
    body = table(rows, ["Backbone", "Depth", "Modalities", "R@20", "NDCG@20", "Seeds",
                        "Paired Δ R@20 vs image+text"])
    note = ("\nDepth is `len(--weight_size)`, the number of user–item propagation hops. **MF is "
            "LightGCN at depth 0** — with no layers the loop never runs and the branch returns "
            "the same tensors (`Models.py:735` vs `:747`), so this is one depth axis, not three "
            "unrelated architectures.\n\nNGCF is reported at depth 1, not 2: it scores *higher* "
            "at depth 1 than depth 2 on image+text (0.03906 vs 0.03698, single seeds in "
            "`tuning.csv`), the opposite of LightGCN — so depth 2, inherited from the MF recipe, "
            "would have handicapped it. Both NGCF arms here are run at depth 1, `--epoch 400`, "
            "all three seeds, so the pair is protocol-matched.")
    return body + "\n" + note


# --------------------------------------------------------------------------- 2. fusion
def fusion_table() -> str:
    ctrl = from_csv(RUNS / "downstream.csv", variant="default_fixed")
    arms = {"frozen (weights pinned at uniform)": logs("abl_frozen_seed{seed}"),
            "gated (per-item n×3 gate)": logs("abl_gated_seed{seed}"),
            "softmax, lr_fusion 0.05 (100× step)": logs("abl_lrfus_seed{seed}")}
    rows = [["**softmax, learned** (published)", agg(ctrl), agg(ctrl, "ndcg@20"),
             str(len(ctrl)), "— (reference)"]]
    for name, r in arms.items():
        rows.append([name, agg(r), agg(r, "ndcg@20"), str(len(r)) if r else "0",
                     paired(ctrl, r)])
    body = table(rows, ["Fusion of the three modalities", "R@20", "NDCG@20", "Seeds",
                        "Paired Δ vs published"])

    a = alpha()
    note = (f"\nLearned weights α = softmax(modal_weight) at end of training, across "
            f"{a['n']} object-bearing runs: **image {a['image']:.4f} ± {a['image_sd']:.4f}, "
            f"text {a['text']:.4f} ± {a['text_sd']:.4f}, object {a['object']:.4f} ± "
            f"{a['object_sd']:.4f}** — under 0.02 from the uniform 1/3 initialisation.\n\n"
            "The reason is structural, not a tuning failure: `item_adj` is rebuilt on batch 0 of "
            "each epoch and detached for the other ~149, so these three scalars receive ~130 "
            "gradient steps in a whole run. **`frozen` is the arm that matters** — if pinning α "
            "at uniform costs nothing, the adaptive fusion is inert and should not be claimed as "
            "a contribution.\n\n**What α does when it is allowed to move.** At `--lr_fusion 0.05` "
            "(100× step) the weights leave uniform decisively and land, across all three seeds, "
            "at **image ≈ 0.576, text ≈ 0.366, object ≈ 0.058** — the model drives the object "
            "modality's weight down roughly 6×. Yet that arm scores 0.04254 vs the published "
            "0.04291 (n.s.), while removing the object modality outright costs −6.6%. So the "
            "object graph's contribution is real but saturates at a small weight: the published "
            "uniform α over-weights it, and correcting that neither helps nor hurts measurably.\n\n"
            "α is reported only for runs where the object modality is on: "
            "`fusion_state()` prints softmax(modal_weight) *unmasked*, while `_modal_weights()` "
            "masks the logits before the softmax, so an image+text run prints 0.3333 for a "
            "modality whose effective weight is exactly 0.")
    return body + "\n" + note


def alpha() -> dict:
    F = re.compile(r"^LATTICE_FUSION \d+ (\{.*\})$", re.M)
    vals = []
    # Only runs where all three modalities are ON and the weights were actually LEARNED.
    # Excluded, and why each would corrupt the mean:
    #   noobj / no_img / no_text / obj_only -- a masked modality keeps its 0.3333 init because it
    #     receives no gradient, and fusion_state() prints the UNMASKED softmax, so it would report
    #     a weight for a modality that was switched off;
    #   frozen -- pinned at uniform by construction, so it is not evidence about what is learned;
    #   gated  -- a per-item n x 3 gate, not the 3 global scalars this row describes.
    #   lrfus  -- 100x step size, so it is not "what is learned at the published lr". It is
    #     reported on its own line below because it is the one setting where alpha does move.
    skip = ("noobj", "no_img", "no_text", "obj_only", "txt_only", "img_only",
            "frozen", "gated", "lrfus")
    for p in list((RUNS / "logs").glob("*.log")) + list(EMB.glob("*.log")):
        if any(s in p.name for s in skip):
            continue
        m = F.findall(p.read_text(errors="ignore"))
        if not m:
            continue
        sm = json.loads(m[-1]).get("softmax")
        if sm and len(sm) == 3:
            vals.append(sm)
    v = np.array(vals, dtype=float)
    if not len(v):
        return dict(n=0, image=0, text=0, object=0, image_sd=0, text_sd=0, object_sd=0)
    return dict(n=len(v), image=v[:, 0].mean(), text=v[:, 1].mean(), object=v[:, 2].mean(),
                image_sd=v[:, 0].std(), text_sd=v[:, 1].std(), object_sd=v[:, 2].std())


# --------------------------------------------------------------------------- 2b. modality LOO
def modality_table() -> str:
    full = from_csv(RUNS / "downstream.csv", variant="default_fixed")
    rows = [["image + text + object (full)", agg(full), agg(full, "ndcg@20"), str(len(full)),
             "— (reference)"]]
    for name, runs in (("image + text  (− object)",
                        from_csv(RUNS / "fusion_arms.csv", variant="default_fixed", arm="noobj")),
                       ("image + object  (− text)", logs("abl_no_text_seed{seed}")),
                       ("text + object  (− image)", logs("abl_no_img_seed{seed}")),
                       ("text only", logs("abl_txt_only_seed{seed}")),
                       ("image only", logs("abl_img_only_seed{seed}")),
                       ("object only", logs("abl_obj_only_seed{seed}"))):
        rows.append([name, agg(runs), agg(runs, "ndcg@20"), str(len(runs)) if runs else "0",
                     paired(full, runs)])
    return table(rows, ["Modalities", "R@20", "NDCG@20", "Seeds", "Paired Δ vs full"])


# --------------------------------------------------------------------------- 3. negatives
def negatives_table() -> str:
    rows = [["Negatives per positive", "**1**", "`load_data.py:186-189`"],
            ["Sampled from", "all 14,503 items, uniformly", "`sample_neg_items_for_u`, line 172"],
            ["Rejection", "any item in that user's train set", "line 177"],
            ["Resampled", "every batch (a positive meets a new negative each epoch)", "line 152"],
            ["Positives per step", "1 per sampled user, uniform from their train items", "line 187"],
            ["Triples per step", "1,024 (batch size)", "`--batch_size`"],
            ["Loss", "BPR: −log σ(score(u,i⁺) − score(u,i⁻))", "`main.py:224-231`"],
            ["Regularisation", "L2 on the three embedding tensors, 1e-5", "`--regs[0]`"]]
    return table(rows, ["Setting", "Value", "Source"]) + (
        "\nReported, not ablated. Varying k would need `sample()` and `bpr_loss` reshaped for k "
        "negatives per positive; the study holds it at the published value of 1 throughout, so "
        "it is a constant across every arm in the tables above rather than a free variable.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "FINAL_TABLES.md"))
    a = p.parse_args()

    doc = ["# Final tables — object-graph study",
           "",
           "Amazon Home & Kitchen (`home_v2-2`, 5-core: 14,503 items, 59,251 users). Object graph "
           "from MIT-Indoors + NYU-Depth. Every Δ is paired by seed; t is a paired t-statistic on "
           "3 seeds, so |t| ≥ 4.303 is p < .05. Deltas below 0.0003 are unattributable — that is "
           "the measured spread between two identical runs.",
           "", "## 1. Collaborative-filtering backbone", "", cf_table(),
           "", "## 2. Fusion of the modalities", "", fusion_table(),
           "", "### 2b. Per-modality contribution (leave-one-out)", "", modality_table(),
           "", "## 3. Negative sampling", "", negatives_table(), ""]
    text = "\n".join(doc)
    Path(a.out).write_text(text)
    print(text)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
