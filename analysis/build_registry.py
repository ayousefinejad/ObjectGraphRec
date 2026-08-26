#!/usr/bin/env python3
"""Emit og_evaluations.csv -- the registry of object-graph-modality evaluations.

Every `n_runs` / `seeds` figure is COUNTED from og_runs.csv rather than typed in,
so the registry cannot drift from the data it describes.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "og_runs.csv").open()))


def sel(**kw):
    """Filter og_runs rows by exact column matches; list value = 'in'."""
    out = rows
    for k, v in kw.items():
        vals = v if isinstance(v, (list, tuple, set)) else [v]
        out = [r for r in out if r.get(k) in {str(x) for x in vals}]
    return out


def stat(sub):
    return len(sub), ",".join(sorted({r["seed"] for r in sub}, key=int))


# id, name, question it answers, row filter, metric columns, x axis, series/group, plot
EVALS = [
    ("E1", "Primary link prediction",
     "How well does the encoder recover held-out object co-occurrence edges?",
     sel(), "test_auc, test_ap (val_auc, val_ap for selection)",
     "configuration / arm", "sweep", "bar or dot + seed spread"),

    ("E2", "Negative-sampling regime",
     "Do degree-matched negatives have resolving power that uniform ones lack?",
     sel(), "test_auc vs test_auc_uniform (+ og_baselines.csv)",
     "method / arm", "negative regime", "paired bar, two panels"),

    ("E3", "Non-neural & untrained baselines",
     "Does the trained encoder beat structure-only and text-only references?",
     None, "og_baselines.csv: auc, ap, auc_uniform (+auc_std for untrained_sage)",
     "method", "-", "bar with encoder line overlaid"),

    ("E4", "Edge weighting (Eq. 2)",
     "Which edge-weight scheme is right: binary, raw count, or c/sqrt(c_a c_b)?",
     sel(sweep="stageA", min_cooc=1), "test_auc",
     "edge_mode (dedup|multiplicity|weighted)", "-", "bar + seed dots"),

    ("E5", "Co-occurrence threshold delta",
     "Should rare co-occurrence edges be thresholded away?",
     sel(sweep="stageA"), "test_auc + og_dataset.json threshold[] coverage",
     "min_cooc (1,2,3,5,10)", "edge_mode", "dual-axis-free: 2 stacked panels"),

    ("E6", "Optimizer sensitivity (OFAT)",
     "Which training hyperparameters actually move AUC?",
     sel(sweep="stageB"), "test_auc",
     "value of the swept knob", "knob (epochs|lr|temperature|dropout|neg_ratio|neg_mode)",
     "small multiples, one panel per knob"),

    ("E7", "lr x temperature interaction",
     "Does tuning one axis at a time mislead?",
     sel(sweep="stageC"), "test_auc",
     "lr (5 levels)", "temperature (5 levels)", "heatmap 5x5, seed-averaged"),

    ("E8", "Backbone ablation",
     "Is GAT really 'less stable' than SAGE, as the paper claims?",
     sel(sweep="stageD", hidden_dim=64, num_layers=2), "test_auc, test_ap + seed std",
     "backbone (gat|sage|wsage|gcn)", "-", "bar + per-seed dots; std as second panel"),

    ("E9", "Capacity: width & depth",
     "Do hidden_dim and num_layers matter on this graph?",
     sel(sweep="stageD"), "test_auc",
     "hidden_dim (16-256) / num_layers (1-4)", "-", "two line panels, sage only"),

    ("E10", "GraphMAE stage ablation",
     "Does the masked-autoencoder second stage do anything?",
     sel(sweep="stageE"), "test_auc, auc_degq1, degree_bias_rho, effective_rank",
     "recipe (stage x remask)", "mask_rate / mae_epochs / mae_alpha",
     "grouped bar; grid heatmap for the repaired arm"),

    ("E11", "Final configurations",
     "How much does cumulative tuning buy, and at what cost?",
     sel(sweep="stageF"), "test_auc, test_ap, auc_degq1, best_epoch",
     "config (7 named tags)", "-", "paired bar: overall AUC vs low-degree Q1"),

    ("E12", "Long-tail / degree quartiles",
     "Does tuning trade rare-object quality for hub-object quality?",
     sel(), "auc_degq1..auc_degq4",
     "degree quartile Q1-Q4", "configuration", "grouped line or slope chart"),

    ("E13", "Co-occurrence strata",
     "Is performance uniform across edge frequency?",
     sel(), "auc_c=1, auc_c=2-4, auc_c>=5 (+ n_c=* as weights)",
     "stratum", "configuration", "grouped bar; annotate n per stratum"),

    ("E14", "Embedding geometry",
     "Is the representation collapsing, and how is it distributed?",
     sel(), "effective_rank, alignment, uniformity, degree_bias_rho",
     "configuration", "metric", "small multiples (4 panels, separate y-scales)"),

    ("E15", "kNN room purity (diagnostic only)",
     "Do neighbours in embedding space share a room type?",
     sel(), "knn_room_purity, n_labelled",
     "configuration", "-", "bar -- label as diagnostic, NOT validation"),

    ("E16", "Training dynamics / convergence",
     "Is the shipped 20-epoch budget converged?",
     sel(), "og_curves.csv (epoch, val_auc, loss) + best_epoch, epochs_run",
     "epoch", "run / arm", "line chart, log-x optional"),

    ("E17", "Leakage tripwire",
     "Did any run leak val/test edges into message passing?",
     sel(), "train_auc vs test_auc",
     "test_auc", "sweep", "scatter with y=x reference"),

    ("E18", "Compute cost",
     "Is any backbone meaningfully more expensive at this graph size?",
     sel(sweep="stageD"), "wall_clock_s, n_params, epochs_run",
     "backbone", "-", "bar; derive ms/epoch = wall_clock_s/epochs_run*1000"),
]

with (HERE / "og_evaluations.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "evaluation", "question", "n_runs", "seeds",
                "required_fields", "x_axis", "group_by", "suggested_plot"])
    for eid, name, q, sub, fields, x, grp, plot in EVALS:
        n, seeds = ("-", "-") if sub is None else stat(sub)
        w.writerow([eid, name, q, n, seeds, fields, x, grp, plot])

print(f"wrote og_evaluations.csv  ({len(EVALS)} evaluations)")
for eid, name, _q, sub, *_ in EVALS:
    n, seeds = ("-", "-") if sub is None else stat(sub)
    print(f"  {eid:4s} {name:34s} n={n:>4}  seeds={seeds}")
