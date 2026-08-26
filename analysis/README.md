# Analysis and figure generation

These scripts produced every table and figure in the paper. They read experiment outputs
(`downstream.csv`, `fusion_arms.csv`, `tuning.csv`, run logs, dumped embeddings) and write the
tables in `RESULTS.md` / `FINAL_TABLES.md` and the figures in `../figures/`.

## They expect the original working-tree layout

The scripts were written against the layout the experiments ran in, and resolve paths as:

```python
HERE = Path(__file__).resolve().parent      # this directory
OG   = HERE.parent / "object-graph"         # ← the recommender working tree
RUNS = OG / "data" / "lattice-runs"         # ← per-variant results and features
EMB  = OG / "objectgraph-eval" / "embeddings"
```

In this repository the code lives in `src/`, not `object-graph/`, and **the data directories are
not included** — `data/lattice-runs/` alone is ~27 GB. So these scripts do not run as-is against a
fresh clone. To use them, either:

1. **Recreate the layout** — place a working tree at `object-graph/` next to `analysis/`, with
   `data/lattice-runs/<variant>/` populated by `scripts/export_lattice_feats.py` and
   `scripts/run_lattice_study.py`; or
2. **Repoint the constant** — change `OG` at the top of the script you want to run. It is one
   line per script and is the only layout assumption.

The scripts are included because they are the exact code behind the reported numbers, and because
each one documents in its docstring what it measures and which artifact it reads. They are a
record of the analysis as much as a tool to re-run it.

## What's here

| Script | Produces |
|---|---|
| `final_tables.py` | `FINAL_TABLES.md` — CF backbone, fusion, negative sampling |
| `make_figures.py` | F01–F14, the object-graph sensitivity study |
| `make_overlap_data.py`, `plot_overlap.py` | F17 — object vocabulary vs catalogue coverage |
| `make_strata_data.py`, `plot_strata.py` | F18 — where the gain lands, plus the placebo |
| `plot_corpus.py` | F19 — MIT vs NYU corpus comparison |
| `make_tsne_data.py`, `plot_tsne_encoders.py` | F15 — GraphSAGE vs GAT embedding geometry |
| `inference_compare.py` | `INFERENCE_COMPARE.md` — worked side-by-side recommendations |
| `semantic_assignment.py`, `make_title_overlap.py` | label→node matching analyses |

`RESULTS.md` is the long-form write-up, including the coverage/strata/placebo analysis behind the
mechanism claim and the per-architecture comparisons. `EXPERIMENTS_2026-08-16.md` is a dated lab
log of one day's runs.
