from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CFG = {
    "hidden_dim": 64,
    "epochs": 20,
    "lr": 1e-3,
    "temperature": 0.5,
    "text_encoder": "all-MiniLM-L6-v2",
    "model_path": ROOT / "data/graph-embeddings/v2/graphsage_model_v2.pt",
    "graphmae_path": ROOT / "data/graph-embeddings/v2/graphmae_model_v2.pt",
    "benchmark_path": ROOT / "data/benchmark_results.json",
    "scenes_path": ROOT / "data/scenes.json",
    # GraphMAE
    "mask_rate": 0.75,
    "mae_alpha": 3.0,
    "replace_rate": 0.1,
    "mae_epochs": 100,
    "mae_lr": 1e-3,
    "init_from": None,
    # --- Sweep / evaluation keys. Every default reproduces the shipped behaviour, so
    # DEFAULT_CFG still builds today's model exactly. ---
    # Reproducibility. split_seed and neg_seed stay fixed for the whole study so every
    # configuration is scored on identical held-out edges; only `seed` varies across repeats.
    "seed": 0,
    "split_seed": 42,
    "neg_seed": 7,
    # Evaluation protocol
    "val_frac": 0.1,
    "test_frac": 0.1,
    # neg_mode selects TRAINING negatives and is a tunable axis; 'uniform' is what
    # torch_geometric.negative_sampling does today, so it stays the default. Evaluation
    # negatives are fixed by protocol -- degree-matched is the headline metric and uniform is
    # always reported alongside it -- and are governed by neg_seed, not by this key.
    "neg_mode": "uniform",  # 'uniform' (current) | 'degree' | 'hard' | 'semihard'
    "neg_alpha": 1.0,
    # Percentile band of the per-anchor similarity ranking that 'semihard' draws from (0 =
    # hardest non-edge, 1 = least similar). Excludes the very top of the ranking on purpose:
    # in a co-occurrence graph the single most-similar non-edge to an object is often a near
    # miss of the split (an edge that would exist with slightly more data), not a true
    # negative, so drawing exclusively from there risks training against mislabelled positives.
    "semihard_band": (0.05, 0.30),
    "neg_ratio": 1.0,
    "eval_every": 5,
    "patience": 50,  # in evaluations, not epochs
    # Training recipe for the study path: 's1' contrastive only, 's2' GraphMAE only,
    # 's1->s2' contrastive then GraphMAE. Default 's1' keeps train_eval() reproducing every
    # stage A-D/F run; train_full() is 's1->s2' by construction and is unaffected.
    "stage": "s1",
    # GraphMAE re-masked before a *linear* decoder, which severs the gradient to the encoder
    # entirely (only decoder.bias trains). False is the repaired behaviour; set True only to
    # reproduce what the shipped code did. See docs/audit-findings.md F14.
    "remask": False,
    # Graph construction
    "min_cooc": 1,
    "edge_mode": "multiplicity",  # 'dedup' | 'multiplicity' (current) | 'weighted' (Eq. 2)
    # Architecture
    "num_layers": 2,
    "dropout": 0.0,
    "backbone": "sage",  # 'sage' | 'wsage' | 'gat' | 'gcn'
    # Neighbourhood aggregator for the 'sage' backbone -- GraphSAGE's own ablation axis
    # (Hamilton et al. Sec 3.3). 'mean' is SAGEConv's default and the published behaviour;
    # 'max' is the paper's pool aggregator, 'add' the unnormalised sum. Ignored by the other
    # backbones, which fix their own aggregation: wsage = weighted mean, gcn = symmetric
    # normalised sum, gat = attention-weighted sum.
    "aggr": "mean",  # 'mean' (current) | 'max' | 'add'
    "heads": 4,
    "normalize": True,
    "log_dir": ROOT / "data/graph-embeddings/sweeps",
}
