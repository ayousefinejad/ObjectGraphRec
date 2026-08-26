# Patches for the third-party baselines

ObjectGraphRec evaluates the object-graph modality inside three host recommenders. LATTICE lives
in `src/` because this repository builds directly on it. **CRANE and MICRO are third-party
projects and are not redistributed here** — this directory carries only the files we changed or
added, so you clone each upstream yourself and copy these over it.

| Baseline | Upstream | Pinned commit | License |
|---|---|---|---|
| CRANE | https://github.com/MKC-Lab/CRANE | `cdd6da3` | no LICENSE file upstream — not redistributed |
| MICRO | https://github.com/CRIPAC-DIG/MICRO | `80f6cc6` | MIT (see `micro/LICENSE.upstream`) |

## CRANE

```bash
git clone https://github.com/MKC-Lab/CRANE && cd CRANE && git checkout cdd6da3
cp -r /path/to/ObjectGraphRec/patches/crane/src/. CRANE/src/
```

What the patch contains:

- `models/crane.py` — adds the `ObjectGraph` module: a kNN graph over the object features and a
  third stream into cross-modal attention. Requires `cross_modal_batch_first: True` **and**
  `object_in_attention: True`; with `cross_modal_batch_first: False` (the upstream default) the
  object features never meet the image or text features and the modality is inert by construction.
- `models/mmgcn.py` — MMGCN (Wei et al., MM'19) ported from
  [enoche/MMRec](https://github.com/enoche/MMRec) into CRANE's plug-in framework, plus one added
  `o_gcn` branch mirroring the existing text branch. Everything outside that branch is the
  unmodified upstream port.
- `configs/model/{CRANE,MMGCN}.yaml`, `configs/dataset/home_v2*.yaml` — object-graph switches and
  the Amazon Home & Kitchen dataset definitions.
- `run_*_objgraph.py`, `queue_*.sh` — seed-replicated ± object runners.

Run from `CRANE/src/` — `Config` resolves `configs/overall.yaml` relative to the working
directory, not to the script, so any other cwd fails with `KeyError: 'valid_metric'`.

## MICRO

```bash
git clone https://github.com/CRIPAC-DIG/MICRO && cd MICRO && git checkout 80f6cc6
cp -r /path/to/ObjectGraphRec/patches/micro/codes/. codes/
```

Adds the object modality alongside image and text, and a `--graph_mode {both,fusion,contrast}`
switch that separates MICRO's two pathways. That switch is what shows the object channel is
harmless in fusion but costly through the per-modality contrastive term — see the MICRO row of
the results table in the top-level README.
