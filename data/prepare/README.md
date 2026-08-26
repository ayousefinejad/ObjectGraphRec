# Prepare data for ObjectGraph

Download **MIT Indoor** (home room categories under `data/mit-indoors/`).

**Ultralytics / YOLO26x is commented out** in this folder. Use your existing **`data/scenes.json`** (NYU Depth format) for graph training.

## Active: download only

```bash
./data/prepare-objectgraph/run_prepare.sh
# or
python data/prepare-objectgraph/download_mit_indoor.py
```

If you see `tarfile.ReadError: unexpected end of data`, the `.tar` is **incomplete/corrupt** (common after interrupted download). Fix:

```bash
rm -f data/prepare-objectgraph/cache/indoorCVPR_09.tar
python data/prepare-objectgraph/download_mit_indoor.py --force-download
```

Extracted layout: `data/mit-indoors/Images/<category>/*.jpg`  
Home categories filter: see `mit_home.py` (`kitchen`, `bedroom`, `bathroom`, …).

## Commented out: Ultralytics YOLO

These files have YOLO blocks commented:

| File | Status |
|------|--------|
| `paths.py` | `YOLO_HUB`, `YOLO_MODEL` commented |
| `yolo_model.py` | `ultralytics` import commented |
| `build_scenes.py` | `_detect_yolo` commented; optional `--from-benchmark` only |
| `run.py` | `build_scenes.py` step commented |

To re-enable: uncomment those blocks and `pip install ultralytics>=8.4`.

```bash
# when YOLO is uncommented:
# python data/prepare-objectgraph/build_scenes.py --mit-scope home --merge --write-benchmark
```

## `scenes.json` format (NYU Depth)

```json
[
  ["Mirror", "Sink", "Toilet", "Light"],
  ["Vanity", "Sink", "Mirror", "Light", "Bathtub", "Towel"]
]
```

Without YOLO, convert an existing `benchmark_results.json` (if you have one):

```bash
python data/prepare-objectgraph/build_scenes.py --from-benchmark data/benchmark_results.json --merge
```

## Scripts

| Script | Purpose |
|--------|---------|
| `download_mit_indoor.py` | Fetch/extract MIT Indoor |
| `mit_home.py` | 14 home MIT category names |
| `build_scenes.py` | Scenes JSON (**YOLO disabled**; `--from-benchmark` only) |
| `scenes_format.py` | Label formatting helpers |
| `run_prepare.sh` | Download only |
| `run.py` | Download only (YOLO step commented) |

## Train ObjectGraph

```python
from ObjectGraph import train_full, build_object_feat

train_full()  # reads data/scenes.json
build_object_feat("home_v2-2")
```

See [ObjectGraph/docs/ARCHITECTURE.md](../../ObjectGraph/docs/ARCHITECTURE.md).
