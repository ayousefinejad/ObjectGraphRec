#!/usr/bin/env bash
# Download + extract MIT Indoor only (~2.4 GB)
# Ultralytics YOLO (build_scenes.py) — commented out; see README
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Download MIT Indoor ==="
python3 download_mit_indoor.py

echo "Done. Images under data/mit-indoors/"
# echo "Next (uncomment Ultralytics in build_scenes.py):"
# python3 build_scenes.py --mit-scope home --merge --write-benchmark
