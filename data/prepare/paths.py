from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"
MIT_TAR = CACHE / "indoorCVPR_09.tar"
MIT_DIR = REPO / "data/mit-indoors"
NYU_DIR = REPO / "data/NYU-Depth"

# --- Ultralytics YOLO (commented out) ---
# YOLO_HUB = "yolo26x"
# YOLO_MODEL = REPO / "assets/yolo26x.pt"

BENCHMARK_JSON = REPO / "data/benchmark_results.json"
SCENES_JSON = REPO / "data/scenes.json"

MIT_URL = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
