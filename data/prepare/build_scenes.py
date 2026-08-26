#!/usr/bin/env python3
"""Build data/scenes.json — Ultralytics YOLO path commented out (use existing scenes.json)."""
import argparse
import json
import sys
from pathlib import Path

# from tqdm import tqdm

from mit_home import MIT_HOME_CATEGORIES, category_from_path, is_home_category
from paths import BENCHMARK_JSON, MIT_DIR, NYU_DIR, SCENES_JSON
from scenes_format import benchmark_to_scenes, merge_scenes

# --- Ultralytics ---
# from paths import YOLO_HUB, YOLO_MODEL
# from yolo_model import load_yolo


def _image_paths(mit: bool, nyu: bool, mit_scope: str) -> list[Path]:
    paths = []
    if mit and MIT_DIR.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.JPG"):
            for p in MIT_DIR.rglob(ext):
                cat = category_from_path(p.resolve(), MIT_DIR)
                if mit_scope == "all" or is_home_category(cat):
                    paths.append(p.resolve())
    if nyu and NYU_DIR.is_dir():
        paths.extend(p.resolve() for p in NYU_DIR.glob("*.png"))
    return sorted(set(paths))


# def _detect_yolo(paths: list[Path], weights: Path | str | None, conf: float) -> dict:
#     from tqdm import tqdm
#
#     model = load_yolo(weights)
#     out = {}
#     for img in tqdm(paths, desc="YOLO26x"):
#         key = str(img)
#         labels = []
#         try:
#             for r in model(str(img), conf=conf, verbose=False):
#                 if r.boxes is not None:
#                     for box in r.boxes:
#                         labels.append(model.names[int(box.cls[0])])
#         except Exception as e:
#             print(f"skip {img.name}: {e}", file=sys.stderr)
#         out[key] = {"yolo": labels}
#     return out


def _write_scenes(path: Path, scenes: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _build_from_benchmark(benchmark_path: Path, min_objects: int) -> list[list[str]]:
    with open(benchmark_path, encoding="utf-8") as f:
        return benchmark_to_scenes(json.load(f), min_objects)


def main():
    p = argparse.ArgumentParser(description="Build data/scenes.json (YOLO path disabled)")
    p.add_argument("--mit-scope", choices=("home", "all"), default="home")
    p.add_argument("--min-objects", type=int, default=2)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--scenes-out", type=Path, default=SCENES_JSON)
    p.add_argument("--from-benchmark", type=Path, default=BENCHMARK_JSON,
                   help="build scenes.json from existing benchmark JSON (no YOLO)")
    # --- Ultralytics CLI (commented) ---
    # p.add_argument("--mit", action="store_true", default=True)
    # p.add_argument("--no-mit", action="store_false", dest="mit")
    # p.add_argument("--nyu", action="store_true", default=True)
    # p.add_argument("--no-nyu", action="store_false", dest="nyu")
    # p.add_argument("--limit", type=int, default=0)
    # p.add_argument("--conf", type=float, default=0.5)
    # p.add_argument("--model", default=None)
    # p.add_argument("--write-benchmark", action="store_true")
    # p.add_argument("--benchmark-out", type=Path, default=BENCHMARK_JSON)
    args = p.parse_args()

    if not args.from_benchmark.exists():
        n = len(_image_paths(True, True, args.mit_scope))
        raise SystemExit(
            f"Ultralytics YOLO is commented out.\n"
            f"  - Use existing {SCENES_JSON} (e.g. NYU Depth), or\n"
            f"  - Provide {args.from_benchmark}, or\n"
            f"  - Uncomment YOLO blocks in build_scenes.py / yolo_model.py / paths.py\n"
            f"  ({n} images ready under mit-indoors / NYU-Depth)"
        )

    scenes = _build_from_benchmark(args.from_benchmark, args.min_objects)
    if args.merge and args.scenes_out.exists():
        with open(args.scenes_out, encoding="utf-8") as f:
            scenes = merge_scenes(json.load(f), scenes)

    _write_scenes(args.scenes_out, scenes)
    print(f"scenes: {args.scenes_out} ({len(scenes)} scenes, from {args.from_benchmark})")


if __name__ == "__main__":
    main()
