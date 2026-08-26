#!/usr/bin/env python3
"""Detect objects in the MIT Indoor dwelling-room images with YOLOv8x and DETR.

Mirrors detect_mit_openai.py: same six dwelling-room categories, same
`format_label` normaliser, same scenes.json output shape (a list of object-name
lists, one per image). That is what makes the three detectors comparable --
only the detector changes.

    python detect_mit_yolo_detr.py --detector yolo --out ../yolo_mit.json
    python detect_mit_yolo_detr.py --detector detr --out ../detr_mit.json

Writes a new file each time. Never touches data/scenes.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import MIT_DIR

_spec = importlib.util.spec_from_file_location(
    "scenes_format", _SCRIPT_DIR / "scenes_format.py")
_sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sf)
format_label = _sf.format_label

# identical to detect_mit_openai.MIT_TARGET_CATEGORIES
CATEGORIES = ("bathroom", "bedroom", "children_room", "dining_room",
              "kitchen", "livingroom")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def images() -> list[Path]:
    out = []
    for c in CATEGORIES:
        d = MIT_DIR / "Images" / c
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.suffix.lower() in SUFFIXES)
    return out


def run_yolo(paths, conf, batch, device):
    from ultralytics import YOLO
    model = YOLO("yolov8x.pt")
    per_image = []
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        for res in model.predict(chunk, conf=conf, device=device, verbose=False):
            names = res.names
            per_image.append([names[int(c)] for c in res.boxes.cls.tolist()])
        print(f"  yolo {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return per_image


def run_detr(paths, conf, batch, device):
    from transformers import DetrForObjectDetection, DetrImageProcessor
    name = "facebook/detr-resnet-101"
    proc = DetrImageProcessor.from_pretrained(name)
    model = DetrForObjectDetection.from_pretrained(name).to(device).eval()
    per_image = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ims, sizes = [], []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            ims.append(im)
            sizes.append((im.height, im.width))
        inp = proc(images=ims, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        done = proc.post_process_object_detection(
            out, target_sizes=torch.tensor(sizes).to(device), threshold=conf)
        for d in done:
            per_image.append([model.config.id2label[int(l)] for l in d["labels"]])
        print(f"  detr {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return per_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["yolo", "detr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold; 0.25 is the ultralytics default")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-objects", type=int, default=2,
                    help="scenes with fewer distinct labels are dropped, as in scenes_format")
    a = ap.parse_args()

    paths = images()
    print(f"{len(paths)} MIT dwelling-room images across {len(CATEGORIES)} categories")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = (run_yolo if a.detector == "yolo" else run_detr)(paths, a.conf, a.batch, device)
    assert len(raw) == len(paths), f"{len(raw)} results for {len(paths)} images"

    # Same normalisation + min_objects filter the OpenAI corpus went through, so a
    # difference between corpora is the detector and not the post-processing.
    scenes, kept_paths = [], []
    for p, labels in zip(paths, raw):
        seen, scene = set(), []
        for lab in labels:
            f = format_label(lab)
            if f and f not in seen:
                seen.add(f)
                scene.append(f)
        if len(scene) >= a.min_objects:
            scenes.append(scene)
            kept_paths.append(str(p.relative_to(MIT_DIR)))

    out = Path(a.out)
    out.write_text(json.dumps(scenes))
    # a manifest so a scene can be traced back to its image -- openai_mit.json
    # has no paths, which is why the three corpora can only be compared in
    # aggregate rather than image-by-image
    Path(str(out).replace(".json", "_manifest.json")).write_text(json.dumps({
        "detector": a.detector, "conf": a.conf, "min_objects": a.min_objects,
        "n_images": len(paths), "n_scenes": len(scenes), "images": kept_paths}))

    labels = {l for s in scenes for l in s}
    total = sum(len(s) for s in scenes)
    print(f"\n{a.detector}: {len(scenes)} scenes kept of {len(paths)} images "
          f"({len(paths) - len(scenes)} dropped by min_objects={a.min_objects})")
    print(f"  distinct labels: {len(labels)}")
    print(f"  objects/scene:   {total / max(len(scenes), 1):.2f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
