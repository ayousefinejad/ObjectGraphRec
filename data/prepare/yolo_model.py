"""Ultralytics YOLO — disabled (commented out). Uncomment to use build_scenes.py."""

# import shutil
# from pathlib import Path
#
# from paths import YOLO_HUB, YOLO_MODEL
#
#
# def load_yolo(weights: Path | str | None = None):
#     from ultralytics import YOLO
#
#     p = Path(weights) if weights else YOLO_MODEL
#     if p.is_file():
#         return YOLO(str(p))
#     print(f"loading {YOLO_HUB} (downloads if needed)")
#     model = YOLO(YOLO_HUB)
#     if not YOLO_MODEL.exists():
#         src = Path(getattr(model, "ckpt_path", None) or f"{YOLO_HUB}.pt")
#         if src.is_file():
#             YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
#             shutil.copy2(src, YOLO_MODEL)
#             print(f"cached weights -> {YOLO_MODEL}")
#     return model


def load_yolo(weights=None):
    raise NotImplementedError(
        "Ultralytics is commented out in prepare-objectgraph. "
        "Uncomment yolo_model.py, paths.py (YOLO_*), and build_scenes.py."
    )
