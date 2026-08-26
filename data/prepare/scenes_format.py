"""NYU Depth scenes.json format: list of object-name lists per scene."""


def format_label(name: str) -> str:
    """e.g. 'potted plant' -> 'Potted plant' (matches data/scenes.json style)."""
    s = name.strip().lower()
    if not s:
        return s
    return s[0].upper() + s[1:]


def yolo_to_scene(labels: list[str], min_objects: int = 2) -> list[str] | None:
    seen: set[str] = set()
    scene: list[str] = []
    for raw in labels:
        lab = format_label(raw)
        if lab and lab not in seen:
            seen.add(lab)
            scene.append(lab)
    return scene if len(scene) >= min_objects else None


def benchmark_to_scenes(benchmark: dict, min_objects: int = 2) -> list[list[str]]:
    scenes = []
    for entry in benchmark.values():
        row = yolo_to_scene(entry.get("yolo", []), min_objects)
        if row:
            scenes.append(row)
    return scenes


def merge_scenes(existing: list[list[str]], new: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for batch in (existing, new):
        for scene in batch:
            key = tuple(sorted(scene))
            if key not in seen:
                seen.add(key)
                out.append(scene)
    return out
