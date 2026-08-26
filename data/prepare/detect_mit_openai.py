#!/usr/bin/env python3
"""
Detect visible objects in MIT Indoor home-room images via OpenAI Vision.

Processes only dwelling-room categories (default: bathroom, bedroom,
children_room, dining_room, kitchen, livingroom) under:

  <mit-dir>/Images/<category>/*.{jpg,jpeg,png}

Writes NYU Depth-style scenes.json: list of object-name lists per image.

Requires OPENAI_KEY or OPENAI_API_KEY in .env (repo root) or environment.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import mimetypes
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

from openai import OpenAI
from tqdm import tqdm

# Allow imports from this package when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import MIT_DIR, SCENES_JSON
from scenes_format import format_label, merge_scenes, yolo_to_scene

# Room folders under mit-indoors/Images/ (MIT Indoor naming)
MIT_TARGET_CATEGORIES = (
    "bathroom",
    "bedroom",
    "children_room",
    "dining_room",
    "kitchen",
    "livingroom",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CHECKPOINT = _SCRIPT_DIR / "cache" / "mit_openai_benchmark.json"
DEFAULT_FAILURES_LOG = _SCRIPT_DIR / "cache" / "mit_openai_failures.jsonl"

DEFAULT_PROMPT = """\
You are labeling objects for an indoor object-graph dataset.

Look at this indoor scene image. List every object that is clearly visible \
(furniture, fixtures, appliances, decor, toiletries, etc.).

Rules:
- Include only objects you can actually see (not guessed or typical for the room).
- Use short common English names (e.g. "Coffee table", "Sink", "Television").
- Capitalize the first letter of each name (title case for multi-word names).
- Do not include the room type, people, pets, or vague regions like "wall" or "floor".
- Do not include duplicates.

Return ONLY a JSON array of strings, nothing else. Example:
["Sink", "Mirror", "Toilet", "Towel"]
"""


def _load_api_key() -> str:
    load_dotenv(_SCRIPT_DIR.parents[1] / ".env")
    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Set OPENAI_KEY or OPENAI_API_KEY in .env or the environment."
        )
    return key


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "image/jpeg"


def _parse_object_list(content: str) -> list[str]:
    text = content.strip()
    if not text:
        return []

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    list_match = re.search(r"\[.*\]", text, re.DOTALL)
    if list_match:
        try:
            parsed = ast.literal_eval(list_match.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    parts = text.replace("[", "").replace("]", "").split(",")
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def detect_objects_openai(
    image_path: Path,
    *,
    client: OpenAI,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    max_retries: int = 3,
) -> list[str]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    mime = _mime_type(image_path)
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                max_completion_tokens=2000,
            )
            content = response.choices[0].message.content or ""
            return _parse_object_list(content)
        except Exception as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAI API failed for {image_path}: {last_err}") from last_err


def iter_mit_images(
    mit_dir: Path,
    categories: tuple[str, ...],
) -> list[Path]:
    images_root = mit_dir / "Images"
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Expected MIT layout at {images_root} "
            f"(extract indoorCVPR_09.tar into {mit_dir})"
        )

    paths: list[Path] = []
    for category in categories:
        folder = images_root / category
        if not folder.is_dir():
            print(f"Warning: missing category folder {folder}", file=sys.stderr)
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(p.resolve())
    return paths


def labels_to_scene(labels: list[str], min_objects: int) -> list[str] | None:
    formatted = [format_label(x) for x in labels]
    return yolo_to_scene(formatted, min_objects=min_objects)


def benchmark_to_scenes(
    benchmark: dict[str, list[str]],
    min_objects: int,
) -> list[list[str]]:
    scenes: list[list[str]] = []
    for labels in benchmark.values():
        row = labels_to_scene(labels, min_objects)
        if row:
            scenes.append(row)
    return scenes


def load_checkpoint(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


def save_checkpoint(path: Path, benchmark: dict[str, list[str]]) -> None:
    """Atomically write checkpoint so a crash mid-write does not corrupt JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".json", prefix=path.stem + "_", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_failure_log(path: Path, image_path: str, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"image": image_path, "error": error, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scenes(path: Path, scenes: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)


def _write_scenes_from_benchmark(args: argparse.Namespace, benchmark: dict[str, list[str]]) -> int:
    scenes = benchmark_to_scenes(benchmark, args.min_objects)
    if args.merge and args.scenes_out.exists():
        with open(args.scenes_out, encoding="utf-8") as f:
            existing = json.load(f)
        scenes = merge_scenes(existing, scenes)
    write_scenes(args.scenes_out, scenes)
    return len(scenes)


def run(args: argparse.Namespace) -> None:
    mit_dir = args.mit_dir.resolve()
    categories = tuple(c.lower() for c in args.categories)

    if args.rebuild_scenes_only:
        benchmark = load_checkpoint(args.checkpoint)
        if not benchmark:
            print(f"No checkpoint at {args.checkpoint}")
            return
        n = _write_scenes_from_benchmark(args, benchmark)
        print(f"Rebuilt {args.scenes_out} ({n} scenes from {len(benchmark)} images)")
        return

    images = iter_mit_images(mit_dir, categories)
    if args.limit:
        images = images[: args.limit]

    if not images:
        print("No images found. Check --mit-dir and category folders.")
        return

    print(f"MIT dir: {mit_dir}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Images to process: {len(images)}")
    print(f"Checkpoint (saved after each image): {args.checkpoint}")

    if args.dry_run:
        for p in images[:20]:
            print(p)
        if len(images) > 20:
            print(f"... and {len(images) - 20} more")
        return

    client = OpenAI(api_key=_load_api_key())
    benchmark = load_checkpoint(args.checkpoint) if args.resume else {}

    todo = [p for p in images if str(p) not in benchmark]
    if args.resume and len(todo) < len(images):
        print(f"Resuming: {len(images) - len(todo)} already in checkpoint, {len(todo)} left")

    interrupted = False

    def _on_interrupt(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True
        print("\nInterrupt received — saving checkpoint and exiting…", file=sys.stderr)

    signal.signal(signal.SIGINT, _on_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_interrupt)

    errors: list[str] = []
    processed_since_scenes = 0
    for image_path in tqdm(todo, desc="OpenAI detection"):
        if interrupted:
            break

        key = str(image_path)
        try:
            labels = detect_objects_openai(
                image_path,
                client=client,
                model=args.model,
            )
            benchmark[key] = labels
        except Exception as exc:
            msg = str(exc)
            errors.append(f"{image_path}: {msg}")
            append_failure_log(args.failures_log, key, msg)
            # Do not add to checkpoint — will retry on next --resume run
            if args.delay:
                time.sleep(args.delay)
            continue

        # Persist after every successful image (safe to resume after network/billing stop)
        save_checkpoint(args.checkpoint, benchmark)
        processed_since_scenes += 1

        if args.write_scenes_every and processed_since_scenes % args.write_scenes_every == 0:
            _write_scenes_from_benchmark(args, benchmark)

        if args.delay:
            time.sleep(args.delay)

    save_checkpoint(args.checkpoint, benchmark)
    n_scenes = _write_scenes_from_benchmark(args, benchmark)

    n_ok = sum(1 for v in benchmark.values() if len(v) > 0)
    print(f"\n{'Interrupted — partial' if interrupted else 'Done'}.")
    print(f"  Checkpoint: {args.checkpoint} ({len(benchmark)} images saved)")
    print(f"  Detections with objects: {n_ok}/{len(benchmark)}")
    print(f"  Scenes written (>= {args.min_objects} objects): {n_scenes}")
    print(f"  Output: {args.scenes_out}")
    if errors:
        print(f"  Failed (not in checkpoint, will retry): {len(errors)}")
        print(f"  Failure log: {args.failures_log}")
        print(f"  First error: {errors[0]}")
    if interrupted:
        print("  Run again with the same command to resume from checkpoint.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="MIT Indoor home rooms → OpenAI object detection → scenes.json"
    )
    p.add_argument(
        "--mit-dir",
        type=Path,
        default=MIT_DIR,
        help="Root containing Images/<category>/ (default: data/mit-indoors)",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(MIT_TARGET_CATEGORIES),
        help="MIT folder names under Images/ (default: 6 home rooms)",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI vision model (e.g. gpt-4o-mini, gpt-4o)",
    )
    p.add_argument(
        "--scenes-out",
        type=Path,
        default=SCENES_JSON,
        help="Output scenes.json path",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Per-image detection cache (resume support)",
    )
    p.add_argument(
        "--min-objects",
        type=int,
        default=2,
        help="Skip scenes with fewer unique objects",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N images")
    p.add_argument("--merge", action="store_true", help="Merge with existing scenes.json")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images already in checkpoint (default: on)",
    )
    p.add_argument(
        "--failures-log",
        type=Path,
        default=DEFAULT_FAILURES_LOG,
        help="Append-only log for API failures (images retried on resume)",
    )
    p.add_argument(
        "--write-scenes-every",
        type=int,
        default=0,
        help="Also update scenes.json every N successes (0 = only at end)",
    )
    p.add_argument(
        "--rebuild-scenes-only",
        action="store_true",
        help="Build scenes.json from checkpoint only (no API calls)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (rate limiting)",
    )
    p.add_argument("--dry-run", action="store_true", help="List images only")
    run(p.parse_args())


if __name__ == "__main__":
    main()
