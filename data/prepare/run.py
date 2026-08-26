#!/usr/bin/env python3
"""Download MIT Indoor. Ultralytics build step commented out."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(script: str, extra: list[str] | None = None):
    cmd = [sys.executable, str(HERE / script)] + (extra or [])
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser(description="Prepare ObjectGraph data (download only)")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    dl_args = ["--skip-download"] if args.skip_download else []
    _run("download_mit_indoor.py", dl_args)

    # --- Ultralytics: uncomment to run YOLO after download ---
    # p.add_argument("--limit", type=int, default=0)
    # p.add_argument("--merge", action="store_true", default=True)
    # bs = ["--merge", "--write-benchmark"] if args.merge else ["--write-benchmark"]
    # if args.limit:
    #     bs += ["--limit", str(args.limit)]
    # _run("build_scenes.py", bs)

    print("Download done. YOLO/build_scenes is commented out — use data/scenes.json or uncomment run.py.")


if __name__ == "__main__":
    main()
