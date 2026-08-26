#!/usr/bin/env python3
"""Download and extract MIT Indoor Scene Recognition dataset (CVPR 2009)."""
import argparse
import tarfile
import urllib.request
from pathlib import Path

from paths import CACHE, MIT_DIR, MIT_TAR, MIT_URL

# Official archive is ~2.41 GiB (2_592_010_240 bytes). Partial downloads are often ~few MB–GB.
MIT_TAR_EXPECTED_BYTES = 2_592_010_240
MIT_TAR_MIN_BYTES = 2_500_000_000


def _check_tar(path: Path) -> str | None:
    """Return error message if tar looks bad, else None."""
    if not path.is_file():
        return "file missing"
    size = path.stat().st_size
    if size < MIT_TAR_MIN_BYTES:
        return f"incomplete download ({size:,} bytes, need ~{MIT_TAR_MIN_BYTES:,})"
    if not tarfile.is_tarfile(path):
        return "not a valid tar archive"
    try:
        with tarfile.open(path, "r") as tar:
            if not tar.getmembers():
                return "empty tar"
    except (tarfile.ReadError, EOFError, OSError) as e:
        return f"corrupt tar index: {e}"
    return None


def download(url: str = MIT_URL, dest: Path = MIT_TAR, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        err = _check_tar(dest)
        if err is None:
            print(f"already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
            return dest
        print(f"removing bad archive: {err}")
        dest.unlink()

    print(f"downloading {url}")
    print(f"  -> {dest}  (~{MIT_TAR_EXPECTED_BYTES / 1e9:.1f} GB)")
    tmp = dest.with_suffix(".tar.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0) or MIT_TAR_EXPECTED_BYTES)
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        pct = min(100, done * 100 / total)
                        print(f"\r  {pct:5.1f}% ({done:,} / {total:,} bytes)", end="", flush=True)
        print()
        if total > 0 and done < total * 0.99:
            tmp.unlink(missing_ok=True)
            raise OSError(f"download stopped early ({done:,} < {total:,} bytes)")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    err = _check_tar(dest)
    if err:
        dest.unlink(missing_ok=True)
        raise OSError(f"download failed validation: {err}")
    print(f"saved {dest.stat().st_size:,} bytes")
    return dest


def extract(tar_path: Path = MIT_TAR, out_dir: Path = MIT_DIR) -> Path:
    err = _check_tar(tar_path)
    if err:
        raise SystemExit(
            f"Cannot extract: {err}\n"
            f"Delete and re-download:\n"
            f"  rm -f {tar_path}\n"
            f"  python download_mit_indoor.py"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tar_path} -> {out_dir}")
    try:
        with tarfile.open(tar_path, "r") as tar:
            try:
                tar.extractall(path=out_dir, filter="data")
            except TypeError:
                tar.extractall(path=out_dir)
    except tarfile.ReadError as e:
        raise SystemExit(
            f"Extract failed ({e}). Archive is corrupt — delete and re-download:\n"
            f"  rm -f {tar_path}\n"
            f"  python download_mit_indoor.py --force-download"
        ) from e

    images_root = out_dir / "Images"
    if images_root.is_dir():
        n = sum(1 for _ in images_root.rglob("*.jpg"))
        print(f"extracted under {images_root} ({n} jpg files)")
    else:
        n = sum(1 for _ in out_dir.rglob("*.jpg"))
        print(f"extracted under {out_dir} ({n} jpg files)")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Download MIT Indoor dataset for ObjectGraph")
    p.add_argument("--skip-download", action="store_true", help="only extract existing tar")
    p.add_argument("--force-download", action="store_true", help="re-download even if tar exists")
    p.add_argument("--tar", type=Path, default=MIT_TAR)
    p.add_argument("--out", type=Path, default=MIT_DIR)
    args = p.parse_args()
    tar = args.tar
    if not args.skip_download:
        tar = download(dest=tar, force=args.force_download)
    if not tar.exists():
        raise SystemExit(f"tar not found: {tar}")
    extract(tar, args.out)


if __name__ == "__main__":
    main()
