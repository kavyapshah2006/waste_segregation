"""Download the TACO annotations and images into data/raw/taco/.

Usage:
    uv run python data/scripts/download_taco.py              # all 1,500 images
    uv run python data/scripts/download_taco.py --limit 20   # a quick sample

Images already on disk are skipped, so the script can be re-run after a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TAXONOMY = REPO / "data" / "taxonomy.yaml"
DEFAULT_OUT = REPO / "data" / "raw" / "taco"
USER_AGENT = "waste-segregation-dataset-downloader/0.1"


def fetch(url: str, dest: Path, attempts: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
                fh.write(resp.read())
            tmp.replace(dest)
            return True
        except OSError as exc:
            if attempt == attempts:
                print(f"failed: {url}: {exc}", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                return False
            time.sleep(2 * attempt)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="download only the first N images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    url = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))["datasets"]["taco"]["annotations"]
    ann_path = args.out / "annotations.json"
    if not fetch(url, ann_path):
        return 1
    images = json.loads(ann_path.read_text(encoding="utf-8"))["images"]
    if args.limit:
        images = images[: args.limit]

    jobs = [(img["flickr_url"], args.out / "images" / img["file_name"]) for img in images]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: fetch(*job), jobs))
    ok = sum(results)
    print(f"{ok}/{len(jobs)} images in {args.out / 'images'}")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
