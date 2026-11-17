"""Turn downloaded COCO-format datasets into one YOLO-format dataset with our classes.

Usage:
    uv run python data/scripts/prepare.py                       # TACO only
    uv run python data/scripts/prepare.py --coco india=data/raw/india/annotations.json

Writes data/processed/waste/:
    images/{train,val,test}/...   images, EXIF orientation applied
    labels/{train,val,test}/...   one .txt per image: "class cx cy w h" (normalised)
    dataset.yaml                  Ultralytics dataset file (class index order lives here)
    report.json                   instance counts, dropped labels, classes left out

The split is by image and deterministic (hash of dataset name + file name), so adding
images never moves existing ones between train, val and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[2]
TAXONOMY = REPO / "data" / "taxonomy.yaml"
DEFAULT_OUT = REPO / "data" / "processed" / "waste"
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


@dataclass(frozen=True)
class Box:
    cls: str
    cx: float
    cy: float
    w: float
    h: float


def load_taxonomy(path: Path = TAXONOMY) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def label_map(taxonomy: dict, dataset: str, categories: list[str]) -> dict[str, str | None]:
    """Our class (or None to drop) for every label in a dataset; unknown labels are errors."""
    spec = taxonomy["datasets"][dataset]
    classes = set(taxonomy["classes"])
    if spec.get("identity"):
        unknown = sorted(set(categories) - classes)
        if unknown:
            raise ValueError(f"{dataset}: labels are not KB classes: {unknown}")
        return {c: c for c in categories}
    mapping = {label: target for label, target in spec.get("map", {}).items()}
    mapping.update({label: None for label in spec.get("drop", [])})
    unknown = sorted(set(categories) - set(mapping))
    if unknown:
        raise ValueError(f"{dataset}: labels missing from taxonomy map/drop: {unknown}")
    return mapping


def split_for(dataset: str, file_name: str) -> str:
    digest = hashlib.sha256(f"{dataset}/{file_name}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for name, share in SPLITS:
        edge += share
        if u < edge:
            return name
    return SPLITS[-1][0]


def coco_boxes(coco: dict, mapping: dict[str, str | None]) -> tuple[dict[int, list[Box]], Counter]:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    sizes = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    boxes: dict[int, list[Box]] = {img["id"]: [] for img in coco["images"]}
    dropped: Counter = Counter()
    for ann in coco["annotations"]:
        label = names[ann["category_id"]]
        target = mapping[label]
        if target is None:
            dropped[label] += 1
            continue
        w_img, h_img = sizes[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w_img), x + w), min(float(h_img), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        boxes[ann["image_id"]].append(
            Box(
                target,
                (x0 + x1) / 2 / w_img,
                (y0 + y1) / 2 / h_img,
                (x1 - x0) / w_img,
                (y1 - y0) / h_img,
            )
        )
    return boxes, dropped


def prepare(sources: dict[str, Path], out: Path, taxonomy: dict) -> dict:
    """Build the dataset; returns the report. Images must sit in <annotations dir>/images/."""
    collected = []  # (dataset, image_path, file_name, boxes, (w, h))
    dropped_by_dataset = {}
    for dataset, ann_path in sources.items():
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        mapping = label_map(taxonomy, dataset, [c["name"] for c in coco["categories"]])
        boxes, dropped = coco_boxes(coco, mapping)
        dropped_by_dataset[dataset] = dict(dropped)
        image_dir = ann_path.parent / "images"
        for img in coco["images"]:
            path = image_dir / img["file_name"]
            if path.exists() and boxes[img["id"]]:
                size = (img["width"], img["height"])
                collected.append((dataset, path, img["file_name"], boxes[img["id"]], size))

    counts = Counter(b.cls for *_, bxs, _ in collected for b in bxs)
    min_n = taxonomy.get("min_instances", 1)
    used = [c for c in taxonomy["classes"] if counts[c] >= min_n]
    left_out = {c: counts[c] for c in taxonomy["classes"] if counts[c] < min_n}
    index = {c: i for i, c in enumerate(used)}

    if out.exists():
        shutil.rmtree(out)
    split_counts: Counter = Counter()
    skipped_size = []
    for dataset, path, file_name, bxs, (w, h) in collected:
        kept = [b for b in bxs if b.cls in index]
        if not kept:
            continue
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != (w, h):
                skipped_size.append(
                    f"{dataset}/{file_name}: image {im.size} vs annotation {(w, h)}"
                )
                continue
            split = split_for(dataset, file_name)
            stem = f"{dataset}__{Path(file_name).with_suffix('').as_posix().replace('/', '__')}"
            img_out = out / "images" / split / f"{stem}.jpg"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            im.save(img_out, quality=92)
        lbl_out = out / "labels" / split / f"{stem}.txt"
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.write_text(
            "".join(f"{index[b.cls]} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for b in kept),
            encoding="utf-8",
        )
        split_counts[split] += 1

    out.mkdir(parents=True, exist_ok=True)
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(used)),
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), "utf-8")
    report = {
        "images_per_split": dict(split_counts),
        "instances": {c: counts[c] for c in used},
        "classes_left_out": left_out,
        "dropped_labels": dropped_by_dataset,
        "skipped_size_mismatch": skipped_size,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--coco",
        action="append",
        default=[],
        metavar="NAME=ANNOTATIONS.json",
        help="extra COCO dataset; NAME must be a dataset in taxonomy.yaml",
    )
    parser.add_argument("--no-taco", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    sources = {} if args.no_taco else {"taco": REPO / "data" / "raw" / "taco" / "annotations.json"}
    for spec in args.coco:
        name, _, path = spec.partition("=")
        sources[name] = Path(path)
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        print("missing annotations (run the download script first):", *missing, file=sys.stderr)
        return 1
    report = prepare(sources, args.out, load_taxonomy())
    print(json.dumps({k: report[k] for k in ("images_per_split", "classes_left_out")}, indent=2))
    print(f"dataset: {args.out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
