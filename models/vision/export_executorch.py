"""Export trained YOLO weights to ExecuTorch (.pte, XNNPACK) and check the result on the host.

Usage (vision group):
    uv run --group vision python models/vision/export_executorch.py --weights best.pt
    uv run --group vision python models/vision/export_executorch.py --weights yolo26n.pt \\
        --imgsz 640 --check-images data/fixtures/images

Writes <out>/model.pte and <out>/metadata.yaml (class names in index order). The check runs
pipeline.detect on the .pte and the PyTorch model on the same images and fails if a
detection has no match with IoU >= --min-iou.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install from inside our env


def export(weights: Path, imgsz: int, out: Path) -> Path:
    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no usage analytics
    exported = Path(YOLO(str(weights)).export(format="executorch", imgsz=imgsz))
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.pte", "metadata.yaml"):
        shutil.copy2(exported / name, out / name)
    return out / "model.pte"


def check(weights: Path, pte: Path, images: list[Path], imgsz: int, min_iou: float) -> bool:
    from ultralytics import YOLO

    from pipeline.detect import Detector, iou_one_to_many

    detector = Detector(pte, REPO / "data" / "taxonomy.yaml")
    model = YOLO(str(weights))
    ok = True
    for path in images:
        ours = detector.detect_path(path)
        ref = model.predict(str(path), imgsz=imgsz, conf=0.25, iou=0.7, verbose=False)[0]
        h, w = ref.orig_shape
        ref_boxes = ref.boxes.xyxy.numpy() / np.array([w, h, w, h])
        worst = 1.0
        for d in ours:
            best = (
                float(iou_one_to_many(np.array(d.bbox), ref_boxes).max()) if len(ref_boxes) else 0.0
            )
            worst = min(worst, best)
        status = "ok" if worst >= min_iou and len(ours) == len(ref_boxes) else "MISMATCH"
        ok &= status == "ok"
        print(f"{status:8} {path.name}: pte={len(ours)} torch={len(ref_boxes)} min IoU={worst:.3f}")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, help="default: models/vision/out/<weights stem>_<imgsz>")
    p.add_argument("--check-images", type=Path, help="folder of images for the host check")
    p.add_argument("--min-iou", type=float, default=0.9)
    args = p.parse_args(argv)

    out = args.out or REPO / "models" / "vision" / "out" / f"{args.weights.stem}_{args.imgsz}"
    pte = export(args.weights, args.imgsz, out)
    print(f"exported {pte} ({pte.stat().st_size / 1e6:.1f} MB)")
    if args.check_images:
        images = sorted(
            p for p in args.check_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not check(args.weights, pte, images, args.imgsz, args.min_iou):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
