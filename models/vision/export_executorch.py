"""Export trained YOLO weights to ExecuTorch (.pte, XNNPACK) and check the result on the host.

Usage (vision group):
    uv run --group vision python models/vision/export_executorch.py --weights best.pt
    uv run --group vision python models/vision/export_executorch.py --weights yolo26n.pt \\
        --imgsz 640 --check-images data/fixtures/images

Writes <out>/model.pte and <out>/metadata.yaml (class names in index order). The check feeds the
same letterboxed tensor to the .pte and the PyTorch model and fails if raw boxes or scores
differ by more than --max-box-px / --max-score or the decoded detections differ.
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


def check(
    weights: Path, pte: Path, images: list[Path], max_box_px: float, max_score: float
) -> bool:
    """Run the .pte and the PyTorch model on the same letterboxed tensor and compare raw outputs.

    Both use the head the export keeps (one-to-many, end2end off), so boxes and scores must
    agree to float precision and the decoded detections must be identical. Comparing against
    Ultralytics predict() instead would mix in its rectangular letterbox and NMS settings,
    which change borderline boxes without saying anything about the export.
    """
    import torch
    from PIL import Image
    from ultralytics import YOLO

    from pipeline.detect import Detector, decode, letterbox

    detector = Detector(pte, REPO / "data" / "taxonomy.yaml")
    model = YOLO(str(weights)).model.float().eval()
    model.model[-1].end2end = False
    ok = True
    for path in images:
        tensor, lb = letterbox(Image.open(path).convert("RGB"))
        ours = detector.raw(tensor)
        with torch.no_grad():
            ref = model(torch.from_numpy(tensor))
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
        box_diff = float(np.abs(ours[:, :4] - ref[:, :4]).max())
        score_diff = float(np.abs(ours[:, 4:] - ref[:, 4:]).max())
        a = decode(ours, lb, detector.class_names)
        b = decode(ref, lb, detector.class_names)
        same = len(a) == len(b) and all(
            x.detector_class == y.detector_class for x, y in zip(a, b, strict=True)
        )
        good = same and box_diff <= max_box_px and score_diff <= max_score
        ok &= good
        print(
            f"{'ok' if good else 'MISMATCH':8} {path.name}: dets={len(a)}/{len(b)} "
            f"box diff={box_diff:.4f} px score diff={score_diff:.2e}"
        )
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, help="default: models/vision/out/<weights stem>_<imgsz>")
    p.add_argument("--check-images", type=Path, help="folder of images for the host check")
    p.add_argument("--max-box-px", type=float, default=0.5, help="max raw box difference")
    p.add_argument("--max-score", type=float, default=1e-3, help="max raw score difference")
    args = p.parse_args(argv)

    out = args.out or REPO / "models" / "vision" / "out" / f"{args.weights.stem}_{args.imgsz}"
    pte = export(args.weights, args.imgsz, out)
    print(f"exported {pte} ({pte.stat().st_size / 1e6:.1f} MB)")
    if args.check_images:
        images = sorted(
            p for p in args.check_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not check(args.weights, pte, images, args.max_box_px, args.max_score):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
