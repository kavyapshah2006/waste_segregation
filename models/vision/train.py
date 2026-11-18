"""Train the YOLO26n waste detector on the prepared dataset.

Meant for a GPU runtime (Colab or Kaggle, see train.ipynb). On the laptop use --smoke to
check the pipeline end to end in a few minutes on CPU, or --device cpu for a slow but complete
baseline run (hours; the GPU run on Colab is the one to publish).

Usage:
    python models/vision/train.py --data data/processed/waste/dataset.yaml
    python models/vision/train.py --data data/processed/waste/dataset.yaml --smoke

Writes runs to models/vision/runs/<name>/ and a per-class metrics table to
models/vision/runs/<name>/per_class.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="yolo26n.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--name", default="yolo26n_waste")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0, help="share of training images to use")
    p.add_argument("--device", help="e.g. cpu or 0; default lets Ultralytics choose")
    p.add_argument("--workers", type=int, default=8, help="data loader processes")
    p.add_argument("--patience", type=int, default=100, help="stop after N epochs without gain")
    p.add_argument("--smoke", action="store_true", help="1 epoch, small images, CPU")
    args = p.parse_args(argv)

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(args.model)
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        fraction=args.fraction,
        workers=args.workers,
        patience=args.patience,
        deterministic=True,
        project=str(REPO / "models" / "vision" / "runs"),
        name=args.name,
        exist_ok=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    if args.smoke:
        kwargs.update(epochs=1, imgsz=320, batch=4, device="cpu", workers=0)
    model.train(**kwargs)

    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=kwargs["imgsz"],
        project=kwargs["project"],
        name=f"{args.name}_test",
        exist_ok=True,
    )
    per_class = {
        model.names[int(c)]: {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "map50": float(metrics.box.ap50[i]),
            "map50_95": float(metrics.box.ap[i]),
        }
        for i, c in enumerate(metrics.box.ap_class_index)
    }
    summary = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), **per_class}
    out = Path(model.trainer.save_dir) / "per_class.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"per-class metrics: {out}")
    print(f"best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
