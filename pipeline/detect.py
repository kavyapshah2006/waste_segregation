"""Run an exported YOLO detector (.pte) on the host and turn its output into Detections.

The Ultralytics ExecuTorch export (8.4.x) keeps the one-to-many head, so the model outputs
raw predictions of shape (1, 4 + num_classes, num_anchors): box centre x, y, width, height in
input pixels, then one sigmoid score per class. Non-maximum suppression happens here (and in
the app), not inside the model.

Pre- and post-processing are spelled out exactly so the Kotlin port can match them:

1. Letterbox: scale r = min(S / h, S / w); resized size = (round_half_even(w * r),
   round_half_even(h * r)), bilinear; pad to S x S with RGB (114, 114, 114); left pad =
   floor((S - new_w) / 2), top pad = floor((S - new_h) / 2).
2. Tensor: RGB, float32, divided by 255, shape (1, 3, S, S).
3. Decode: per anchor, score = max class score, class = argmax; keep score >= conf.
4. NMS: class-aware, IoU threshold `iou`, highest score first, at most `max_det` boxes.
5. Map back: x = (x - left) / r, y = (y - top) / r, clip to the image, then divide by the
   image width and height to get bbox = [x1, y1, x2, y2] in 0-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

INPUT_SIZE = 640
PAD_VALUE = 114
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 100


@dataclass(frozen=True)
class Letterbox:
    scale: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    item_id: str | None
    detector_class: str
    confidence: float
    bbox: tuple[float, float, float, float]
    attributes: dict = field(default_factory=dict)
    source: str = "detector"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "detector_class": self.detector_class,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 4) for v in self.bbox],
            "attributes": dict(self.attributes),
            "source": self.source,
        }


def letterbox(image: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, Letterbox]:
    image = image.convert("RGB")
    w, h = image.size
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)  # Python round = round half to even
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    left, top = math.floor((size - new_w) / 2), math.floor((size - new_h) / 2)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    canvas.paste(resized, (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), Letterbox(r, left, top, w, h)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, max_det: int):
    """Class-aware NMS. Returns indices into the inputs, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        same = classes[rest] == classes[i]
        overlap = iou_one_to_many(boxes[i], boxes[rest]) > iou
        order = rest[~(same & overlap)]
    return keep


def decode(
    output: np.ndarray,
    lb: Letterbox,
    class_names: list[str],
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    max_det: int = MAX_DETECTIONS,
    known_items: set[str] | frozenset[str] = frozenset(),
) -> list[Detection]:
    pred = np.asarray(output, dtype=np.float32)[0].T  # (anchors, 4 + classes)
    if pred.shape[1] != 4 + len(class_names):
        raise ValueError(
            f"model has {pred.shape[1] - 4} classes, names list has {len(class_names)}"
        )
    scores_all = pred[:, 4:]
    classes = scores_all.argmax(1)
    scores = scores_all.max(1)
    mask = scores >= conf
    if not mask.any():
        return []
    cx, cy, w, h = (pred[mask, i] for i in range(4))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scores, classes = scores[mask], classes[mask]

    detections = []
    for i in nms(boxes, scores, classes, iou, max_det):
        x1, y1, x2, y2 = boxes[i]
        x1 = min(max((x1 - lb.left) / lb.scale, 0.0), lb.width) / lb.width
        x2 = min(max((x2 - lb.left) / lb.scale, 0.0), lb.width) / lb.width
        y1 = min(max((y1 - lb.top) / lb.scale, 0.0), lb.height) / lb.height
        y2 = min(max((y2 - lb.top) / lb.scale, 0.0), lb.height) / lb.height
        name = class_names[int(classes[i])]
        detections.append(
            Detection(
                item_id=name if name in known_items else None,
                detector_class=name,
                confidence=float(scores[i]),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return detections


def class_names_for(model_path: Path) -> list[str]:
    """Class names in index order, from the metadata.yaml written next to the .pte."""
    meta = model_path.parent / "metadata.yaml"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found; export with models/vision/export_executorch.py")
    names = yaml.safe_load(meta.read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)]


class Detector:
    def __init__(
        self,
        model_path: Path | str,
        taxonomy_path: Path | str | None = None,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
    ) -> None:
        from executorch.runtime import Runtime

        self.model_path = Path(model_path)
        self.class_names = class_names_for(self.model_path)
        known: set[str] = set()
        if taxonomy_path and Path(taxonomy_path).exists():
            taxonomy = yaml.safe_load(Path(taxonomy_path).read_text(encoding="utf-8"))
            known = set(taxonomy["classes"])
        self.known_items = frozenset(known)
        self.conf, self.iou = conf, iou
        program = Runtime.get().load_program(str(self.model_path))
        self._method = program.load_method("forward")

    def raw(self, tensor: np.ndarray) -> np.ndarray:
        import torch

        (out,) = self._method.execute([torch.from_numpy(tensor)])
        return out.numpy()

    def detect(self, image: Image.Image) -> list[Detection]:
        tensor, lb = letterbox(image)
        return decode(
            self.raw(tensor),
            lb,
            self.class_names,
            self.conf,
            self.iou,
            known_items=self.known_items,
        )

    def detect_path(self, path: Path | str) -> list[Detection]:
        from PIL import ImageOps

        with Image.open(path) as image:
            return self.detect(ImageOps.exif_transpose(image))
