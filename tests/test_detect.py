import numpy as np
import pytest
import torch
import yaml
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from PIL import Image

from pipeline.detect import Detector, Letterbox, decode, letterbox, nms

REPO_TAXONOMY = "data/taxonomy.yaml"


def test_letterbox_landscape():
    image = Image.new("RGB", (1280, 720), (255, 0, 0))
    tensor, lb = letterbox(image)
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert lb == Letterbox(scale=0.5, left=0, top=140, width=1280, height=720)
    assert np.allclose(tensor[0, :, 0, 0], 114 / 255)  # padding
    assert np.allclose(tensor[0, :, 320, 320], [1.0, 0.0, 0.0])  # image content, RGB order


def test_letterbox_odd_padding_rounds_down():
    _, lb = letterbox(Image.new("RGB", (100, 61)))
    # r = 6.4, new size 640 x 390 (390.4 rounds to 390), padding 250 -> top 125
    assert (lb.left, lb.top) == (0, 125)


def _pred(rows, num_classes):
    """rows: (cx, cy, w, h, class, score) in input pixels -> (1, 4 + C, N) model output."""
    out = np.zeros((1, 4 + num_classes, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_nms_and_mapping():
    output = _pred(
        [
            (100, 300, 50, 50, 0, 0.9),
            (102, 301, 50, 50, 0, 0.8),  # same class, overlaps the first: suppressed
            (101, 300, 50, 50, 1, 0.7),  # other class, same place: kept (class-aware NMS)
            (400, 400, 20, 20, 0, 0.1),  # below the confidence threshold
            (10, 10, 40, 40, 0, 0.6),  # mostly in the top padding: clipped to the image edge
        ],
        num_classes=2,
    )
    lb = Letterbox(scale=0.5, left=0, top=140, width=1280, height=720)
    dets = decode(output, lb, ["pet_bottle", "dog"], known_items={"pet_bottle"})
    assert [(d.detector_class, d.item_id) for d in dets] == [
        ("pet_bottle", "pet_bottle"),
        ("dog", None),
        ("pet_bottle", "pet_bottle"),
    ]
    # Input box x 75-125, y 275-325 -> original x 150-250, y (275-140)/0.5=270 to 370.
    assert dets[0].bbox == pytest.approx((150 / 1280, 270 / 720, 250 / 1280, 370 / 720))
    # Input y -10 to 30 lies above the top pad (140): clipped to y = 0.
    assert dets[2].bbox[1] == 0.0 and dets[2].bbox[3] == 0.0


def test_decode_rejects_wrong_class_count():
    with pytest.raises(ValueError, match="classes"):
        decode(_pred([(1, 1, 1, 1, 0, 0.9)], 2), Letterbox(1, 0, 0, 1, 1), ["only_one"])


def test_nms_respects_max_det():
    boxes = np.array([[i * 10, 0, i * 10 + 5, 5] for i in range(10)], dtype=np.float32)
    keep = nms(boxes, np.linspace(1, 0.1, 10), np.zeros(10), 0.5, max_det=3)
    assert keep == [0, 1, 2]


class _FixedHead(torch.nn.Module):
    """Stands in for YOLO: returns a fixed (1, 4 + 2, 3) prediction whatever the image."""

    def __init__(self, pred: np.ndarray):
        super().__init__()
        self.register_buffer("pred", torch.from_numpy(pred))

    def forward(self, x):
        return self.pred + 0 * x.mean()


def test_detector_runs_a_pte_end_to_end(tmp_path):
    pred = _pred(
        [(320, 320, 64, 64, 0, 0.95), (100, 100, 20, 20, 1, 0.6), (5, 5, 2, 2, 1, 0.05)], 2
    )
    example = (torch.zeros(1, 3, 640, 640),)
    program = to_edge_transform_and_lower(
        torch.export.export(_FixedHead(pred).eval(), example), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    (tmp_path / "model.pte").write_bytes(program.buffer)
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"names": {0: "pet_bottle", 1: "dog"}}))

    detector = Detector(tmp_path / "model.pte", REPO_TAXONOMY)
    dets = detector.detect(Image.new("RGB", (640, 640)))

    assert [d.item_id for d in dets] == ["pet_bottle", None]
    assert dets[0].to_dict()["bbox"] == [0.45, 0.45, 0.55, 0.55]
    assert dets[0].confidence == pytest.approx(0.95)
