"""Detector post-processing cases shared with the Kotlin core
(tests/fixtures/detect_cases.json). Regenerate expected with REGEN=1 and read the diff."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pipeline.detect import decode, letterbox_geometry

PATH = Path(__file__).parent / "fixtures" / "detect_cases.json"
DATA = json.loads(PATH.read_text(encoding="utf-8"))


def _letterbox(case):
    lb, new_w, new_h = letterbox_geometry(case["w"], case["h"], case["size"])
    return {"scale": lb.scale, "new_w": new_w, "new_h": new_h, "left": lb.left, "top": lb.top}


def _decode(case):
    lb, _, _ = letterbox_geometry(*case["image"], 640)
    out = np.asarray(case["output"], dtype=np.float32)[None]
    dets = decode(out, lb, case["names"], case["conf"], case["iou"], case["max_det"],
                  known_items=frozenset(case["known"]))  # fmt: skip
    return [d.to_dict() for d in dets]


@pytest.mark.parametrize("case", DATA["letterbox"], ids=lambda c: f"{c['w']}x{c['h']}")
def test_letterbox_geometry_fixture(case):
    assert _letterbox(case) == case["expected"]


@pytest.mark.parametrize("case", DATA["decode"], ids=lambda c: c["id"])
def test_decode_fixture(case):
    assert _decode(case) == case["expected"]


def test_regen():
    if not os.environ.get("REGEN"):
        pytest.skip("set REGEN=1 to rewrite expected values")
    for case in DATA["letterbox"]:
        case["expected"] = _letterbox(case)
    for case in DATA["decode"]:
        case["expected"] = _decode(case)
    PATH.write_text(json.dumps(DATA, indent=1) + "\n", encoding="utf-8")


def test_expected_decode_reads_right():
    by = {c["id"]: c["expected"] for c in DATA["decode"]}
    first = by["two_bottles_overlap_one_battery"]
    assert [d["detector_class"] for d in first] == [
        "pet_bottle",
        "aa_battery",
        "person",
        "aa_battery",
    ]
    assert first[2]["item_id"] is None  # not a KB item
    assert by["nothing_above_conf"] == []
    assert len(by["max_det_caps_the_result"]) == 2
    assert all(0 <= v <= 1 for d in by["boxes_in_padding_are_clipped"] for v in d["bbox"])
