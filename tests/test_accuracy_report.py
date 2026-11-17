"""eval/accuracy/report.py on a hand-built split whose metrics are worked out by hand."""

import json

import pytest

from eval.accuracy.report import average_precision, ground_truth, iou, load_predictions, score

STREAMS = {"pet_bottle": "dry", "glass_bottle": "dry", "aa_battery": "special_care"}


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "dataset.yaml").write_text(
        "names:\n  0: pet_bottle\n  1: glass_bottle\n  2: aa_battery\n", encoding="utf-8"
    )
    labels = {
        "img1": "0 0.25 0.25 0.2 0.2",  # pet, found
        "img2": "1 0.5 0.5 0.4 0.4",  # glass, predicted as pet (same stream)
        "img3": "2 0.5 0.5 0.2 0.2",  # battery, missed
        "img4": "",  # nothing, a false positive lands here
        "img5": "0 0.5 0.5 0.3 0.3",  # pet, only a low-confidence prediction
    }
    for split_dir in ("images/test", "labels/test"):
        (tmp_path / split_dir).mkdir(parents=True)
    for stem, text in labels.items():
        (tmp_path / "images" / "test" / f"{stem}.jpg").write_bytes(b"")
        (tmp_path / "labels" / "test" / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")

    def pred(image, item, conf, bbox):
        return {"image": image, "detections": [{"item_id": item, "confidence": conf, "bbox": bbox}]}

    preds = [
        pred("img1.jpg", "pet_bottle", 0.9, [0.15, 0.15, 0.35, 0.35]),
        pred("img2.jpg", "pet_bottle", 0.8, [0.3, 0.3, 0.7, 0.7]),
        pred("img4.jpg", "aa_battery", 0.7, [0.1, 0.1, 0.2, 0.2]),
        pred("img5.jpg", "pet_bottle", 0.4, [0.35, 0.35, 0.65, 0.65]),
    ]
    (tmp_path / "preds.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    return tmp_path


def test_report_on_a_hand_worked_split(dataset):
    gt = ground_truth(dataset, "test")
    assert len(gt) == 5 and gt["img4.jpg"] == []
    r = score(gt, load_predictions(dataset / "preds.jsonl"), STREAMS, gate=0.5)
    assert r["objects"] == 4
    assert r["item_accuracy"] == 0.25  # only img1
    assert r["stream_accuracy"] == 0.5  # img1, and img2's glass called pet: both dry
    pet = r["per_class"]["pet_bottle"]
    # AP: 0.9 TP (R .5, P 1), 0.8 FP, 0.4 TP (R 1, P 2/3) -> .5*1 + .5*2/3
    assert pet == {"n_gt": 2, "ap50": 0.8333, "precision": 0.5, "recall": 0.5}
    assert r["per_class"]["glass_bottle"] == {
        "n_gt": 1,
        "ap50": 0.0,
        "precision": None,
        "recall": 0.0,
    }
    assert r["per_class"]["aa_battery"]["precision"] == 0.0
    assert r["map50"] == round((0.8333 + 0 + 0) / 3, 4)
    assert r["confusion"] == {
        "aa_battery": {"missed": 1},
        "glass_bottle": {"pet_bottle": 1},
        "pet_bottle": {"missed": 1, "pet_bottle": 1},
    }
    assert r["background_false_positives"] == {"aa_battery": 1}


def test_lower_gate_trades_misses_for_hits(dataset):
    gt = ground_truth(dataset, "test")
    r = score(gt, load_predictions(dataset / "preds.jsonl"), STREAMS, gate=0.3)
    assert r["item_accuracy"] == 0.5 and r["stream_accuracy"] == 0.75


def test_iou_and_ap_edges():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert iou((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)) == 0.0
    assert average_precision([(0.9, True), (0.8, True)], 2) == 1.0
    assert average_precision([], 0) == 0.0
