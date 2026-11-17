"""Detector accuracy on a held-out split (FuturePlan WP11). The headline is stream accuracy.

    # 1. predictions with the exported detector (laptop; the same .pte the phone runs)
    uv run --group vision python eval/accuracy/report.py predict \\
        --dataset data/prepared --split test --detector models/vision/out/yolo26n.pte \\
        --out eval/tmp/preds_test.jsonl
    # 2. score them
    uv run python eval/accuracy/report.py score --dataset data/prepared --split test \\
        --predictions eval/tmp/preds_test.jsonl --label yolo26n_test

Ground truth is the prepared dataset (data/scripts/prepare.py): labels/<split>/*.txt with
"class cx cy w h", class order from dataset.yaml (class names are KB item ids).
Predictions are JSONL lines {"image": "<file name>", "detections": [Detection.to_dict()]}.

Metrics, all at IoU >= 0.5:
- per class: AP50 (all-point interpolated, all predictions), and precision / recall at the
  confidence gate; mAP50 over classes that have ground truth;
- item accuracy: share of ground-truth objects whose best overlapping prediction (at or above
  the gate, any class) names the right item;
- **stream accuracy**: the same, but right if the predicted item belongs to the same KB stream
  (what the user needs: the right bin). This is the number behind the 92% target. Missed
  objects count as wrong;
- confusion: ground-truth item -> predicted item (or "missed"), plus false positives on
  background.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS = REPO / "eval" / "results"
IOU = 0.5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_names(dataset: Path) -> list[str]:
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def ground_truth(dataset: Path, split: str) -> dict[str, list[tuple[str, tuple]]]:
    """image file name -> [(item_id, (x1, y1, x2, y2) normalised)]"""
    names = class_names(dataset)
    out: dict[str, list] = {}
    for image in sorted((dataset / "images" / split).iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = dataset / "labels" / split / f"{image.stem}.txt"
        boxes = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()
                cx, cy, w, h = map(float, (cx, cy, w, h))
                boxes.append((names[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        out[image.name] = boxes
    return out


def average_precision(scored: list[tuple[float, bool]], n_gt: int) -> float:
    """All-point interpolated AP from (confidence, is_true_positive) pairs."""
    if n_gt == 0:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tp = fp = 0
    points = []
    for _, hit in scored:
        tp += hit
        fp += not hit
        points.append((tp / n_gt, tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for i, (recall, _) in enumerate(points):
        best_precision = max(p for _, p in points[i:])
        ap += (recall - prev_recall) * best_precision
        prev_recall = recall
    return ap


def _match(gt_boxes, preds, same_class: bool):
    """Greedy matching by confidence. Returns (pairs [(gt_index, pred)], unmatched preds)."""
    used, pairs, extra = set(), [], []
    for pred in sorted(preds, key=lambda p: -p["confidence"]):
        best, best_iou = None, IOU
        for i, (item, box) in enumerate(gt_boxes):
            if i in used or (same_class and item != pred["item_id"]):
                continue
            o = iou(box, pred["bbox"])
            if o >= best_iou:
                best, best_iou = i, o
        if best is None:
            extra.append(pred)
        else:
            used.add(best)
            pairs.append((best, pred))
    return pairs, extra


def score(gt: dict, predictions: dict, stream_of: dict[str, str], gate: float) -> dict:
    classes = sorted({item for boxes in gt.values() for item, _ in boxes})
    n_gt = Counter(item for boxes in gt.values() for item, _ in boxes)
    ap_scores: dict[str, list] = defaultdict(list)
    tp = Counter()
    fp = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    background_fp = Counter()
    item_right = stream_right = total = 0

    for image, boxes in gt.items():
        preds = [p for p in predictions.get(image, []) if p.get("item_id")]
        # AP: class-aware matching over every prediction.
        for item in {p["item_id"] for p in preds} | {b[0] for b in boxes}:
            mine = [p for p in preds if p["item_id"] == item]
            pairs, extra = _match([b for b in boxes], mine, same_class=True)
            ap_scores[item] += [(p["confidence"], True) for _, p in pairs]
            ap_scores[item] += [(p["confidence"], False) for p in extra]
        gated = [p for p in preds if p["confidence"] >= gate]
        pairs, extra = _match(boxes, gated, same_class=True)
        for _, p in pairs:
            tp[p["item_id"]] += 1
        for p in extra:
            fp[p["item_id"]] += 1
        # Accuracy and confusion: class-agnostic matching, so a wrong item still pairs up.
        pairs, extra = _match(boxes, gated, same_class=False)
        predicted = {i: p["item_id"] for i, p in pairs}
        for i, (item, _) in enumerate(boxes):
            total += 1
            guess = predicted.get(i)
            confusion[item][guess or "missed"] += 1
            item_right += guess == item
            stream_right += guess is not None and stream_of.get(guess) == stream_of.get(item)
        for p in extra:
            background_fp[p["item_id"]] += 1

    per_class = {}
    for item in sorted(set(classes) | set(ap_scores)):
        n = n_gt[item]
        per_class[item] = {
            "n_gt": n,
            "ap50": round(average_precision(ap_scores[item], n), 4) if n else None,
            "precision": round(tp[item] / (tp[item] + fp[item]), 4)
            if tp[item] + fp[item]
            else None,
            "recall": round(tp[item] / n, 4) if n else None,
        }
    aps = [c["ap50"] for c in per_class.values() if c["ap50"] is not None]
    return {
        "images": len(gt),
        "objects": total,
        "gate": gate,
        "iou": IOU,
        "map50": round(sum(aps) / len(aps), 4) if aps else None,
        "item_accuracy": round(item_right / total, 4) if total else None,
        "stream_accuracy": round(stream_right / total, 4) if total else None,
        "per_class": per_class,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
        "background_false_positives": dict(sorted(background_fp.items())),
    }


def load_predictions(path: Path) -> dict[str, list[dict]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["image"]] = row["detections"]
    return out


def kb_streams() -> dict[str, str]:
    from kb.build_kb import build
    from pipeline.kb import KnowledgeBase

    db = REPO / "eval" / "tmp" / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    return {i: kb.item(i)["stream"] for i in kb.item_ids}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("predict", help="run the detector over a split")
    pr.add_argument("--dataset", type=Path, required=True)
    pr.add_argument("--split", default="test")
    pr.add_argument("--detector", type=Path, required=True)
    pr.add_argument("--taxonomy", type=Path, default=REPO / "data" / "taxonomy.yaml")
    pr.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score", help="score predictions against the split's labels")
    sc.add_argument("--dataset", type=Path, required=True)
    sc.add_argument("--split", default="test")
    sc.add_argument("--predictions", type=Path, required=True)
    sc.add_argument("--gate", type=float, default=0.5, help="confidence gate (pipeline/stability)")
    sc.add_argument("--label", help="save to eval/results/accuracy_<label>.json")
    args = p.parse_args(argv)

    if args.cmd == "predict":
        from pipeline.detect import Detector

        detector = Detector(args.detector, args.taxonomy)
        images = sorted(
            f for f in (args.dataset / "images" / args.split).iterdir()
            if f.suffix.lower() in IMAGE_SUFFIXES
        )  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for image in images:
                dets = [d.to_dict() for d in detector.detect_path(image)]
                fh.write(json.dumps({"image": image.name, "detections": dets}) + "\n")
        print(f"wrote {len(images)} predictions to {args.out}")
        return 0

    report = score(
        ground_truth(args.dataset, args.split),
        load_predictions(args.predictions),
        kb_streams(),
        args.gate,
    )
    report = {"split": args.split, **report}
    summary = {
        k: report[k] for k in ("images", "objects", "map50", "item_accuracy", "stream_accuracy")
    }
    print(json.dumps(summary, indent=2))
    if args.label:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"accuracy_{args.label}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
