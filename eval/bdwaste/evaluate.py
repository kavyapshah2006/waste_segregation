"""Four-bin test on BDWaste: South Asian close-up phone photos of single waste items.

BDWaste (Rahman et al., Data in Brief 53, 2024, doi:10.1016/j.dib.2024.110153; data
doi:10.17632/96g5pgfnfw.1, CC BY 4.0) has 20 item classes photographed around Dhaka. Each
photo shows one kind of item, so it tests the app's photo path end to end:

1. detector (the exported .pte, app pre/post-processing): the most confident item detection;
   at or above the gate it becomes the verdict only if SigLIP also ranks that item in its top 3
   (the agreement rule, pipeline.fallback.confirms). "detector_first_*" scores the previous
   rule, where the detector decided alone;
2. otherwise the SigLIP 2 fallback ranks the KB items; the app shows the top 3 for the user
   to pick, never a verdict. Scored here as top-1 (what a user tapping the first suggestion
   gets) and top-3 (the right item is among the offered three).

Ground truth: each class maps to the KB items that are right for it and to the item's KB
stream. Classes with no KB item (masks, gloves) are skipped; "Pastic" (jugs, jerrycans,
buckets) has no exact KB item and is scored on the stream only.

    uv run --group finetune python eval/bdwaste/evaluate.py --images data/raw/bdwaste/images \\
        --label yolo26n_siglip2
Per-image results are cached in eval/tmp/bdwaste_<label>.jsonl, so reruns only add new images.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS = REPO / "eval" / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# BDWaste class (folder name, whitespace-normalised) -> KB items that count as right.
# An empty set means "no exact KB item": scored on the stream only (STREAM_ONLY).
CLASS_ITEMS: dict[str, set[str]] = {
    # digestible
    "Banana peel": {"fruit_veg_peels"},
    "Lemon Peel": {"fruit_veg_peels"},
    "Mango Peel": {"fruit_veg_peels"},
    "Potato Peel": {"fruit_veg_peels"},
    "Shell of Malta": {"fruit_veg_peels"},
    "Sugarcane husk": {"fruit_veg_peels", "garden_waste"},
    "Egg shell": {"eggshells"},
    "Rice": {"cooked_food_leftovers", "food_scraps"},
    "Fish ash": {"meat_fish_bones", "food_scraps"},
    "Paper": {"newspaper"},
    "Coffee cup": {"paper_cup"},
    # indigestible
    "Polythene": {"plastic_carry_bag"},
    "Bottle": {"pet_bottle"},
    "Cane": {"aluminium_can", "tin_can"},
    "Chips packet": {"chips_packet"},
    "Empty medicine packet": {"blister_pack"},
    "Glass": {"broken_glass"},
    "Wire": {"charger_cable"},
    "Pastic": set(),
}
STREAM_ONLY = {"Pastic": "dry"}
SKIPPED = {"Mask", "Glaves"}  # no KB item for masks or gloves


def kb_streams() -> dict[str, str]:
    from kb.build_kb import build
    from pipeline.kb import KnowledgeBase

    db = REPO / "eval" / "tmp" / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    return {i: kb.item(i)["stream"] for i in kb.item_ids}


def truth_stream(cls: str, streams: dict[str, str]) -> str:
    if cls in STREAM_ONLY:
        return STREAM_ONLY[cls]
    found = {streams[i] for i in CLASS_ITEMS[cls]}
    assert len(found) == 1, (cls, found)
    return found.pop()


def summarise(rows: list[dict], streams: dict[str, str], gate: float) -> dict:
    """rows: {"class", "det": [item, conf] or None, "top3": [item, item, item]}."""
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cls = r["class"]
        right_items = CLASS_ITEMS[cls]
        stream = truth_stream(cls, streams)
        c = per[cls]
        c["images"] += 1
        det = r["det"]
        verdict = det is not None and det[1] >= gate
        top1 = r["top3"][0]
        if verdict:
            c["det_verdicts"] += 1
            c["det_item_right"] += det[0] in right_items
            c["det_stream_right"] += streams[det[0]] == stream
        c["fb_top1_item"] += top1 in right_items
        c["fb_top3_item"] += any(i in right_items for i in r["top3"])
        c["fb_top1_stream"] += streams[top1] == stream
        # Previous app rule: the detector's verdict whenever it clears the gate.
        first_item = det[0] if verdict else top1
        c["first_stream_right"] += streams[first_item] == stream
        c["first_item_right"] += first_item in right_items
        # Shipped rule (pipeline.fallback.confirms): the verdict only if SigLIP agrees.
        agreed = verdict and det[0] in r["top3"]
        app_item = det[0] if agreed else top1
        c["app_verdicts"] += agreed
        c["app_verdict_stream_right"] += agreed and streams[det[0]] == stream
        c["app_stream_right"] += streams[app_item] == stream
        c["app_item_right"] += app_item in right_items

    def rates(c: dict) -> dict:
        n = c["images"]
        v = c["det_verdicts"]
        return {
            "images": n,
            "detector_coverage": round(v / n, 4),
            "detector_item_acc": round(c["det_item_right"] / v, 4) if v else None,
            "detector_stream_acc": round(c["det_stream_right"] / v, 4) if v else None,
            "fallback_top1_item": round(c["fb_top1_item"] / n, 4),
            "fallback_top3_item": round(c["fb_top3_item"] / n, 4),
            "fallback_top1_stream": round(c["fb_top1_stream"] / n, 4),
            "detector_first_item_acc": round(c["first_item_right"] / n, 4),
            "detector_first_stream_acc": round(c["first_stream_right"] / n, 4),
            "app_verdict_coverage": round(c["app_verdicts"] / n, 4),
            "app_verdict_stream_acc": (
                round(c["app_verdict_stream_right"] / c["app_verdicts"], 4)
                if c["app_verdicts"]
                else None
            ),
            "app_item_acc": round(c["app_item_right"] / n, 4),
            "app_stream_acc": round(c["app_stream_right"] / n, 4),
        }

    per_class = {cls: rates(c) | {"stream": truth_stream(cls, streams)} for cls, c in per.items()}
    by_stream: dict[str, dict] = {}
    for s in sorted({v["stream"] for v in per_class.values()}):
        tot = defaultdict(int)
        for cls, c in per.items():
            if truth_stream(cls, streams) == s:
                for k, v in c.items():
                    tot[k] += v
        by_stream[s] = rates(tot)
    allc = defaultdict(int)
    for c in per.values():
        for k, v in c.items():
            allc[k] += v
    # Macro average over streams, so the many dry images do not hide wet or special-care errors.
    macro = {
        k: round(sum(s[k] for s in by_stream.values()) / len(by_stream), 4)
        for k in (
            "app_stream_acc",
            "detector_first_stream_acc",
            "fallback_top1_stream",
            "fallback_top3_item",
        )
    }
    return {
        "gate": gate,
        "overall": rates(allc),
        "macro_over_streams": macro,
        "by_stream": by_stream,
        "per_class": dict(sorted(per_class.items())),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images", type=Path, required=True, help="folder with one subfolder per class")
    p.add_argument("--detector", type=Path, default=REPO / "models/vision/out/model.pte")
    p.add_argument(
        "--fallback", type=Path, default=REPO / "models/fallback/out/siglip2_image_int8.pte"
    )
    p.add_argument("--text", type=Path, default=REPO / "models/fallback/out/text_embeddings.json")
    p.add_argument("--gate", type=float, default=0.5)
    p.add_argument("--label", required=True)
    args = p.parse_args(argv)

    from PIL import Image, ImageOps

    from pipeline.detect import Detector
    from pipeline.fallback import Fallback

    cache = REPO / "eval" / "tmp" / f"bdwaste_{args.label}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["class"], r["file"])] = r
    detector = Detector(args.detector, REPO / "data" / "taxonomy.yaml")
    fallback = Fallback(args.fallback, args.text)
    with cache.open("a", encoding="utf-8") as fh:
        for folder in sorted(d for d in args.images.iterdir() if d.is_dir()):
            cls = " ".join(folder.name.split())
            if cls in SKIPPED:
                continue
            if cls not in CLASS_ITEMS:
                raise SystemExit(f"unmapped BDWaste class: {cls}")
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            new = 0
            for f in files:
                if (cls, f.name) in done:
                    continue
                image = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                dets = [d for d in detector.detect(image) if d.item_id]
                best = max(dets, key=lambda d: d.confidence) if dets else None
                top3 = [s["item_id"] for s in fallback.suggest(image, k=3)]
                r = {
                    "class": cls,
                    "file": f.name,
                    "det": [best.item_id, round(best.confidence, 4)] if best else None,
                    "top3": top3,
                }
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[(cls, f.name)] = r
                new += 1
            print(f"{cls}: {len(files)} images ({new} new)", flush=True)

    report = summarise(list(done.values()), kb_streams(), args.gate)
    report = {"dataset": "BDWaste (CC BY 4.0)", "label": args.label, **report}
    headline = ("overall", "macro_over_streams", "by_stream")
    print(json.dumps({k: report[k] for k in headline}, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"bdwaste_{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
