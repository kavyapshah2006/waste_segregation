"""Zero-shot check of the fallback on labelled object crops from a prepared dataset.

    uv run --group finetune python models/fallback/check_crops.py --dataset eval/tmp/taco80 \\
        --label taco80_sanity

Crops every ground-truth box (with 10% margin) from all splits and asks the fallback for its
top 3 items. Reports top-1/top-3 item and stream hit rates for each encoder, and how often
fp32 and int8 agree. A small TACO sample is a sanity check, not a benchmark: TACO is
mostly European litter photos, and several of our classes never appear in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "models" / "fallback" / "out"
RESULTS = REPO / "eval" / "results"


def crops(dataset: Path):
    from eval.accuracy.report import ground_truth

    for split in ("train", "val", "test"):
        if not (dataset / "images" / split).exists():
            continue
        for image_name, boxes in ground_truth(dataset, split).items():
            if not boxes:
                continue
            image = Image.open(dataset / "images" / split / image_name).convert("RGB")
            w, h = image.size
            for item, (x1, y1, x2, y2) in boxes:
                mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                box = (max(0, (x1 - mx) * w), max(0, (y1 - my) * h),
                       min(w, (x2 + mx) * w), min(h, (y2 + my) * h))  # fmt: skip
                if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                    continue  # too small to judge
                yield item, image.crop(box)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--label")
    args = p.parse_args(argv)

    from eval.accuracy.report import kb_streams
    from pipeline.fallback import Fallback

    streams = kb_streams()
    models = {name: Fallback(OUT / f"siglip2_image_{name}.pte", OUT / "text_embeddings.json")
              for name in ("fp32", "int8")}  # fmt: skip
    stats = {name: {"top1": 0, "top3": 0, "stream_top1": 0} for name in models}
    agree = n = 0
    per_item: dict[str, int] = {}
    for item, crop in crops(args.dataset):
        n += 1
        per_item[item] = per_item.get(item, 0) + 1
        tops = {}
        for name, fb in models.items():
            top = [s["item_id"] for s in fb.suggest(crop, k=3)]
            tops[name] = top[0]
            stats[name]["top1"] += top[0] == item
            stats[name]["top3"] += item in top
            stats[name]["stream_top1"] += streams[top[0]] == streams[item]
        agree += tops["fp32"] == tops["int8"]
    report = {
        "dataset": str(args.dataset),
        "crops": n,
        "items": dict(sorted(per_item.items())),
        **{name: {k: round(v / n, 3) for k, v in s.items()} for name, s in stats.items()},
        "fp32_int8_top1_agreement": round(agree / n, 3) if n else None,
    }
    print(json.dumps(report, indent=2))
    if args.label:
        out = RESULTS / f"fallback_{args.label}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
