"""Check the zero-shot "broken" decision on labelled crops, and suggest thresholds.

    uv run --group finetune python models/attributes/check_attributes.py \\
        --dataset data/processed/waste --label taco_glass

TACO has no dirty/broken flags, but it labels broken glass separately from glass bottles. So:
every `glass_bottle` crop counts as intact and every `broken_glass` crop as broken, and both
are scored with the glass bottle's broken/intact prompts through the int8 encoder that ships.
This is a proxy (TACO's broken glass is mostly shards, not a cracked bottle), so treat the
result as a sanity check. "Dirty" cannot be checked until the India photo set has labels.

Writes eval/results/attributes_<label>.json: ROC AUC of the margin, results with the current
thresholds, and the widest symmetric "ask the user" band that keeps the decided answers at
least 90% correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.attributes.build import OUT  # noqa: E402

RESULTS = REPO / "eval" / "results"
ENCODER = REPO / "models" / "fallback" / "out" / "siglip2_image_int8.pte"


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random broken crop scores above a random intact one (ties half)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def band_result(scored: list[tuple[bool, float]], low: float, high: float) -> dict:
    from pipeline.attributes import decide

    decided = [(truth, decide(m, low, high)) for truth, m in scored]
    answered = [(t, d) for t, d in decided if d is not None]
    correct = sum(t == d for t, d in answered)
    return {
        "low": low,
        "high": high,
        "answered_rate": round(len(answered) / len(scored), 3) if scored else None,
        "accuracy_when_answered": round(correct / len(answered), 3) if answered else None,
    }


def suggest_band(scored: list[tuple[bool, float]], target: float = 0.9) -> dict | None:
    """Narrowest symmetric band around the best split point that reaches `target` accuracy."""
    margins = sorted({round(m, 4) for _, m in scored})
    best = None
    for centre in margins:
        for half in [x / 1000 for x in range(0, 201, 5)]:
            r = band_result(scored, round(centre - half, 4), round(centre + half, 4))
            if r["accuracy_when_answered"] is not None and r["accuracy_when_answered"] >= target:
                if best is None or r["answered_rate"] > best["answered_rate"]:
                    best = r
                break
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args(argv)

    from models.fallback.check_crops import crops
    from pipeline.attributes import AttributeChecker
    from pipeline.fallback import Fallback

    prompts = json.loads((OUT / "attribute_prompts.json").read_text(encoding="utf-8"))
    checker = AttributeChecker(prompts)
    encoder = Fallback(ENCODER, REPO / "models" / "fallback" / "out" / "text_embeddings.json")
    truth_of = {"glass_bottle": False, "broken_glass": True}
    scored: list[tuple[bool, float]] = []
    for item, crop in crops(args.dataset):
        if item not in truth_of:
            continue
        result = checker.check("glass_bottle", encoder.embed(crop))["broken"]
        scored.append((truth_of[item], result["margin"]))
    pos = [m for t, m in scored if t]
    neg = [m for t, m in scored if not t]
    current = prompts["thresholds"]["broken"]
    report = {
        "attribute": "broken",
        "proxy": "TACO broken_glass (broken) vs glass_bottle (intact), glass bottle prompts",
        "encoder": "siglip2_image_int8.pte",
        "kb_version": prompts["kb_version"],
        "crops": {"broken": len(pos), "intact": len(neg)},
        "auc": round(auc(pos, neg), 3) if pos and neg else None,
        "mean_margin": {
            "broken": round(sum(pos) / len(pos), 4) if pos else None,
            "intact": round(sum(neg) / len(neg), 4) if neg else None,
        },
        "current_band": band_result(scored, current["low"], current["high"]),
        "suggested_band_90pct": suggest_band(scored),
    }
    print(json.dumps(report, indent=2))
    out = RESULTS / f"attributes_{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
