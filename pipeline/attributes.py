"""Is this item dirty or broken? Zero-shot, from the SigLIP 2 embedding (FuturePlan WP7 step 3).

Only ten KB items change their advice with condition (a dirty milk pouch goes to a different
bin; a broken CFL bulb is a hazard). For those, the app compares the crop's SigLIP 2 image
embedding (the same encoder as the unknown-item fallback, so no extra model ships) with two
item-specific text embeddings: "a dirty milk pouch" against "a clean milk pouch".

    margin = cos(image, positive) - cos(image, negative)

Above `high` the attribute is set, below `low` it is not, and in between the answer is None:
the app asks the user ("Is it broken?") instead of guessing. Thresholds start as a narrow band
around 0 and are calibrated on labelled crops (models/attributes/check_attributes.py).

Built by models/attributes/build.py (attribute_prompts.json). The Kotlin port is
`AttributeChecker` in android/core (shared cases: tests/fixtures/attribute_cases.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def margin(image: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """cos(image, positive) - cos(image, negative); all three are L2-normalised."""
    img = np.asarray(image, dtype=np.float64)
    pos = float(np.dot(img, np.asarray(positive, dtype=np.float64)))
    neg = float(np.dot(img, np.asarray(negative, dtype=np.float64)))
    return pos - neg


def decide(value: float, low: float, high: float) -> bool | None:
    """True above high, False below low, None (ask the user) in between."""
    if low > high:
        raise ValueError(f"low {low} > high {high}")
    if value > high:
        return True
    if value < low:
        return False
    return None


class AttributeChecker:
    def __init__(self, prompts_json: Path | str | dict):
        data = (
            prompts_json
            if isinstance(prompts_json, dict)
            else json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        )
        self.thresholds: dict[str, dict] = data["thresholds"]
        self.items: dict[str, dict] = data["items"]

    def attributes_for(self, item_id: str) -> list[str]:
        return sorted(self.items.get(item_id, {}))

    def check(self, item_id: str, image_embedding: np.ndarray) -> dict[str, dict]:
        """{attribute: {"value": True/False/None, "margin": float}} for this item's attributes."""
        out = {}
        for attribute in self.attributes_for(item_id):
            spec = self.items[item_id][attribute]
            m = margin(image_embedding, spec["positive"], spec["negative"])
            t = self.thresholds[attribute]
            out[attribute] = {"value": decide(m, t["low"], t["high"]), "margin": round(m, 4)}
        return out
