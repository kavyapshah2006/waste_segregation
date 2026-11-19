"""Unknown-item fallback: suggest KB items from an image with SigLIP 2 (FuturePlan WP7).

Used only when the detector stays unsure (pipeline/stability.py state "unsure" or
"empty"). It returns ranked suggestions for the user to confirm ("Is this a ...?"), never a
verdict: zero-shot image-text matching is much less reliable than the trained detector.

Built by models/fallback/build.py: an image-encoder .pte and text_embeddings.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 224


def preprocess(image: Image.Image) -> np.ndarray:
    """SigLIP preprocessing: RGB, 224x224 bilinear (no padding), [0,1], then (x-0.5)/0.5."""
    img = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None].copy()


class Fallback:
    def __init__(self, model_path: Path | str, text_embeddings: Path | str):
        import executorch.kernels.quantized  # noqa: F401  (kernels the int8 model needs)
        from executorch.runtime import Runtime

        self._method = Runtime.get().load_program(str(model_path)).load_method("forward")
        data = json.loads(Path(text_embeddings).read_text(encoding="utf-8"))
        self.item_ids: list[str] = data["item_ids"]
        self.text = np.asarray(data["embeddings"], dtype=np.float32)
        self.scale, self.bias = data["logit_scale"], data["logit_bias"]

    def embed(self, image: Image.Image) -> np.ndarray:
        import torch

        out = self._method.execute([torch.from_numpy(preprocess(image))])[0]
        return out.numpy()[0]

    def suggest(self, image: Image.Image, k: int = 3) -> list[dict]:
        return rank(self.text, self.item_ids, self.embed(image), self.scale, self.bias, k)


def confirms(detector_item: str, suggestions: list[dict]) -> bool:
    """Agreement rule: a detector verdict stands only when SigLIP also ranks that item among
    its suggestions (top 3); otherwise the app shows the suggestions instead of a verdict.

    Measured on 2,371 BDWaste photos (eval/bdwaste/evaluate.py): the TACO-trained detector
    confidently calls fruit peels and fish waste "carry bag", so detector-first sends wet waste
    to the dry bin 45% of the time; with this rule the right bin rises from 75% to 92%.
    Shared with the Kotlin core (FallbackRank.confirms) through tests/fixtures/agreement_cases.json.
    """
    return any(s["item_id"] == detector_item for s in suggestions)


def rank(
    text: np.ndarray, item_ids: list[str], image_emb: np.ndarray, scale: float, bias: float, k: int
) -> list[dict]:
    """Top-k items by cosine similarity (ties: KB order), with SigLIP's sigmoid score.
    Shared with the Kotlin core (FallbackRank) through tests/fixtures/fallback_rank.json."""
    cos = [float(np.dot(row.astype(np.float64), image_emb.astype(np.float64))) for row in text]
    order = sorted(range(len(cos)), key=lambda i: (-cos[i], i))[:k]
    return [
        {
            "item_id": item_ids[i],
            "cosine": round(cos[i], 4),
            "score": round(1 / (1 + math.exp(-(scale * cos[i] + bias))), 4),
        }
        for i in order
    ]
