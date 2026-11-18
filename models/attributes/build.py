"""Build attribute_prompts.json: item-specific text embeddings for the dirty / broken check.

    uv run --group finetune python models/attributes/build.py   # after models/fallback/build.py

Uses the SigLIP 2 text encoder already downloaded by models/fallback/build.py (same model and
revision as the image encoder that ships). Only items whose KB advice changes with condition
(their `condition_rules`) get prompts; all other items skip the check.

Thresholds: `low`/`high` per attribute (see pipeline/attributes.py). Between them the app asks
the user. Change them only from a calibration run (models/attributes/check_attributes.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.fallback.build import MODEL, REVISION  # noqa: E402
from models.fallback.build import OUT as FALLBACK_OUT  # noqa: E402

OUT = REPO / "models" / "attributes" / "out"

PROMPTS = {
    "broken": {
        "positive": ["a photo of a broken {}", "a photo of a shattered {} in pieces"],
        "negative": ["a photo of an intact {}", "a photo of a whole, undamaged {}"],
    },
    "contaminated": {
        "positive": [
            "a photo of a dirty {} with food waste on it",
            "a photo of a greasy, food-stained {}",
        ],
        "negative": ["a photo of a clean {}", "a photo of a clean, empty {}"],
    },
}

# "Ask the user" bands. broken: the band that kept decided answers 90% right on TACO glass
# crops (eval/results/attributes_taco_glass.json; it decides only ~16% of crops, the rest are
# asked). contaminated: no labelled data yet, so the band covers every possible margin and the
# app always asks. Recalibrate both on the India photo set (checklist steps 19, 25).
THRESHOLDS = {
    "broken": {"low": -0.0196, "high": 0.0304},
    "contaminated": {"low": -2.0, "high": 2.0},
}


def item_name(item: dict) -> str:
    name = item["names"]["en"]
    # "Glass bottle or jar" -> "glass bottle or jar", but keep "CFL bulb" and "PET plastic".
    return name[0].lower() + name[1:] if name[1:2].islower() else name


@torch.no_grad()
def embed(model, tokenizer, texts: list[str]) -> list[float]:
    batch = tokenizer(texts, padding="max_length", max_length=64, truncation=True,
                      return_tensors="pt")  # fmt: skip
    out = model.get_text_features(**batch)
    emb = out.pooler_output if hasattr(out, "pooler_output") else out
    mean = torch.nn.functional.normalize(emb, dim=-1).mean(0)
    return [round(float(x), 6) for x in torch.nn.functional.normalize(mean, dim=0)]


def build_prompts(model, tokenizer, kb) -> dict:
    items = {}
    for item_id in kb.item_ids:
        item = kb.item(item_id)
        attributes = sorted({k for rule in item.get("condition_rules", []) for k in rule["if"]})
        if not attributes:
            continue
        name = item_name(item)
        items[item_id] = {}
        for attribute in attributes:
            spec = PROMPTS[attribute]
            items[item_id][attribute] = {
                "positive_texts": [t.format(name) for t in spec["positive"]],
                "negative_texts": [t.format(name) for t in spec["negative"]],
                "positive": embed(model, tokenizer, [t.format(name) for t in spec["positive"]]),
                "negative": embed(model, tokenizer, [t.format(name) for t in spec["negative"]]),
            }
    return {
        "model": MODEL,
        "revision": REVISION,
        "kb_version": kb.version,
        "thresholds": THRESHOLDS,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    from kb.build_kb import build
    from pipeline.kb import KnowledgeBase

    local = FALLBACK_OUT / "hf"
    if not (local / "model.safetensors").exists():
        print("run models/fallback/build.py first (it downloads the SigLIP 2 weights)")
        return 2
    model = AutoModel.from_pretrained(local, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(local)
    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / "kb.sqlite"
    build(out=db)
    data = build_prompts(model, tokenizer, KnowledgeBase(db))
    out = args.out / "attribute_prompts.json"
    out.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}: {sorted(data['items'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
