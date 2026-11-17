"""Unknown-item fallback (pipeline/fallback.py): preprocessing and ranking. The .pte itself is
built and checked by models/fallback/build.py (needs a 1.5 GB download)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.fallback import SIZE, preprocess, rank

PATH = Path(__file__).parent / "fixtures" / "fallback_rank.json"
DATA = json.loads(PATH.read_text(encoding="utf-8"))


def _rank(case):
    return rank(np.asarray(DATA["text"], dtype=np.float32), DATA["item_ids"],
                np.asarray(case["image"], dtype=np.float32), DATA["logit_scale"],
                DATA["logit_bias"], case["k"])  # fmt: skip


@pytest.mark.parametrize("i", range(len(DATA["cases"])))
def test_rank_fixture(i):
    assert _rank(DATA["cases"][i]) == DATA["cases"][i]["expected"]


def test_regen():
    if not os.environ.get("REGEN"):
        pytest.skip("set REGEN=1 to rewrite expected values")
    for case in DATA["cases"]:
        case["expected"] = _rank(case)
    PATH.write_text(json.dumps(DATA, indent=2) + "\n", encoding="utf-8")


def test_rank_orders_by_similarity_and_breaks_ties_by_kb_order():
    first = DATA["cases"][0]["expected"]
    # cosines: tin 0.18, newspaper 0.6*0.15+0.8*0.10 = 0.17, battery 0.15, bottle 0.10
    assert [s["item_id"] for s in first] == ["tin_can", "newspaper", "aa_battery"]
    assert first[0]["score"] > 0.9 > 0.6 > first[2]["score"] > 0.4  # sigmoid not saturated
    tie = DATA["cases"][2]["expected"]
    # newspaper (0.168) first, then battery and bottle tie at 0.12: KB order decides
    assert [s["item_id"] for s in tie][1:3] == ["aa_battery", "pet_bottle"]


def test_preprocess_matches_siglip_processing():
    white = preprocess(Image.new("RGB", (640, 480), (255, 255, 255)))
    assert white.shape == (1, 3, SIZE, SIZE) and white.dtype == np.float32
    assert np.allclose(white, 1.0)
    black = preprocess(Image.new("L", (100, 300), 0))  # grey input becomes RGB
    assert np.allclose(black, -1.0)
