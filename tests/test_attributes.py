"""pipeline/attributes.py. tests/fixtures/attribute_cases.json is shared with the Kotlin core
(AttributesTest)."""

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.attributes import AttributeChecker, decide

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "attribute_cases.json").read_text("utf-8")
)
CHECKER = AttributeChecker(FIXTURE["prompts"])


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["id"] for c in FIXTURE["cases"]])
def test_fixture(case):
    # The app's embedding is float32, as here.
    embedding = np.asarray(case["embedding"], dtype=np.float32)
    assert CHECKER.check(case["item_id"], embedding) == case["expected"]


def test_band_edges_ask_the_user():
    assert decide(0.01, -0.01, 0.01) is None
    assert decide(-0.01, -0.01, 0.01) is None
    with pytest.raises(ValueError):
        decide(0.0, 0.1, -0.1)
