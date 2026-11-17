"""Golden fixtures: the contract the Python pipeline and the Android app must both meet."""

import json
from pathlib import Path

import pytest

from tests.fixtures.golden.regen import expected_for

GOLDEN = sorted((Path(__file__).parent / "fixtures" / "golden").glob("*.json"))


def test_there_are_enough_fixtures():
    assert len(GOLDEN) >= 20


@pytest.mark.parametrize("path", GOLDEN, ids=[p.stem for p in GOLDEN])
def test_golden(kb, path):
    case = json.loads(path.read_text(encoding="utf-8"))
    assert case["id"] == path.stem
    assert expected_for(kb, case["input"]) == case["expected"]
