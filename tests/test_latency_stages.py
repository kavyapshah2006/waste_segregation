"""eval/latency/stages.py. Cases in tests/fixtures/latency_cases.json are shared with the
Kotlin core (LatencyTest), so the phone overlay and eval/REPORT.md agree."""

import json
from pathlib import Path

import pytest

from eval.latency.stages import percentile, summarise

CASES = json.loads((Path(__file__).parent / "fixtures" / "latency_cases.json").read_text("utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture(case):
    summary = summarise(case["samples"])
    assert summary == case["expected"]
    if "expected_order" in case:
        assert list(summary) == case["expected_order"]


def test_percentile_edges():
    assert percentile([], 0.5) is None
    assert percentile([3.0, 1.0], 1.0) == 3.0
    with pytest.raises(ValueError):
        percentile([1.0], 0)


def test_negative_duration_rejected():
    with pytest.raises(ValueError):
        summarise({"detect": [1.0, -0.1]})
