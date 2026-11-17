import json
from pathlib import Path

import pytest

from pipeline.impact import Scan, current_streak, dedupe, local_day
from tests.fixtures.impact.regen import expected_for

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "impact" / "cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_impact_fixture(kb, case):
    assert expected_for(kb, case) == case["expected"]


def test_dedupe_measures_from_the_last_counted_scan():
    scans = [Scan(0, "a"), Scan(20, "a"), Scan(35, "a"), Scan(50, "a"), Scan(66, "a")]
    counted, dropped = dedupe(scans)
    assert [s.time_s for s in counted] == [0, 35, 66] and dropped == 2


def test_local_day_and_streak():
    ist = 330
    assert local_day(1794960000, ist) == local_day(1794960000 + 3600, ist)
    assert current_streak({10, 11, 12}, 12) == 3
    assert current_streak({10, 11, 12}, 13) == 3  # today not scanned yet
    assert current_streak({10, 11, 12}, 14) == 0
    assert current_streak(set(), 5) == 0


def test_summary_fixture_values_are_plausible():
    mixed = next(c for c in CASES if c["id"] == "summary_mixed_streams_and_diversion")
    exp = mixed["expected"]
    assert exp["scans"] == 6
    assert exp["by_stream"] == {"wet": 1, "dry": 3, "sanitary": 1, "special_care": 1}
    assert exp["diverted_items"] == 3  # greasy cardboard, battery, broken bottle
    assert exp["top_items"][0] == ["chips_packet", 2]
