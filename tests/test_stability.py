"""Frame voting (pipeline/stability.py). Cases in tests/fixtures/stability_cases.json are
shared with the Kotlin core; regenerate `expected` with REGEN=1 and review the diff."""

import json
import os
from pathlib import Path

import pytest

from pipeline.stability import FrameVoter, frame_vote

PATH = Path(__file__).parent / "fixtures" / "stability_cases.json"
CASES = json.loads(PATH.read_text(encoding="utf-8"))


def run(case) -> list[dict]:
    voter = FrameVoter(**case["params"])
    return [voter.push(frame) for frame in case["frames"]]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_stability_fixture(case):
    assert run(case) == case["expected"]


def test_regen():
    if not os.environ.get("REGEN"):
        pytest.skip("set REGEN=1 to rewrite expected values")
    for case in CASES:
        case["expected"] = run(case)
    PATH.write_text(json.dumps(CASES, indent=2) + "\n", encoding="utf-8")


def test_expected_states_read_right():
    by_id = {c["id"]: [s["state"] for s in c["expected"]] for c in CASES}
    assert by_id["steady_item_becomes_stable_on_the_third_vote"] == ["unsure", "unsure", "stable"]
    assert by_id["below_the_gate_never_counts"] == ["unsure", "unsure", "unsure", "unsure"]
    assert by_id["old_votes_leave_the_window"][-1] == "empty"
    switching = next(c for c in CASES if c["id"] == "switching_items_needs_a_new_majority")
    # The new item takes over only once it has 3 of the last 5 votes (third tin frame).
    assert [s["item_id"] for s in switching["expected"]][2:] == [
        "pet_bottle", "pet_bottle", "pet_bottle", "tin_can"
    ]  # fmt: skip


def test_frame_vote_and_bad_params():
    assert frame_vote([]).detected is False
    assert frame_vote([{"item_id": "x", "confidence": 0.4}], gate=0.5).item_id is None
    with pytest.raises(ValueError):
        FrameVoter(window=3, min_votes=4)
