"""pipeline/corrections.py. tests/fixtures/corrections_cases.json is shared with the Kotlin
core (CorrectionsTest): both must write exactly the same lines."""

import json
from pathlib import Path

import pytest

from pipeline.corrections import Correction, main, parse_line, summarise, to_line

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "corrections_cases.json").read_text("utf-8")
)


def _correction(row: dict) -> Correction:
    return Correction(**row)


def test_lines_match_fixture():
    lines = [to_line(_correction(row)) for row in CASES["corrections"]]
    assert lines == CASES["lines"]


def test_lines_round_trip():
    for line, row in zip(CASES["lines"], CASES["corrections"], strict=True):
        assert parse_line(line) == _correction(row)


def test_summary():
    corrections = [parse_line(line) for line in CASES["lines"]]
    assert summarise(corrections) == CASES["summary"]


def test_unknown_items_flagged():
    corrections = [parse_line(line) for line in CASES["lines"]]
    known = {"glass_bottle", "kulhad"}
    assert summarise(corrections, known)["unknown_items"] == ["fruit_veg_peels"]


@pytest.mark.parametrize("row", CASES["invalid"])
def test_invalid_rejected(row):
    with pytest.raises(ValueError):
        _correction(row)


def test_cli(tmp_path, capsys):
    path = tmp_path / "corrections.jsonl"
    path.write_text("\n".join(CASES["lines"]) + "\n", encoding="utf-8")
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == CASES["summary"]
