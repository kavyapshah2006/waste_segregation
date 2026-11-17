"""Shared checker cases (tests/fixtures/check_cases.json), also run by the Kotlin core tests."""

import json
from pathlib import Path

import pytest

from pipeline.assistant import is_on_topic
from pipeline.verify import (
    bins_mentioned,
    compostable_claims,
    extract_json,
    negates_step,
    negates_verdict,
    recyclable_claims,
    unsafe_instruction,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "check_cases.json").read_text(encoding="utf-8")
)


def _params(section: str):
    return pytest.mark.parametrize(
        "case", CASES[section], ids=lambda c: repr(c.get("text") or c.get("question"))[:50]
    )


def _verdict(kb, case):
    return kb.resolve(case["item"], case.get("attributes", {}), case.get("language", "en"))


@_params("extract_json")
def test_extract_json(case):
    assert extract_json(case["text"]) == case["expected"]


@_params("bins_mentioned")
def test_bins_mentioned(case):
    assert sorted(bins_mentioned(case["text"])) == case["expected"]


@_params("negates_verdict")
def test_negates_verdict(kb, case):
    assert negates_verdict(case["text"], _verdict(kb, case)) is case["expected"]


@_params("recyclable_claims")
def test_recyclable_claims(case):
    assert sorted(recyclable_claims(case["text"])) == case["expected"]


@_params("compostable_claims")
def test_compostable_claims(case):
    assert sorted(compostable_claims(case["text"])) == case["expected"]


@_params("negates_step")
def test_negates_step(kb, case):
    assert negates_step(case["text"], _verdict(kb, case)) is case["expected"]


@_params("unsafe_instruction")
def test_unsafe_instruction(kb, case):
    assert unsafe_instruction(case["text"], _verdict(kb, case)) is case["expected"]


@_params("is_on_topic")
def test_is_on_topic(kb, case):
    assert is_on_topic(case["question"], kb.names_of(case["item"])) is case["expected"]


@_params("search")
def test_search(kb, case):
    assert kb.search(case["text"]) == case["expected"]
