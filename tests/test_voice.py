"""Voice transcript -> KB item matching (pipeline/voice.py). Cases in
tests/fixtures/voice_cases.json are shared with the Kotlin core."""

import json
import os
from pathlib import Path

import pytest

from pipeline.voice import key, match, romanise, similarity, window_similarity

PATH = Path(__file__).parent / "fixtures" / "voice_cases.json"
DATA = json.loads(PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", DATA["romanise"], ids=lambda c: c["text"])
def test_romanise_fixture(case):
    assert [romanise(case["text"]), key(case["text"])] == case["expected"]


@pytest.mark.parametrize("case", DATA["match"], ids=lambda c: c["transcript"] or "silence")
def test_match_fixture(kb, case):
    assert match(kb, case["transcript"]) == case["expected"]


def test_regen(kb):
    if not os.environ.get("REGEN"):
        pytest.skip("set REGEN=1 to rewrite expected values")
    for case in DATA["romanise"]:
        case["expected"] = [romanise(case["text"]), key(case["text"])]
    for case in DATA["match"]:
        case["expected"] = match(kb, case["transcript"])
    PATH.write_text(json.dumps(DATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_expected_matches_read_right():
    top = {c["transcript"]: (c["expected"][0]["item_id"] if c["expected"] else None)
           for c in DATA["match"]}  # fmt: skip
    assert top["gatte ka dibba"] == "cardboard_box"
    assert top["eliuminiam ka ken"] == "aluminium_can"
    assert top["दबा की खाली पति"] == "blister_pack"
    assert top["मेरे पास पानी की बोतल है इसे कहाँ डालूँ"] == "pet_bottle"
    assert top[""] is None
    no_item = next(c for c in DATA["match"] if c["transcript"] == "यह किस डिब्बे में जाएगा")
    assert all(m["confirm"] for m in no_item["expected"])  # nothing confident


def test_similarity_basics():
    assert similarity("abc", "abc") == 1.0
    assert similarity("", "") == 1.0
    assert similarity("abcd", "abce") == 0.75
    assert window_similarity("xxbeteriixx", "beterii") == 1.0
    assert window_similarity("xxpanixx", "pani") == similarity("xxpanixx", "pani")  # short name


def test_schwa_deletion():
    assert romanise("बैटरी") == "baitrii"
    assert romanise("बोतल") == "botal"  # final consonant: no vowel sign follows
    assert romanise("खाना") == "khaanaa"


def test_no_confident_wrong_item_on_regression_transcripts():
    by = {c["transcript"]: c["expected"] for c in DATA["match"]}
    battery = by["Purani Baitri ke kak"]
    assert battery[0]["item_id"] == "aa_battery"
    water_bottle = by["मेरे पास पानी की बोतल है इसे कहाँ डालूँ"]
    assert water_bottle[0]["item_id"] == "pet_bottle"
    assert all(m["item_id"] != "plastic_carry_bag" or m["confirm"] for m in water_bottle)
