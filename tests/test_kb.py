"""Knowledge base sources are valid and compile into the expected SQLite layout."""

import copy
import json
import sqlite3

import pytest

from kb.build_kb import KB_DIR, build, load_source, unreviewed, validate

STREAMS = {"wet", "dry", "sanitary", "special_care"}


@pytest.fixture(scope="module")
def src():
    return load_source()


def test_sources_are_valid(src):
    assert validate(src) == []


def test_covers_every_stream_and_hard_cases(src):
    assert len(src.items) >= 40
    assert {item["stream"] for item in src.items.values()} == STREAMS
    for hard_case in (
        "chips_packet",
        "tetra_pak",
        "blister_pack",
        "cardboard_box",
        "broken_glass",
        "cfl_bulb",
        "aa_battery",
        "sanitary_pad",
        "syringe",
        "kulhad",
    ):
        assert hard_case in src.items


def test_default_city_uses_swm_2026_colours(src):
    assert src.cities["default_in"]["bin_colours"] == {
        "wet": "green",
        "dry": "blue",
        "sanitary": "red",
        "special_care": "black",
    }


def test_every_item_has_english_and_hindi_text(src):
    for item in src.items.values():
        for field in ("names", "prep_steps", "impact"):
            assert item[field]["en"] and item[field]["hi"], (item["id"], field)


def test_hindi_is_flagged_unreviewed_until_checked(src):
    assert set(unreviewed(src)) <= {"hi", "ta", "te", "bn", "mr"}


def test_validate_reports_bad_stream_and_mismatched_id(src):
    bad = copy.deepcopy(src)
    bad.items["pet_bottle"]["stream"] = "plastic"
    bad.items["pet_bottle"]["id"] = "bottle"
    errors = validate(bad)
    assert any("stream" in e for e in errors)
    assert any("does not match file name" in e for e in errors)


def test_validate_reports_duplicate_detector_class(src):
    bad = copy.deepcopy(src)
    bad.items["tin_can"]["detector_classes"] = ["aluminium_can"]
    assert any("aluminium_can" in e for e in validate(bad))


def test_validate_reports_missing_drop_off_text(src):
    bad = copy.deepcopy(src)
    del bad.cities["default_in"]["drop_off_info"]["scrap_dealer"]
    assert any("scrap_dealer" in e for e in validate(bad))


def test_build_writes_sqlite(tmp_path):
    out = tmp_path / "kb.sqlite"
    src = build(out=out)
    con = sqlite3.connect(out)
    try:
        assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == len(src.items)
        (item_json,) = con.execute("SELECT json FROM items WHERE id = 'pet_bottle'").fetchone()
        assert json.loads(item_json)["stream"] == "dry"
        hits = con.execute(
            "SELECT item_id FROM aliases WHERE alias = ?", ("पानी की बोतल",)
        ).fetchall()
        assert hits == [("pet_bottle",)]
        assert con.execute(
            "SELECT item_id FROM detector_classes WHERE class = 'cfl_bulb'"
        ).fetchone() == ("cfl_bulb",)
        version = con.execute("SELECT value FROM meta WHERE key = 'kb_version'").fetchone()[0]
        assert len(version) == 16
    finally:
        con.close()


def test_build_refuses_invalid_sources(tmp_path):
    kb_copy = tmp_path / "kb"
    for sub in ("items", "cities"):
        (kb_copy / sub).mkdir(parents=True)
        for f in (KB_DIR / sub).glob("*.json"):
            (kb_copy / sub / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("schema.json", "city_schema.json", "strings.json"):
        (kb_copy / name).write_text((KB_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    broken = json.loads((kb_copy / "items" / "diaper.json").read_text(encoding="utf-8"))
    del broken["names"]["hi"]
    (kb_copy / "items" / "diaper.json").write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="diaper"):
        build(kb_dir=kb_copy, out=tmp_path / "kb.sqlite")
