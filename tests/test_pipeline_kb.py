import json

from kb.build_kb import build
from pipeline.kb import KnowledgeBase


def test_search_exact_alias_in_hindi_and_english(kb):
    assert kb.search("पानी की बोतल") == ["pet_bottle"]
    assert kb.search("Pizza Box")[0] == "cardboard_box"
    assert kb.search("  ") == []


def test_search_partial_match(kb):
    assert "aa_battery" in kb.search("remote battery please")


def test_detector_class_maps_to_item(kb):
    assert kb.item_for_class("cfl_bulb") == "cfl_bulb"
    assert kb.item_for_class("not_a_class") is None


def test_default_resolution_uses_city_colour(kb):
    v = kb.resolve("aa_battery")
    assert (v.stream, v.bin_colour, v.drop_off) == ("special_care", "black", "e_waste_centre")
    assert v.drop_off_text and v.hazard
    assert v.applied == ()


def test_attributes_that_do_not_match_a_rule_change_nothing(kb):
    assert kb.resolve("cardboard_box", {"broken": True}) == kb.resolve("cardboard_box")


def test_stream_change_clears_drop_off_and_replaces_steps(kb):
    v = kb.resolve("cardboard_box", {"contaminated": True})
    assert v.stream == "wet" and v.bin_colour == "green"
    assert v.drop_off is None and v.drop_off_text is None
    assert not v.recyclable and v.compostable
    assert v.steps[0] == "Tear off the greasy or food-soiled parts"
    assert v.notes


def test_rule_can_clear_drop_off_without_changing_stream(kb):
    whole = kb.resolve("glass_bottle")
    broken = kb.resolve("glass_bottle", {"broken": True})
    assert whole.drop_off == "scrap_dealer"
    assert broken.stream == "dry" and broken.drop_off is None and broken.hazard


def test_broken_mercury_items_get_clean_up_steps(kb):
    for item_id in ("cfl_bulb", "mercury_thermometer"):
        whole = kb.resolve(item_id)
        broken = kb.resolve(item_id, {"broken": True})
        assert (broken.stream, broken.bin_colour) == (whole.stream, whole.bin_colour)
        assert broken.drop_off == "hazardous_collection"
        assert broken.steps != whole.steps
        assert any("vacuum" in step for step in broken.steps)
        assert broken.applied == ("rule:broken=true",)
        hi = kb.resolve(item_id, {"broken": True}, language="hi")
        assert len(hi.steps) == len(broken.steps) and not hi.missing_translation


def test_extra_steps_are_prepended(kb):
    v = kb.resolve("pet_bottle", {"contaminated": True})
    assert v.steps[0] == "Rinse out food or liquid first"
    assert v.steps[1:] == kb.resolve("pet_bottle").steps


def test_hindi_text(kb):
    v = kb.resolve("cfl_bulb", language="hi")
    assert v.name == "सीएफ़एल बल्ब या ट्यूबलाइट"
    assert v.bin_colour_name == "काला"
    assert not v.missing_translation


def test_missing_language_falls_back_to_english(kb):
    v = kb.resolve("pet_bottle", language="ta")
    assert v.name == "PET plastic bottle"
    assert v.missing_translation


def test_city_override_beats_national_default(kb_source_copy, tmp_path):
    city = json.loads((kb_source_copy / "cities" / "default_in.json").read_text(encoding="utf-8"))
    city["id"] = "testville"
    city["bin_colours"]["dry"] = "white"
    city["overrides"] = {
        "kulhad": {
            "stream": "wet",
            "note": {"en": "Testville composts clay.", "hi": "टेस्टविल मिट्टी की खाद बनाता है।"},
        }
    }
    (kb_source_copy / "cities" / "testville.json").write_text(
        json.dumps(city, ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "kb.sqlite"
    build(kb_dir=kb_source_copy, out=out)
    kb = KnowledgeBase(out)

    kulhad = kb.resolve("kulhad", city_id="testville")
    assert (kulhad.stream, kulhad.bin_colour) == ("wet", "green")
    assert kulhad.notes == ("Testville composts clay.",)
    assert kulhad.applied == ("city:testville",)
    assert kb.resolve("pet_bottle", city_id="testville").bin_colour == "white"
