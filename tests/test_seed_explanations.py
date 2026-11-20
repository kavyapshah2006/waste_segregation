"""The shipped explanations must use the app's cache keys (ExplanationCache.key in Kotlin)."""

import json
from pathlib import Path

from models.llm.seed_explanations import cache_key, jobs

SEED = Path(__file__).resolve().parents[1] / "models/llm/explanations_seed.jsonl"


def test_cache_key_matches_the_app_format():
    # Kotlin: "v$VERSION|$kbVersion|$itemId|$attrs|$language", booleans sorted by name.
    assert cache_key("abc", "pet_bottle", {}, "en") == "v1|abc|pet_bottle||en"
    key = cache_key("abc", "milk_pouch", {"wet": False, "contaminated": True}, "hi")
    assert key == "v1|abc|milk_pouch|contaminated=true,wet=false|hi"
    assert cache_key("abc", "x", {"note": "dirty"}, "en") == "v1|abc|x||en"  # non-booleans dropped


def test_jobs_cover_each_item_and_each_condition_answer(kb):
    attrs = {"milk_pouch": {"contaminated": {}}}
    all_jobs = list(jobs(kb, attrs, ["en"]))
    assert len(all_jobs) == len(kb.item_ids) + 2
    assert ("milk_pouch", {"contaminated": True}, "en") in all_jobs
    assert ("milk_pouch", {"contaminated": False}, "en") in all_jobs


def test_shipped_seed_rows_are_checked_answers_for_this_kb(kb):
    if not SEED.exists():
        return
    rows = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    for row in rows:
        version, kb_version, item_id, attrs_text, language = row["key"].split("|")
        assert version == "v1" and kb_version == kb.version and item_id in kb.item_ids
        assert language in ("en", "hi")
        assert row["status"] in ("ok", "corrected")
        attrs = {}
        for pair in filter(None, attrs_text.split(",")):
            name, value = pair.split("=")
            attrs[name] = value == "true"
        # The KB decides the bin; the shipped answer must carry the KB's bin for that condition.
        assert row["output"]["bin"] == kb.resolve(item_id, attrs, language).bin_colour
