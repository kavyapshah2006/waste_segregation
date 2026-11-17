"""Impact summary of a user's scans (FuturePlan WP10). Reference for the Kotlin app.

Everything here is an estimate built from KB fields, and the app must label it so:
weights are each item's typical weight (`typical_weight_g`), not a measurement. No CO2 or
money figures: the KB has no sourced factors for them.

"Diverted" means kept out of mixed landfill waste: the verdict is recyclable, compostable,
or has a special collection drop-off (e-waste, hazardous, pharmacy take-back).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pipeline.kb import DEFAULT_CITY, KnowledgeBase

DEDUPE_SECONDS = 30
STREAMS = ("wet", "dry", "sanitary", "special_care")
SPECIAL_DROP_OFFS = {"e_waste_centre", "pharmacy_takeback", "hazardous_collection"}
TOP_ITEMS = 5


@dataclass(frozen=True)
class Scan:
    time_s: float  # Unix time, seconds
    item_id: str
    attributes: dict = field(default_factory=dict)


def dedupe(scans: list[Scan]) -> tuple[list[Scan], int]:
    """Counted scans in time order, and how many repeats were dropped. A scan is a repeat if
    the same item was counted less than DEDUPE_SECONDS earlier (a camera seeing the same
    item again, or a double tap)."""
    counted: list[Scan] = []
    last: dict[str, float] = {}
    for scan in sorted(scans, key=lambda s: (s.time_s, s.item_id)):
        prev = last.get(scan.item_id)
        if prev is not None and scan.time_s - prev < DEDUPE_SECONDS:
            continue
        counted.append(scan)
        last[scan.item_id] = scan.time_s
    return counted, len(scans) - len(counted)


def local_day(time_s: float, tz_offset_minutes: int) -> int:
    """Days since 1970-01-01 in the user's local time."""
    return int((time_s + tz_offset_minutes * 60) // 86400)


def current_streak(active_days: set[int], today: int) -> int:
    """Consecutive active days ending today, or yesterday if today has no scan yet."""
    day = today if today in active_days else today - 1
    streak = 0
    while day in active_days:
        streak += 1
        day -= 1
    return streak


def summary(
    kb: KnowledgeBase,
    scans: list[Scan],
    now_s: float,
    tz_offset_minutes: int = 330,  # India
    city_id: str = DEFAULT_CITY,
) -> dict:
    counted, duplicates = dedupe(scans)
    by_stream = Counter({s: 0 for s in STREAMS})
    items = Counter()
    diverted_g = 0.0
    diverted = 0
    needed_cleaning = 0
    for scan in counted:
        verdict = kb.resolve(scan.item_id, scan.attributes, "en", city_id)
        by_stream[verdict.stream] += 1
        items[scan.item_id] += 1
        if scan.attributes.get("contaminated"):
            needed_cleaning += 1
        if verdict.recyclable or verdict.compostable or verdict.drop_off in SPECIAL_DROP_OFFS:
            diverted += 1
            diverted_g += kb.item(scan.item_id)["typical_weight_g"]
    days = {local_day(s.time_s, tz_offset_minutes) for s in counted}
    n = len(counted)
    return {
        "scans": n,
        "duplicates_ignored": duplicates,
        "by_stream": {s: by_stream[s] for s in STREAMS},
        "top_items": [[i, c] for i, c in sorted(items.items(), key=lambda x: (-x[1], x[0]))][
            :TOP_ITEMS
        ],
        "diverted_items": diverted,
        "diverted_kg_estimate": round(diverted_g / 1000, 3),
        "needed_cleaning_rate": round(needed_cleaning / n, 3) if n else None,
        "active_days": len(days),
        "current_streak_days": current_streak(days, local_day(now_s, tz_offset_minutes)),
    }
