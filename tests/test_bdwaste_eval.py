"""eval/bdwaste/evaluate.py scoring on hand-built rows."""

from eval.bdwaste.evaluate import CLASS_ITEMS, summarise

STREAMS = {
    "pet_bottle": "dry",
    "chips_packet": "dry",
    "charger_cable": "special_care",
    "small_electronics": "special_care",
    "fruit_veg_peels": "wet",
    "food_scraps": "wet",
    "plastic_toy": "dry",
}


def row(cls, det, top3):
    return {"class": cls, "det": det, "top3": top3}


def test_detector_verdict_first_then_fallback_top1():
    rows = [
        # detector sure and right
        row("Bottle", ["pet_bottle", 0.9], ["chips_packet", "pet_bottle", "plastic_toy"]),
        # detector below the gate: fallback top-1 is used (wrong item, right stream)
        row("Bottle", ["pet_bottle", 0.3], ["chips_packet", "pet_bottle", "plastic_toy"]),
        # wire: no detection, fallback right
        row("Wire", None, ["charger_cable", "small_electronics", "pet_bottle"]),
        # wire: fallback top-1 in the wrong bin, right item only in top 3
        row("Wire", None, ["pet_bottle", "charger_cable", "small_electronics"]),
        # peel: right item second, top-1 still wet
        row("Banana peel", None, ["food_scraps", "fruit_veg_peels", "pet_bottle"]),
        # stream-only class: any dry item is right on stream, never on item
        row("Pastic", None, ["plastic_toy", "pet_bottle", "chips_packet"]),
    ]
    r = summarise(rows, STREAMS, gate=0.5)
    bottle = r["per_class"]["Bottle"]
    assert bottle["detector_coverage"] == 0.5 and bottle["detector_item_acc"] == 1.0
    # pet_bottle is in SigLIP's top 3, so the first photo keeps the detector's verdict
    assert bottle["app_verdict_coverage"] == 0.5 and bottle["app_verdict_stream_acc"] == 1.0
    assert bottle["app_item_acc"] == 0.5 and bottle["app_stream_acc"] == 1.0
    assert bottle["fallback_top1_item"] == 0.0 and bottle["fallback_top3_item"] == 1.0
    wire = r["per_class"]["Wire"]
    assert wire["detector_coverage"] == 0.0 and wire["detector_item_acc"] is None
    assert wire["app_stream_acc"] == 0.5 and wire["fallback_top3_item"] == 1.0
    assert r["per_class"]["Banana peel"]["app_stream_acc"] == 1.0
    assert r["per_class"]["Pastic"]["app_stream_acc"] == 1.0
    assert r["per_class"]["Pastic"]["app_item_acc"] == 0.0
    assert r["by_stream"]["special_care"]["app_stream_acc"] == 0.5
    assert r["overall"]["images"] == 6
    # dry 3/3, special care 1/2, wet 1/1 -> macro (1 + .5 + 1) / 3
    assert r["macro_over_streams"]["app_stream_acc"] == round(2.5 / 3, 4)


def test_agreement_overrules_a_detector_that_siglip_does_not_back():
    rows = [
        # detector sure it is a carry bag; SigLIP sees peels: no verdict, top-1 suggestion used
        row(
            "Banana peel",
            ["plastic_carry_bag", 0.9],
            ["fruit_veg_peels", "food_scraps", "pet_bottle"],
        ),
    ]
    streams = STREAMS | {"plastic_carry_bag": "dry"}
    peel = summarise(rows, streams, gate=0.5)["per_class"]["Banana peel"]
    assert peel["detector_first_stream_acc"] == 0.0  # previous rule: dry bin
    assert peel["app_verdict_coverage"] == 0.0 and peel["app_verdict_stream_acc"] is None
    assert peel["app_stream_acc"] == 1.0 and peel["app_item_acc"] == 1.0


def test_every_mapped_class_has_one_stream():
    # Mixed-stream mappings would make "right bin" ambiguous.
    from eval.bdwaste.evaluate import kb_streams

    streams = kb_streams()
    for cls, items in CLASS_ITEMS.items():
        assert items <= set(streams), cls
        assert len({streams[i] for i in items}) <= 1, cls
