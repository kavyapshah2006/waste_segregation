import json
from pathlib import Path

import pytest

from data.scripts.prepare import load_taxonomy
from kb.build_kb import load_source

TACO_CATEGORIES = json.loads(
    (Path(__file__).parent / "fixtures" / "taco_categories.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


def test_classes_are_exactly_the_kb_detector_classes(taxonomy):
    kb_classes = {c for item in load_source().items.values() for c in item["detector_classes"]}
    assert len(taxonomy["classes"]) == len(set(taxonomy["classes"]))
    assert set(taxonomy["classes"]) == kb_classes


@pytest.mark.parametrize("dataset", ["taco", "roboflow_ewaste"])
def test_map_targets_are_classes_and_map_and_drop_do_not_overlap(taxonomy, dataset):
    spec = taxonomy["datasets"][dataset]
    assert set(spec["map"].values()) <= set(taxonomy["classes"])
    assert not set(spec["map"]) & set(spec["drop"])


def test_every_taco_label_is_mapped_or_dropped(taxonomy):
    spec = taxonomy["datasets"]["taco"]
    assert sorted(list(spec["map"]) + spec["drop"]) == TACO_CATEGORIES


def test_taco_annotations_url_is_pinned_to_a_commit(taxonomy):
    url = taxonomy["datasets"]["taco"]["annotations"]
    assert "/master/" not in url and len(url.split("/")[5]) == 40
