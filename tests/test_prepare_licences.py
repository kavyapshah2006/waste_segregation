import json

import pytest
from PIL import Image

from data.scripts.prepare import licence_ok, prepare

ODBL = "ODBL (c) OpenLitterMap & Contributors"


@pytest.mark.parametrize(
    ("lic", "policy", "keep"),
    [
        (None, "all", True),
        (None, "known", False),
        ("CC", "known", True),
        ("CC", "odbl", False),
        (ODBL, "odbl", True),
        (ODBL, "known", True),
    ],
)
def test_licence_ok(lic, policy, keep):
    img = {"id": 1} if lic is None else {"id": 1, "license": lic}
    assert licence_ok(img, policy) is keep


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        licence_ok({}, "some")


def test_prepare_drops_images_by_licence_and_reports_it(tmp_path):
    images = tmp_path / "taco" / "images"
    images.mkdir(parents=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Clear plastic bottle"}],
    }
    for i, lic in enumerate([ODBL, "CC", None]):
        name = f"img{i}.jpg"
        Image.new("RGB", (100, 80)).save(images / name)
        entry = {"id": i, "file_name": name, "width": 100, "height": 80}
        if lic:
            entry["license"] = lic
        coco["images"].append(entry)
        coco["annotations"].append(
            {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 30, 40]}
        )
    ann = tmp_path / "taco" / "annotations.json"
    ann.write_text(json.dumps(coco))
    taxonomy = {
        "classes": ["pet_bottle"],
        "min_instances": 1,
        "datasets": {"taco": {"map": {"Clear plastic bottle": "pet_bottle"}, "drop": []}},
    }
    report = prepare({"taco": ann}, tmp_path / "out", taxonomy, licences="odbl")
    assert sum(report["images_per_split"].values()) == 1
    assert report["dropped_by_licence"] == {"CC": 1, "None": 1}
    assert report["licence_policy"] == "odbl"
