import json
from collections import Counter

import pytest
import yaml
from PIL import Image

from data.scripts.prepare import label_map, load_taxonomy, prepare, split_for


def test_label_map_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Mystery"):
        label_map(load_taxonomy(), "taco", ["Clear plastic bottle", "Mystery"])


def test_identity_dataset_only_accepts_kb_classes():
    taxonomy = load_taxonomy()
    assert label_map(taxonomy, "india", ["kulhad"]) == {"kulhad": "kulhad"}
    with pytest.raises(ValueError, match="dog"):
        label_map(taxonomy, "india", ["dog"])


def test_split_is_deterministic_and_roughly_80_10_10():
    names = [f"img_{i}.jpg" for i in range(3000)]
    first = [split_for("taco", n) for n in names]
    assert first == [split_for("taco", n) for n in names]
    share = {k: v / len(names) for k, v in Counter(first).items()}
    assert share["train"] == pytest.approx(0.8, abs=0.03)
    assert share["val"] == pytest.approx(0.1, abs=0.03)
    assert share["test"] == pytest.approx(0.1, abs=0.03)


def _write_coco(root, images, annotations, categories):
    (root / "images").mkdir(parents=True)
    coco = {"images": [], "annotations": annotations, "categories": categories}
    for img_id, (name, size, exif_orientation) in enumerate(images):
        exif = Image.Exif()
        if exif_orientation:
            exif[0x0112] = exif_orientation
        Image.new("RGB", size, (10, 200, 10)).save(root / "images" / name, exif=exif)
        coco["images"].append(
            {"id": img_id, "file_name": name, "width": size[0], "height": size[1]}
        )
    (root / "annotations.json").write_text(json.dumps(coco))
    return root / "annotations.json"


def test_prepare_end_to_end(tmp_path):
    categories = [{"id": 1, "name": "Clear plastic bottle"}, {"id": 2, "name": "Cigarette"}]
    annotations = [
        {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 60]},
        {"id": 2, "image_id": 0, "category_id": 2, "bbox": [0, 0, 5, 5]},  # dropped label
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [90, 90, 50, 50]},  # clipped to image
        {"id": 4, "image_id": 2, "category_id": 1, "bbox": [1, 1, 10, 10]},  # EXIF size mismatch
    ]
    ann = _write_coco(
        tmp_path / "taco",
        [("a.jpg", (200, 100), None), ("b.jpg", (100, 100), None), ("c.jpg", (200, 100), 6)],
        annotations,
        categories,
    )
    taxonomy = load_taxonomy()
    taxonomy["min_instances"] = 1
    out = tmp_path / "out"

    report = prepare({"taco": ann}, out, taxonomy)

    assert report["instances"] == {"pet_bottle": 3}
    assert report["dropped_labels"] == {"taco": {"Cigarette": 1}}
    assert (
        len(report["skipped_size_mismatch"]) == 1 and "c.jpg" in report["skipped_size_mismatch"][0]
    )
    assert "kulhad" in report["classes_left_out"]
    dataset = yaml.safe_load((out / "dataset.yaml").read_text())
    assert dataset["names"] == {0: "pet_bottle"}

    labels = {p.stem: p.read_text().split() for p in out.glob("labels/*/*.txt")}
    assert labels["taco__a"] == ["0", "0.150000", "0.500000", "0.200000", "0.600000"]
    assert labels["taco__b"] == ["0", "0.950000", "0.950000", "0.100000", "0.100000"]
    assert "taco__c" not in labels
    assert len(list(out.glob("images/*/*.jpg"))) == 2
