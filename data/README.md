# Data

| Path | What it is | In git |
|---|---|---|
| `taxonomy.yaml` | Detector classes (= KB item ids) and how each dataset's labels map to them | yes |
| `scripts/download_taco.py` | Downloads TACO annotations (pinned commit) and images | yes |
| `scripts/prepare.py` | COCO datasets → one YOLO dataset with our classes, deterministic train/val/test split | yes |
| `raw/` | Downloaded datasets | no |
| `processed/` | Prepared dataset, `dataset.yaml`, `report.json` | no |

```bash
uv run python data/scripts/download_taco.py            # --limit 20 for a quick sample
uv run python data/scripts/prepare.py                  # TACO only
uv run python data/scripts/prepare.py --coco india=data/raw/india/annotations.json
```

### Image licences

TACO images carry mixed licences (ODbL, plain "CC", or none). `prepare.py --licences all|known|odbl` chooses which to keep; `uv run python data/scripts/licence_report.py` writes [taco_licences.md](taco_licences.md), which shows what each choice costs. With every image TACO trains only 14 of the 53 KB items (`known`: 9, `odbl`: 7), so the India photo set and other datasets are needed in any case.

`prepare.py` applies EXIF orientation and skips any image whose oriented size disagrees with its annotation. It also leaves out classes with fewer than `min_instances` boxes (listed in `report.json`). The split is a hash of dataset and file name, so adding images never moves existing ones between splits.

## Datasets

| Dataset | Status | Needs |
|---|---|---|
| TACO | Scripted; 12-image sample tested on 2026-11-17 | Nothing. Verify the data licence before shipping |
| Roboflow e-waste | Mapping placeholder in `taxonomy.yaml` | Roboflow account + API key, then fill `map`/`drop` |
| Garbage Dataset (2026, Kaggle) | Not used for detection (classification images) | Kaggle account; candidate for the attribute head (WP7) |
| India photos | Planned (WP7), labelled with KB ids directly | Taking photos |

TACO alone covers about 14 of our 40 classes (bottles, cans, cartons, paper, bags, wrappers, foam, food waste, batteries, blister packs, aerosols). The rest (e-waste, sanitary, Indian items such as kulhad and milk pouches) need the other sources.
