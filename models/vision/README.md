# Vision: waste detector

YOLO26n (Ultralytics 8.4.202), fine-tuned on our taxonomy (`data/taxonomy.yaml`) and exported to ExecuTorch with XNNPACK.

| File | Purpose |
|---|---|
| `train.py` | Train on `data/processed/waste/dataset.yaml`; writes per-class test metrics |
| `train.ipynb` | Colab/Kaggle GPU notebook wrapping the data scripts and `train.py` |
| `export_executorch.py` | `.pt` → `model.pte` + `metadata.yaml`, then a host check against PyTorch |
| `out/` | Export output (gitignored model files) |

## Pipeline

```bash
uv sync --group vision
uv run python data/scripts/download_taco.py          # data/raw/taco (gitignored)
uv run python data/scripts/prepare.py                # data/processed/waste + report.json
# train on a GPU: models/vision/train.ipynb  (or --smoke on the laptop to test the flow)
uv run --group vision python models/vision/export_executorch.py --weights best.pt --check-images <dir>
uv run python -m pipeline.cli --image photo.jpg --detector models/vision/out/best_640/model.pte
```

## Output format (important for the app)

The Ultralytics 8.4.x ExecuTorch exporter **does not keep YOLO26's NMS-free head** (it logs "This export format does not support end2end models"). The `.pte` outputs raw predictions `(1, 4 + num_classes, 8400)` for a 640×640 input: centre x, centre y, width, height in input pixels, then sigmoid class scores. NMS is done in `pipeline/detect.py`, and the app must do the same. The exact pre- and post-processing steps are in that module's docstring.

## Verified so far (host, not reportable)

Pretrained COCO `yolo26n.pt` exported on 2026-11-17 with ExecuTorch 1.5.1 (9.4 MB `.pte`). `pipeline.detect` on that `.pte` vs Ultralytics PyTorch inference:

| Image | Detections (pte / torch) | Min IoU |
|---|---|---|
| bus.jpg | 5 / 5 | 0.937 |
| zidane.jpg | 3 / 3 | 0.979 |

Confidence differs slightly because Ultralytics predicts on a non-square letterbox while the `.pte` input is fixed at 640×640.

## Not done yet (needs a human)

- **Training**: run `train.ipynb` on Colab/Kaggle GPU and upload `best.pt` to a GitHub Release.
- **More data**: the Roboflow e-waste set needs a Roboflow account; Indian photos are WP7.
- **Baseline metrics** table goes here after the first training run.

## Licence

Ultralytics code and models trained with it are AGPL-3.0, so this repo will be released under AGPL-3.0 (FuturePlan section 3).
