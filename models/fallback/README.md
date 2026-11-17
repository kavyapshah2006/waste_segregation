# Unknown-item fallback (WP7)

When the detector stays unsure about an object (`pipeline/stability.py` says "unsure" or "empty"), the app asks a general image–text model which KB items the image looks like, and offers the top 3 for the user to confirm ("Is this a …?"). It never gives a verdict on its own.

## Model and licence

`google/siglip2-base-patch16-224`, Apache-2.0, not gated, pinned to revision `75de2d55`. Only the image encoder runs on the phone; the KB items' text embeddings are computed on the laptop and shipped as a small JSON file.

Apple's MobileCLIP was considered and rejected: its code is MIT, but its weights are licensed for non-commercial research only and exclude product development.

## Build

```bash
uv run --group finetune python models/fallback/build.py
```

Downloads the model (1.5 GB), writes `models/fallback/out/`:

| File | Size | Check against PyTorch |
|---|---|---|
| `siglip2_image_fp32.pte` | 372 MB | cosine 1.00000 |
| `siglip2_image_int8.pte` | 94 MB | cosine ≥ 0.9989 (dynamic int8, XNNPACK) |
| `text_embeddings.json` | ~0.3 MB | one embedding per KB item: mean of "a photo of <name>" over its English names and aliases |

The phone uses the int8 encoder. Input: RGB resized to 224×224 (bilinear, no padding), scaled to 0–1, then `(x − 0.5) / 0.5`; output: an L2-normalised 768-d embedding. Rebuild `text_embeddings.json` after KB changes.

## Sanity check (2026-11-17, laptop)

`models/fallback/check_crops.py` on 133 labelled object crops from 80 TACO images:

| Encoder | Top-1 item | Top-3 item | Top-1 bin |
|---|---|---|---|
| fp32 | 69.2% | 83.5% | 83.5% |
| int8 | 68.4% | 83.5% | 81.2% |

The int8 and fp32 encoders pick the same top item 93% of the time. TACO is mostly European street litter and covers 15 of our 40 items, so this is a sanity check, not a benchmark; repeat it on the India photo set.

## Code

- `pipeline/fallback.py`: preprocessing, `Fallback` (runs the `.pte`), `rank()` (cosine ranking + SigLIP sigmoid score).
- `android/core/.../FallbackRank.kt`: the same ranking in Kotlin, checked on `tests/fixtures/fallback_rank.json`.
