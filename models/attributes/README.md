# Dirty / broken check

Ten KB items change their advice with condition: a dirty milk pouch, PET bottle, Tetra Pak,
newspaper, cardboard box, carry bag or aluminium foil goes to a different bin, and a broken CFL
bulb, mercury thermometer or glass bottle needs hazard handling. For those items only, the app
checks the crop's condition.

## How it works

No extra model ships. The app already carries the SigLIP 2 image encoder for the unknown-item
fallback ([models/fallback/](../fallback/README.md)). For each of the ten items,
`build.py` stores two text embeddings per attribute, for example "a photo of a broken glass
bottle or jar" against "a photo of an intact glass bottle or jar". On the phone:

    margin = cos(image, positive) - cos(image, negative)

Above `high` the attribute is set, below `low` it is not, and in between the app asks the user
("Is it broken?"). Code: `pipeline/attributes.py`, Kotlin `Attributes.kt` (same cases:
`tests/fixtures/attribute_cases.json`).

```bash
uv run --group finetune python models/fallback/build.py --text-only   # if the KB changed
uv run --group finetune python models/attributes/build.py             # writes out/attribute_prompts.json
uv run --group finetune python models/attributes/check_attributes.py \
    --dataset data/processed/waste --label taco_glass
```

## Result so far (zero-shot, int8 encoder)

TACO has no dirty or broken flags, but it labels broken glass apart from glass bottles. As a
proxy, 136 broken-glass crops count as "broken" and 110 glass-bottle crops as "intact", all
scored with the glass bottle's prompts (`eval/results/attributes_taco_glass.json`):

| Measure | Value |
|---|---|
| AUC (0.5 = chance) | 0.678 |
| Narrow band ±0.01: answered / right when answered | 49% / 68% |
| Band −0.0196 … 0.0304 (shipped): answered / right when answered | 16% / 90% |

Zero-shot is weak: on its own it can settle "broken" for about one crop in six, so the app asks
the user in most cases. "Dirty" cannot be measured without labelled photos, so for now the app
always asks. TACO's broken glass is mostly shards on the ground, not a cracked bottle in a hand,
so even these numbers are only a rough guide.

## Next

With the India photo set (checklist steps 19 and 25): label dirty and broken, recalibrate the
bands, and if zero-shot stays weak, train a small head on the same SigLIP embeddings (a linear
probe needs no new model on the phone) or a separate classifier.
