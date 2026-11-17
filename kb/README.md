# Knowledge base

Source of truth for disposal decisions under India's Solid Waste Management Rules 2026: each item maps to a stream (wet, dry, sanitary, special care), a bin colour, preparation steps, hazards, a drop-off point and an impact fact. Llama explains these facts; it never decides them.

| Path | What it is |
|---|---|
| `items/<id>.json` | One file per item (40 in v1), validated by `schema.json` |
| `cities/<id>.json` | City profiles (bin colours, per-item overrides, drop-off text), validated by `city_schema.json`. `default_in` is the national default |
| `strings.json` | Stream, colour and language names |
| `build_kb.py` | Validates everything and writes `build/kb.sqlite` (gitignored) |

```bash
uv run python kb/build_kb.py          # validate + write build/kb.sqlite
uv run python kb/build_kb.py --check  # validate only
```

## Adding or changing an item

1. Copy an existing item file in the same stream; the file name must equal `id`.
2. Fill in `en` and `hi` for `names`, `prep_steps` and `impact`; add a source.
3. Set `reviewed.hi` to `false` unless a fluent Hindi speaker has checked the text.
4. Run `uv run python kb/build_kb.py` and `uv run pytest`.

## How a decision is resolved

1. Start from the item file.
2. Apply the city's `overrides` for that item (local rules win over the national default).
3. Apply `condition_rules` whose `if` matches the detected attributes (`contaminated`, `broken`).
4. The bin colour comes from the city's `bin_colours[stream]`.

The logic lives in `pipeline/kb.py`; the Android app ports it and is checked against the same golden fixtures.

## Review status

All Hindi text in v1 was written without a native-speaker review and is marked `"reviewed": {"hi": false}`. `build_kb.py` prints how many items still need review.
