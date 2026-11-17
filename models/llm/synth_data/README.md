# Synthetic fine-tuning data (WP8)

`generate.py` turns the knowledge base into prompt/completion pairs for supervised fine-tuning:

- **Prompts** are exactly what the app sends (`pipeline/prompt.py`).
- **Completions** are the JSON answers we want, written from KB facts and ending in `<|eot_id|>`.
- **Coverage:** 40 items × each condition variant × English and Hindi × 5 question kinds (explain, which bin, recyclable?, how to prepare, and an off-topic question that the model should decline). That's 470 examples in KB v1.
- **Verified:** every completion must pass `pipeline/verify.py` with status `ok`, or generation fails.
- **Held-out split:** about 15% of items are hashed into `val` and never appear in `train`, so evaluation measures answers about unseen items.

```bash
uv run python models/llm/synth_data/generate.py           # writes out/{train,val}.jsonl (gitignored)
uv run python eval/llm_quality/run.py --write-cases cases.jsonl
uv run python eval/llm_quality/run.py --answers answers.jsonl --label phone_base
```

## Limits

- The text comes from templates: repetitive, but grounded. The aim is to teach the output format and staying on the facts, not new knowledge.
- Hindi templates and KB Hindi text are unreviewed.

## Not done yet

The LoRA fine-tune itself (Colab GPU + the gated `meta-llama/Llama-3.2-1B-Instruct` checkpoint), merging the adapter, re-exporting with `models/llm/export.py --mode int4`, and comparing base and fine-tuned with `eval/llm_quality/run.py`. Ship the fine-tune only if its useful rate beats base + prompt after 4-bit quantisation.
