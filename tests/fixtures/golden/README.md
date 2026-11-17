# Golden fixtures

Each `*.json` file is one end-to-end case: a detection (or none), a language, an optional question and an optional recorded Llama output, with the expected verdict, prompt and verified answer.

Both implementations must reproduce `expected` exactly:

- Python: `tests/test_pipeline_golden.py`
- Android (from WP6): a JVM unit test reading this same folder

To add a case, copy a file, change `id` (must match the file name), `description` and `input`, set `expected` to `null`, then run:

```bash
uv run python tests/fixtures/golden/regen.py
```

Review the diff of every changed fixture by hand before committing. Regenerating after a code change without reading the diffs defeats the purpose of these files.
