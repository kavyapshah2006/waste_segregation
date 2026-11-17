# Llama phone benchmark (WP1 run-list)

One borrowed-phone session, about 45 minutes. Prepare everything on the laptop first.

## Before the session (laptop)

- [ ] `.pte` and `tokenizer.model` downloaded (`models/llm/README.md`)
- [ ] ExecuTorch Llama Android demo app built against `org.pytorch:executorch-android:1.5.1` (from the `executorch-examples` repo, `llm/android/LlamaDemo`)
- [ ] `adb.exe devices` works with a test phone, or at least on the laptop
- [ ] Copy `eval/results/llm_TEMPLATE.json` to `eval/results/llm_<device>_<yyyy-mm-dd>.json`

## During the session

1. Ask the owner, enable Developer options and USB debugging, connect.
2. `./scripts/device_info.sh > docs/DEVICE.md`
3. `./scripts/push_models.sh <pte> <tokenizer.model>`
4. Install the demo app (`adb.exe install -r app-debug.apk`), select the model and tokenizer, model type Llama 3.
5. **Turn on airplane mode.** Close other apps. Note the battery level and whether the phone feels warm.
6. Run each prompt below three times. From the app's stats, record load time, time to first token, prefill tok/s, decode tok/s.
7. Peak memory: while generating, `adb.exe shell dumpsys meminfo <package> | grep TOTAL`.
8. Sustained run: keep asking prompt 1 for 5 minutes, then record decode tok/s again.
9. Save one full answer per prompt (copy from the app) for the verifier: `eval/results/llm_<device>_<date>_answers.txt`.
10. Uninstall the app and turn off USB debugging if the owner prefers.

## Prompts

Use the exact prompts from these golden fixtures (field `expected.prompt`) so results can be verified later:

1. `tests/fixtures/golden/pet_bottle_ok_en.json`
2. `tests/fixtures/golden/cardboard_greasy_en.json`
3. `tests/fixtures/golden/diaper_ok_hi.json` (Hindi)

Print a prompt with:

```bash
python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['expected']['prompt'])" tests/fixtures/golden/pet_bottle_ok_en.json
```

## After the session

```bash
uv run python -m pipeline.cli --item pet_bottle --llm-output eval/results/<answer-file>
```

Commit the results JSON and answers, then update the WP1 row in `docs/FuturePlan.md`.
