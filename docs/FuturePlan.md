# Future Plan: Implementation Guide

A self-contained plan for building the waste segregation assistant, written so that an AI coding model (or a new teammate) can execute it one work package at a time without any other context.

**Last researched:** 2026-11-17. Versions and model names below were checked on that date; re-check anything marked *verify* before relying on it.

---

## 1. How to use this file

1. Read sections 2–5 fully before writing code. They are the rules and the interfaces.
2. Pick the next work package (WP) from the status table in section 12 whose dependencies are done.
3. Do exactly that WP. Do not start another in the same session unless it is trivial.
4. A WP is done only when its **acceptance check** passes, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` are green, and the status table is updated in the same commit.
5. If something in this file is wrong or out of date, fix this file in the same commit and say so in the commit message.

Related docs: [DESIGN.md](DESIGN.md) (why), [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md), [roadmap.md](roadmap.md) (phase overview with trade-offs), [DEV_SETUP.md](DEV_SETUP.md). Where they disagree with this file, **this file wins** (see section 3).

---

## 2. Project in one page

**Challenge:** Bharat AI-SoC Challenge, PS2. Build an assistant that identifies waste, classifies it, gives disposal advice, explains recycling, and educates users. It must run on an Arm device with **Llama (reasoning/advice) deployed through ExecuTorch**. No cloud inference.

**Judging:** technical innovation (Llama + ExecuTorch on Arm), accuracy and performance, real-world viability, social and environmental impact. Slide targets: 92% sorting accuracy, <50 ms inference, 0 KB cloud traffic. These are goals to measure, not claims.

**Deliverables:** source code, working app, demo video, technical report.

**Core idea:** *vision perceives, the knowledge base decides, Llama explains.*

```
Camera / photo
  → detector (YOLO26n, ExecuTorch)            what is it, where
  → attribute head (ExecuTorch)               clean/dirty, intact/broken
  → temporal voting + confidence gate         or "not sure"
  → knowledge base lookup (SQLite)            stream, bin, steps, hazards   ← decision, instant
  → verdict card + bin colour + voice line    shown immediately
  → Llama 3.2 (ExecuTorch, Arm CPU)           explanation, Q&A, education   ← streams afterwards
  → verifier                                  KB overrides Llama on facts
```

**Target device:** one borrowed Arm Android phone (arm64-v8a, 6 GB+ RAM). The dev laptop is x86 and is never a demo or benchmark device.

---

## 3. Fixed decisions (do not reopen)

| Area | Decision |
|---|---|
| Platform | Android phone first. Raspberry Pi 5 kiosk is optional (WP13). iPhone is not a target |
| LLM | Llama 3.2 1B Instruct, INT4, ExecuTorch, XNNPACK (KleidiAI). 3B only if the phone has 8 GB+ and benchmarks allow |
| Decision logic | Disposal stream, bin, hazards and drop-off come from the knowledge base, never from Llama |
| Regulation | India's Solid Waste Management Rules 2026: wet, dry, sanitary, special care |
| Detector | **Ultralytics YOLO26n** (8.4.202), exported with `format="executorch"`. The exporter drops YOLO26's NMS-free head, so the `.pte` outputs raw `(1, 4 + classes, 8400)` predictions and **NMS runs in post-processing** (`pipeline/detect.py`, then the app). *Supersedes YOLO11n in DESIGN.md and roadmap.md* |
| Licence | Repo is released under **AGPL-3.0** at submission, because Ultralytics code and trained weights are AGPL-3.0. *New since DESIGN.md* |
| App | Kotlin, Jetpack Compose, CameraX, `org.pytorch:executorch-android` |
| Python | uv only (`uv add`, `uv run`, `uv sync --locked`). Python 3.12. `torch==2.14.0` + `executorch==1.5.1` move together |
| Training | Google Colab or Kaggle GPUs. The laptop has no CUDA |
| Speech | sherpa-onnx (ASR + TTS), separate pipeline from Llama |
| Languages | English, Hindi first; then Tamil, Telugu, Bengali, Marathi |
| Git | Plain commit messages. **No AI co-author or "generated with" lines.** Pushes go over SSH and need the repo owner's passphrase |

---

## 4. Ground rules for the implementing model

- **Verify before pinning.** Check a version, API name or model id against its official source before writing it into code. If you cannot confirm it, mark it `TODO(verify)` and say so.
- **Never commit** model binaries (`.pte`, `.pt`, `.onnx`, `.hef`), raw datasets, APKs or secrets. `.gitignore` already covers these. Models go to GitHub Releases; `scripts/download_models.sh` fetches them.
- **Tests with every WP.** Python logic gets `pytest` tests. Kotlin logic gets JVM unit tests. Anything shared uses the golden fixtures (section 5.6).
- **Small commits**, one logical change each, CI green.
- **Do not fake results.** Benchmarks and accuracy numbers come only from real runs, saved under `eval/results/` with device, model file hash and commit hash. Laptop numbers are labelled "host, not reportable".
- **Stop and ask the human** for anything in the table below.

| Needs a human | Why |
|---|---|
| Any step on the phone (install, run, benchmark, record) | Phone is borrowed and only available in sessions. Prepare everything first, then hand over a short run-list |
| Hugging Face login and Llama 3.2 licence acceptance | Account action |
| Taking training photos | Physical task (iPhone is fine for this) |
| `git push` | SSH passphrase is typed by the repo owner |
| Colab/Kaggle runs | Account and GPU quota; provide a ready notebook |
| Reviewing Hindi and other Indic text | Native-speaker check |

---

## 5. System contracts

Build against these. Change them only by editing this section in the same commit.

### 5.1 Waste streams (SWM Rules 2026)

| `stream` id | Name | Default bin colour | Examples |
|---|---|---|---|
| `wet` | Wet waste | green | Food scraps, peels, tea leaves, garden waste |
| `dry` | Dry waste | blue | Paper, cardboard, PET, metal cans, glass, multilayer packets |
| `sanitary` | Sanitary waste | red | Diapers, sanitary pads, used tissues (wrapped securely) |
| `special_care` | Special care waste | black | Batteries, bulbs, medicines, paint cans, small e-waste, sharps |

Colours are a default; a city profile may override them.

### 5.2 Knowledge base item (`kb/items/<item_id>.json`)

```json
{
  "id": "pet_bottle",
  "names": {"en": "PET plastic bottle", "hi": "पीईटी प्लास्टिक की बोतल"},
  "aliases": {"en": ["water bottle", "soft drink bottle"], "hi": ["पानी की बोतल"]},
  "material": "plastic_pet",
  "stream": "dry",
  "recyclable": true,
  "prep_steps": {"en": ["Empty it", "Rinse", "Crush and put the cap back on"], "hi": ["..."]},
  "hazard": null,
  "drop_off": null,
  "condition_rules": [
    {"if": {"contaminated": true}, "then": {"prep_steps_key": "rinse_first"}}
  ],
  "impact": {"en": "Recycled PET becomes fibre for clothes and new bottles.", "hi": "..."},
  "typical_weight_g": 25,
  "detector_classes": ["pet_bottle"],
  "sources": ["SWM Rules 2026", "https://..."]
}
```

Implemented in `kb/schema.json`. Beyond the example above, every item also has `reviewed` (per-language flag for native-speaker review) and may have `compostable`. Stream, colour and language names live in `kb/strings.json`.

Rules:
- `stream` is one of the four ids in 5.1. `drop_off` is `null` or one of `e_waste_centre`, `pharmacy_takeback`, `hazardous_collection`, `scrap_dealer`, `municipal_sanitary`.
- `condition_rules` may change `stream`, `recyclable`, or steps based on attributes (for example greasy cardboard → `wet` or non-recyclable dry, per city).
- Every item needs `en` and `hi` text and at least one source.
- `kb/schema.json` (JSON Schema) is the validator. `kb/build_kb.py` validates and writes `kb.sqlite`.

### 5.3 City profile (`kb/cities/<city_id>.json`)

```json
{
  "id": "default_in",
  "name": {"en": "India (default)", "hi": "भारत (डिफ़ॉल्ट)"},
  "bin_colours": {"wet": "green", "dry": "blue", "sanitary": "red", "special_care": "black"},
  "overrides": {"greasy_cardboard": {"stream": "wet"}},
  "drop_off_info": {"e_waste_centre": {"en": "Take it to an authorised e-waste collection centre.", "hi": "..."}}
}
```

### 5.4 Detection record (vision → rest of the system)

```json
{
  "item_id": "pet_bottle",
  "detector_class": "pet_bottle",
  "confidence": 0.91,
  "bbox": [0.12, 0.30, 0.45, 0.80],
  "attributes": {"contaminated": false, "broken": false},
  "source": "detector"
}
```

- `bbox` is `[x1, y1, x2, y2]` normalised to 0–1.
- `detector_class` is the model's class name; `item_id` is set only when that name is a KB item in `data/taxonomy.yaml`.
- `source` is `detector`, `clip_fallback` or `user_typed`.
- Below the confidence gate the record is replaced by `{"item_id": null, "reason": "low_confidence"}` and the UI asks the user to reframe, show the label, or type the item.

### 5.5 Llama prompt and output

Prompt uses the Llama 3.2 chat format. The system message is identical for every item and language (so its KV cache can be pre-filled once); the user message carries everything that varies. Implemented in `pipeline/prompt.py`; the exact text for each golden fixture is stored in `expected.prompt`.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a waste disposal guide for India. Use only the FACTS given and never contradict them.
The app already shows the facts to the user; your job is to explain. Reply with one JSON object
with exactly two keys: "why" (answer the QUESTION in one or two short sentences) and "tip" (one
short practical tip). Write both in the requested LANGUAGE. Output nothing except the JSON
object.<|eot_id|><|start_header_id|>user<|end_header_id|>

FACTS: {verdict facts JSON}
CONDITION: {attributes JSON}
LANGUAGE: {language name}
QUESTION: {question, default "Explain how to dispose of this and why."}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"why": "
```

The prompt ends with the answer prefix `{"why": "` (`ANSWER_PREFIX`), so the model's first token is already inside the `why` string. The runner returns only what comes after it; join the two (`full_answer()`) before verifying. The retry prompt (`short=True`) keeps only item, stream, bin and steps in FACTS and drops CONDITION.

Llama writes only:

```json
{"why": "Clean PET is easy to recycle; leftover liquid spoils paper in the same bin.",
 "tip": "Reuse it a few times before recycling."}
```

Why two keys: the first format asked Llama to copy item, stream, bin, steps and language back as well. The app never showed those copies (it uses the KB), but the model spent most of its tokens on them: in the stock-model baseline nearly every Hindi answer hit the 200-token limit and was cut off, and only 10% of answers were useful (`eval/results/llm_quality_spinquant_stock_host.json`). Two keys cut the answer to a fraction of the tokens, which also cuts phone latency.

Verifier rules (identical in Python `pipeline/verify.py` and Kotlin):
1. Remove `<|eot_id|>` / `<|end_of_text|>` text, then extract the first balanced `{...}` (string- and escape-aware). One repair: if the object ends after a complete value but lacks its final `}`, close it. Text cut off inside a string is never repaired. If parsing fails, retry once with the short prompt, then fall back to KB text only (`status: "fallback"`).
2. `item`, `stream`, `bin`, `steps`, `hazard` and `drop_off` shown to the user always come from the KB. If Llama adds `stream` or `bin` keys anyway and they disagree, a contradiction is recorded (stream ids, stream names and colour names in any supported language count as agreeing).
3. If Llama adds a `steps` key, it is never shown; if it names a different bin, `steps_bin` is recorded.
4. `why` and `tip` are the only text shown from Llama. Each is whitespace-normalised, cut to 400 characters, and dropped (`why_bin` / `tip_bin`) if it names a bin other than the verdict's bin or a bin the KB steps or notes themselves name (greasy cardboard: green and blue). After a stream or bin contradiction, only the verdict's bin is allowed.
5. Content checks, clause by clause (clauses split at punctuation, "and/but/or/so that/because", "और/लेकिन/या/ताकि/क्योंकि/जिससे"). If either field fails any check (including rule 4), **both** `why` and `tip` are dropped: in the stock-model run the other field usually repeated the same mistake.
   - `*_negates`: a negated clause ("not", "never", "don't", "नहीं", "मत", standalone "न") names the verdict's own bin or stream, or the bin or stream follows "instead of" (English) or comes before "के बजाय / की जगह" (Hindi).
   - `*_recyclable` / `*_compostable`: any clause claims the opposite of the KB flag ("not recyclable", "रीसायकल नहीं हो सकता" vs "can be recycled", "रीसायकल किया जा सकता है"). A bare leading "No," / "नहीं," or "Yes," / "हाँ," in a text about recycling is a claim too ("No, glass can be recycled" contradicts itself and fails either way). Hedged claims ("rarely", "hard to", "शायद ही", "मुश्किल") count as no claim.
   - `*_negates_step`: a negated clause carries at least 80% of the content words of a positive KB step ("Do not remove stickers").
   - `*_unsafe`: an un-negated clause tells the user to put waste in a drain, toilet, road or river (English verbs, Hindi command forms only) or to burn it, unless it repeats the KB's own text (diapers: empty solid waste into the toilet).
   These are word rules, chosen to drop rather than pass when unsure. Every KB-written training answer passes them (the generator fails otherwise).
6. Output `status` is `ok` (no contradictions), `corrected` (contradictions found and overridden), `fallback`, or `declined` (see below).

**Off-topic guard** (`pipeline/assistant.py`, before any Llama call): a free-text question that uses no waste word (English or Hindi list), does not name the item, and does not say "this"/"यह" is answered with the fixed `messages.off_topic` text from `kb/strings.json`, without calling Llama (`status: "declined"`, no prompt). The system prompt also tells Llama to decline off-topic questions, as a second line.

EOS ids for Llama 3.2: 128009 and 128001. *Verify the chat template against Meta's Llama 3.2 model card before first use.*

### 5.6 Golden fixtures (`tests/fixtures/golden/`)

JSON files of `{id, description, input: {detection, city, language, question, raw_llm_output}, expected: {verdict, prompt, verified}}`. `raw_llm_output` is the text the model generated after the prompt (it starts inside the `why` string). The Python pipeline and the Kotlin app both run them, so the two implementations cannot drift (including the exact prompt text). Add a fixture for every bug found. `tests/fixtures/golden/regen.py` recomputes `expected` after an intended change; review every diff by hand before committing.

### 5.7 Android module boundaries (`android/app/src/main/java/.../`)

| Package | Responsibility | Depends on |
|---|---|---|
| `camera` | CameraX preview, frame capture, resize/normalise | – |
| `vision` | Detector + attribute head via ExecuTorch `Module`; temporal voting; confidence gate; CLIP fallback | `camera` |
| `kb` | Read `kb.sqlite` from assets; item + city lookup; condition rules | – |
| `llm` | `LlmModule` wrapper, prompt builder, streaming, verifier | `kb` |
| `speech` | sherpa-onnx ASR and TTS | – |
| `impact` | Local log (Room), stats, education content | `kb` |
| `ui` | Compose screens: Scan, Verdict, Chat, Impact, Settings | all |

ExecuTorch Android API (from the official docs, *verify signatures against the 1.5.1 AAR*):
- Dependency: `implementation("org.pytorch:executorch-android:1.5.1")`. The AAR version must equal the Python `executorch` version used for export.
- LLM: `org.pytorch.executorch.extension.llm.LlmModule`, `LlmModuleConfig`, `LlmGenerationConfig`, `LlmCallback` (`onResult`, `onStats`, `onError`), plus `load`, `stop`, `resetContext`. Run generation on a background `HandlerThread`.
- Vision: `org.pytorch.executorch.Module.load(path)`, `module.forward(EValue.from(tensor))`, `Tensor.fromBlob(...)`.
- Large model files are pushed to app storage (`adb push` during development, download-on-first-run or bundled asset pack for release), not compiled into the APK.

---

## 6. Host reference pipeline (develop without the phone)

A Python package `pipeline/` that mirrors the app's logic on the laptop:

```
pipeline/
├── detect.py      # run the detector .pte with executorch.runtime, decode outputs
├── kb.py          # sqlite lookup, city overrides, condition rules
├── prompt.py      # build the Llama prompt (section 5.5)
├── verify.py      # parse + verify Llama output
├── llm.py         # optional: run the Llama .pte on host for functional checks
└── cli.py         # uv run python -m pipeline.cli --image x.jpg --lang hi
```

Why: the phone is borrowed, so most logic must be testable in CI. `kb`, `prompt` and `verify` are pure functions with golden tests; the Kotlin versions are ports checked against the same fixtures.

Limits: host runs are x86 and **never reported** as results. Their only purpose is correctness.

---

## 7. Work packages

Format: **Goal · Depends on · Human needed · Create · Steps · Tests · Acceptance.**

### WP1: Llama on the phone

- **Goal:** Llama 3.2 1B answering on the borrowed phone through ExecuTorch, with a benchmark file.
- **Depends on:** Phase 0 (done). **Human:** HF licence, phone session.
- **Create:** `models/llm/README.md`, `models/llm/export.py`, `scripts/download_models.sh`, `scripts/push_models.sh`, `eval/latency/llm_bench.md`, `eval/results/llm_<device>_<date>.json`, `docs/DEVICE.md`.
- **Steps:**
  1. Fast path: `./scripts/download_models.sh llama-spinquant --accept-llama-license` fetches the prebuilt `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` (`.pte` about 1.1 GB plus tokenizer; not gated). Alternative: `llama-qlora`. First check it loads on the host with `pipeline.cli --llm-model`, since these files predate ExecuTorch 1.0. (Done: it loads with ExecuTorch 1.5.1 once the `llama::custom_sdpa`, `llama::fast_hadamard_transform` and `quantized_decomposed::embedding_byte` kernels are registered, which `pipeline/llm.py` does. The Android AAR must include the same custom and quantized kernels; check this in the first phone session.)
  2. Build the ExecuTorch Llama Android demo app (from the `executorch-examples` repo) against AAR 1.5.1. Push model + tokenizer with `adb.exe push`.
  3. Human session: run `scripts/device_info.sh > docs/DEVICE.md`; run three prompts in airplane mode; record load time, time to first token, tokens/s, peak RSS, and tokens/s after 5 minutes of use.
  4. Own export: `models/llm/export.py` wraps `python -m extension.llm.export.export_llm` with the XNNPACK SpinQuant (or QAT+LoRA) config and Llama 3.2 params. It must reproduce a `.pte` that loads in the same app.
  5. If the phone has 8 GB+, repeat with 3B and record the comparison. Decide 1B vs 3B in `models/llm/README.md`.
- **Tests:** `tests/test_llm_export_config.py` checks the export script builds the right arguments (no model download in CI).
- **Acceptance:** benchmark JSON committed; a waste question gets a sensible answer on the phone in airplane mode.
- **Reference numbers** (ExecuTorch README, OnePlus 12, prompt length 64): 1B SpinQuant 50.2 tok/s, 0.3 s to first token, 1,921 MiB RSS; 3B SpinQuant 19.7 tok/s, 0.7 s, 3,726 MiB. Expect less on mid-range phones.

### WP2: Knowledge base v1

- **Goal:** validated KB with 40 items, default city, English + Hindi.
- **Depends on:** none. **Human:** Hindi review.
- **Create:** `kb/schema.json`, `kb/city_schema.json`, `kb/items/*.json` (40), `kb/cities/default_in.json`, `kb/build_kb.py`, `tests/test_kb.py`.
- **Steps:**
  1. Write the schemas from sections 5.2 and 5.3. Add `jsonschema` with `uv add`.
  2. Write items covering all four streams. Must include the hard cases: chips packet (multilayer), Tetra Pak, blister pack, medicine strip, greasy pizza box, broken glass, CFL/LED bulb, AA battery, phone, charger cable, sanitary pad, diaper, kulhad, tea bag, coconut shell, thermocol, paint can, syringe.
  3. `build_kb.py`: validate → check every item has `en` + `hi` → check `detector_classes` are unique → write `kb.sqlite` (tables `items`, `aliases`, `cities`, `strings`).
  4. Cite a source per item (SWM Rules 2026 text, E-Waste Rules, CPCB/SBM guidance).
- **Tests:** schema validation; no orphan references; alias lookup in both languages; condition rules change the result as expected.
- **Acceptance:** `uv run python kb/build_kb.py` exits 0 and writes `kb.sqlite`; tests green in CI.

### WP3: Data, taxonomy and detector baseline

- **Goal:** a YOLO26n detector for our classes, exported to `.pte`, verified on host.
- **Depends on:** WP2 (class list). **Human:** Colab run.
- **Create:** `data/taxonomy.yaml`, `data/scripts/download_taco.py`, `data/scripts/prepare.py`, `models/vision/{train.py,train.ipynb,export_executorch.py,README.md}`, `pipeline/detect.py`, tests.
- **Steps:**
  1. `taxonomy.yaml`: the 40 KB items as classes, plus each dataset's label map and drop list. Record each dataset's licence; drop non-commercial sets from anything shipped.
  2. `download_taco.py` (pinned annotations commit) and `prepare.py` (COCO → YOLO format, EXIF orientation applied, hash-based train/val/test split, classes under `min_instances` left out and reported).
  3. `train.ipynb` runs on Colab/Kaggle with the runtime's CUDA torch plus `uv pip install --system ultralytics==8.4.202`; `train.py` writes per-class test metrics. Upload `best.pt` to a GitHub Release.
  4. Export: `export_executorch.py --weights best.pt` (raw head, see section 3); if latency is too high, also export at 480 and 320.
  5. Host check: the same script runs `pipeline.detect` on the `.pte` and Ultralytics PyTorch inference on the same images and fails below IoU 0.9.
- **Tests:** every taxonomy class maps to an existing KB item; every external label maps to a class or is listed as dropped.
- **Acceptance:** baseline mAP table in `models/vision/README.md`; `.pte` output matches PyTorch within tolerance on host.
- **Note:** track clear plastic and glass separately; they are the known weak classes in prior work.

### WP4: Host reference pipeline

- **Goal:** `pipeline/` package with golden tests (section 6).
- **Depends on:** WP2. (Detector part needs WP3; LLM part can use recorded outputs.) **Human:** none.
- **Create:** `pipeline/*.py`, `tests/fixtures/golden/*.json` (20+), `tests/test_pipeline_*.py`.
- **Steps:** implement `kb.py`, `prompt.py`, `verify.py` first as pure functions; then `detect.py`; then `cli.py`. Write fixtures for: normal item, contaminated item, composite item, malformed JSON, truncated JSON, Llama contradicting the KB, low confidence, Hindi output.
- **Acceptance:** `uv run pytest` green; `uv run python -m pipeline.cli --item pet_bottle --lang hi` prints a verified verdict without any model file.

### WP5: Android app skeleton

- **Goal:** an installable app that opens the camera and loads both models.
- **Depends on:** WP1. **Human:** Android Studio installed; phone session to install.
- **Create:** `android/` Gradle project (Kotlin DSL), Compose UI shell with five empty screens, CameraX preview, ExecuTorch AAR dependency, model-file locator, `.github/workflows/android.yml` (assembleDebug + unit tests).
- **Steps:** min SDK 26, `abiFilters += "arm64-v8a"`; load a tiny test `.pte` at startup and show its status; settings screen shows model paths and versions.
- **Tests:** JVM unit tests run in CI; CI builds a debug APK artifact.
- **Acceptance:** CI green; APK installs and shows a live camera preview plus "models loaded".

### WP6: End-to-end MVP on the phone

- **Goal:** scan → instant verdict → streamed Llama explanation, offline.
- **Depends on:** WP2, WP3, WP4, WP5. **Human:** phone session.
- **Create:** Kotlin `vision`, `kb`, `llm` packages; Scan, Verdict and Chat screens; photo-upload path; Kotlin golden-fixture test runner.
- **Steps:**
  1. Detector: bitmap → float tensor (match Ultralytics preprocessing: letterbox, RGB, 0–1) → `Module.forward` → decode the NMS-free output → Detection records.
  2. KB: bundle `kb.sqlite` as an asset; port `kb.py` logic.
  3. Verdict card appears as soon as detection + KB finish: item name, bin colour (full-width colour band), steps.
  4. Llama: port `prompt.py` and `verify.py`; stream tokens into the card; "Ask a question" opens Chat with the same facts.
  5. Memory: pause camera analysis while Llama generates; keep one heavy model active at a time on 6 GB phones.
- **Tests:** Kotlin runs every golden fixture and must match Python's expected output.
- **Acceptance:** 10 common items work end to end in airplane mode; the verdict is visible before the first Llama token.

### WP7: Accuracy and robustness

- **Goal:** reliable on real Indian waste, and honest when unsure.
- **Depends on:** WP6. **Human:** photo collection, labelling review, Colab.
- **Create:** `data/india/README.md` (collection guide), `models/vision/attributes/` (training + export), `models/clip/export.py`, updates to `vision` package, `eval/accuracy/`.
- **Steps:**
  1. Collect 1,000–2,000 photos (iPhone is fine): each class in hand, on floor, in clutter, in poor light; include dirty and broken variants. No faces or personal documents.
  2. Label, merge, retrain YOLO26n; compare against the WP3 baseline on a held-out India test set that is never trained on.
  3. Attribute head: small classifier (MobileNetV4-class) on crops → `contaminated`, `broken`; export to `.pte`.
  4. Temporal voting over 5–10 frames; pick the confidence threshold on the validation set (report the precision/coverage trade-off).
  5. Fallback: MobileCLIP image embedding vs precomputed text embeddings of KB item names, only when the detector is below threshold.
  6. "This is wrong" button stores the crop and the corrected label locally.
- **Acceptance:** per-class precision/recall and confusion matrix committed; documented change on clear plastic/glass; gate behaviour shown on unknown objects.

### WP8: Llama specialisation

- **Goal:** fewer contradictions, valid JSON, shorter answers.
- **Depends on:** WP2, WP4. **Human:** Colab.
- **Create:** `models/llm/synth_data/generate.py`, `models/llm/lora/train.ipynb`, `eval/llm_quality/run.py`, results JSON.
- **Steps:** generate Q&A from KB entries (en + hi), auto-check each answer with `verify.py` (done: `models/llm/synth_data/generate.py`); score base Llama on the held-out cases with `eval/llm_quality/run.py` (phone answers or host `--llm-model`); LoRA fine-tune 1B on `train.jsonl` (completion-only loss); merge, re-export with `models/llm/export.py --mode int4`; re-score. Ship the fine-tune only if its **useful rate** beats base + prompt after quantisation, with a Hindi spot check.
- **Acceptance:** evaluation table committed with a clear ship / do-not-ship decision.

### WP9: Languages and voice

- **Goal:** full voice round trip offline in Hindi plus one more language; text in all six.
- **Depends on:** WP6. **Human:** native-speaker review, phone session.
- **Create:** `kb/i18n/` or per-item strings for ta/te/bn/mr, Kotlin `speech` package, `models/speech/README.md`, `docs/PIPELINES.md`.
- **Steps:**
  1. Extend KB strings; mark machine-translated text as unreviewed until checked.
  2. ASR: sherpa-onnx Android with a multi-Indic IndicConformer INT8 model (about 188 MB) or the Dolphin models; push-to-talk; show the transcript for confirmation. *Verify model licences.*
  3. TTS: sherpa-onnx with Piper/VITS voices where available; fall back to Android's built-in TTS per language. Pre-rendered voice lines for the instant verdict.
  4. Llama 3.2 officially supports English and Hindi among these. For ta/te/bn/mr, show KB text directly and use Llama only where quality is acceptable; optional translation model is a stretch goal.
  5. Load speech models on demand and release them after use.
  6. `docs/PIPELINES.md`: document vision, Llama and speech as separate pipelines (which runtime, which processor, model, quantisation).
- **Acceptance:** Hindi voice question → spoken Hindi answer in airplane mode; language switch covers all six for KB content.

### WP10: Impact and education

- **Goal:** show what the user's sorting adds up to, and teach.
- **Depends on:** WP6. **Human:** none.
- **Create:** Kotlin `impact` package (Room), Impact screen, education cards.
- **Steps:** log scans (de-duplicate repeats within 30 s); show items by stream, estimated kg diverted (count × `typical_weight_g`), contamination rate; label all estimates as estimates and cite factors; "Did you know" and myth/fact cards from KB `impact` text; optional quiz generated by Llama from KB facts.
- **Acceptance:** stats update after each scan and survive restart; works offline.

### WP11: Evaluation

- **Goal:** defensible numbers for every target.
- **Depends on:** WP7 (and WP8/WP9 if shipped). **Human:** phone session.
- **Create:** `eval/latency/` harness (in-app timing export), `eval/accuracy/report.py`, `eval/offline_check.md`, `eval/results/*.json`, plots.
- **Steps:** per-stage p50/p95 (capture, preprocess, detector, attributes, KB, Llama first token, tokens/s, TTS); cold and sustained (15-minute) runs; try an NPU/GPU delegate for vision if the phone's chipset has one, keep XNNPACK as default; battery via `dumpsys batterystats`; airplane-mode proof with a screen recording and network stats; repeat on a second phone if one can be borrowed.
- **Acceptance:** one command regenerates all report tables from `eval/results/`.

### WP12: Submission

- **Goal:** everything the brief asks for, reproducible from a fresh clone.
- **Depends on:** WP11. **Human:** video recording, final review.
- **Create:** final `README.md`, `docs/TECHNICAL_REPORT.md`, `LICENSE` (AGPL-3.0) and `THIRD_PARTY.md` (Llama 3.2 Community License, Ultralytics, datasets, speech models), GitHub Release with APK + models, `docs/DEMO_SCRIPT.md`.
- **Demo script:** mixed items on a table → instant verdicts; a composite item; a dirty item; an unknown object ("not sure"); Hindi voice question; airplane mode visible; impact screen; latency overlay.
- **Acceptance:** a teammate follows the README on a clean machine and reproduces the demo.

### WP13 (optional): Raspberry Pi 5 kiosk

- **Goal:** the same models and KB in a smart-bin form factor.
- **Depends on:** WP4, hardware. Reuse `pipeline/`; add camera loop and a simple screen UI under `pi/` (uv group `pi`). If an AI HAT+ is used, compile the detector to `.hef` on x86 and document HailoRT as a separate vision pipeline; Llama stays on ExecuTorch on the Pi CPU.

---

## 8. Order and parallel work

```
Phase 0 (done)
   ├── WP1 Llama on phone ──────────────┐
   ├── WP2 KB ──┬── WP3 detector ───────┤
   │            └── WP4 host pipeline ──┼── WP5 app skeleton ── WP6 MVP ──┬── WP7 accuracy ──┐
   │                                    │                                  ├── WP9 voice      ├── WP11 eval ── WP12 submit
   │            WP8 Llama tuning (after WP2, WP4) ─────────────────────────┤                  │
   │                                                                       └── WP10 impact ───┘
   └── WP13 Pi kiosk (optional, after WP4)
```

- **No phone needed:** WP2, WP3, WP4, WP8, and most of WP5. Do these while waiting for phone sessions.
- **Phone sessions** (batch them): WP1 benchmark, WP5 install check, WP6 demo, WP9 voice, WP11 measurements.
- **Minimum viable submission:** WP1, WP2, WP3, WP4, WP5, WP6, then a reduced WP11 and WP12. Everything else raises the score but is not required.

---

## 9. Targets and how they are measured

| Target | Goal | Measured by |
|---|---|---|
| Sorting accuracy | ≥ 92% correct stream on the held-out India test set | `eval/accuracy/report.py`; also per-class precision/recall |
| Vision inference | < 50 ms per frame for the detector on the target phone | In-app timer around `Module.forward`, p50/p95 |
| Decision latency | < 150 ms from frame to verdict card | Capture → card timestamp |
| Llama first token | < 1.5 s | `LlmCallback.onStats` |
| Llama speed | ≥ 15 tokens/s sustained | `onStats`, after 5 minutes |
| Peak memory | App stays under about 60% of device RAM | `dumpsys meminfo` |
| Cloud traffic | 0 bytes | Airplane mode + app network stats |
| LLM grounding | 0 shown contradictions (verifier), raw contradiction rate reported | `eval/llm_quality/run.py` |

If a goal is missed, report the real number and the reason; do not tune the test set.

---

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Llama too slow or too large on the borrowed phone | 1B only, shorter prompts, pre-filled system prompt, lower max tokens; verdict never waits for Llama |
| AAR and export versions drift | Keep `executorch` (Python) and `executorch-android` (Gradle) on the same version; bump together |
| YOLO26 ExecuTorch export fails on an op | Export works (checked 2026-11-17, raw head + NMS in post-processing). If a later version breaks: pin Ultralytics, try another `imgsz`, fall back to YOLO11n; last resort a classification-only model |
| AGPL-3.0 is unacceptable later | Swap the detector for an Apache-licensed one (for example an RT-DETR-family model outside Ultralytics) and retrain |
| Not enough India-specific data | Prioritise the 15 most common items; lean on the confidence gate and CLIP fallback |
| Phone unavailable near the deadline | Keep a recorded benchmark set and demo footage from every session |
| Indic ASR/TTS quality or licence problems | Android built-in recognition/TTS for that language; text-only mode |
| Memory pressure with all models | Load on demand; one heavy model at a time; smaller input size |
| Machine-translated strings wrong | Mark unreviewed strings; ship only reviewed languages by voice |

---

## 11. Where this stands against existing work

| System | What it does | What it lacks |
|---|---|---|
| DWaste (2025) | YOLOv11n on phones and edge devices, energy-aware benchmarking | No language model, no disposal reasoning, generic categories |
| ReLeaf (2026) | MobileNetV2 classifier + LLM recycling guidance | Six coarse classes; guidance not grounded in Indian rules |
| Pi smart-bin papers | YOLO on Pi 5 / Hailo, some with servo sorting | 4–6 classes, no advice, weak on clear plastic and glass |
| Consumer apps (Oscar Pocket, What I Recycle, Recycle Mate) | Photo or chat → bin | Cloud-based, Western rules, no Indian languages |

What this project adds: SWM-2026 four-stream decisions from an auditable KB; Llama on ExecuTorch as a grounded explainer; condition and composite-item handling; a calibrated "not sure"; offline voice in Indian languages; honest split between instant decision and streamed explanation.

---

## 12. Status

Update this table in the same commit that completes a WP.

| WP | Title | Status | Notes |
|---|---|---|---|
| 0 | Repo and tooling | Done | uv, CI, ExecuTorch smoke test |
| 1 | Llama on the phone | Code done | `models/llm/export.py` (verified on a tiny random-weight model), download/push scripts, host runner (`pipeline/llm.py`), phone run-list `eval/latency/llm_bench.md`. Licence accepted; prebuilt SpinQuant 1B downloaded and answering on the host. **Needs a human:** phone session |
| 2 | Knowledge base v1 | Done | 40 items, 4 streams; Hindi written but **unreviewed** (needs a native speaker) |
| 3 | Data, taxonomy, detector baseline | Code done | Scripts, notebook, `pipeline/detect.py`, export + host check all work (COCO yolo26n: same detections as PyTorch, IoU ≥ 0.94). **Needs a human:** Colab training run, Roboflow download; baseline mAP table still empty |
| 4 | Host reference pipeline | Done | `pipeline/` kb, prompt (why + tip answer format with prefilled `{"why": "`), verify, assistant (retry), cli, detect, llm; 23 golden fixtures |
| 5 | Android app skeleton | Core done | `android/core`: the KB, prompt, verifier and off-topic guard in plain Kotlin (Gradle 9.9.1, Kotlin 2.4.21, JVM 17 bytecode). It passes every golden fixture and the shared `tests/fixtures/check_cases.json` on the JVM, in CI (`kotlin-core` job). **Needs a human:** Android SDK licence for the `app` module (Compose, CameraX, ExecuTorch AAR); the app reads `kb.sqlite` with `android.database` into `KbData` |
| 6 | End-to-end MVP | Not started | |
| 7 | Accuracy and robustness | Not started | |
| 8 | Llama specialisation | Data + eval done; base scored | `generate.py`: KB-verified examples (en/hi, 5 question kinds, 6 held-out items); `eval/llm_quality/run.py`. Stock SpinQuant 1B on the host, 80 held-out cases (`eval/results/llm_quality_spinquant_stock_host*.json`, raw answers in `*.answers.jsonl`): 7-key answer 10% useful (format checks only); why + tip 97.5% on format checks alone, **61% of the 64 on-topic cases once the content checks run** (`_checked`); all 16 off-topic questions declined by the guard, no false declines. Hand reading agrees roughly (about 2 in 3 English, half of Hindi right). What the checks cannot see: nonsense that states no checkable fact, mostly Hindi "how do I prepare this?" answers. That needs the LoRA run. LoRA tooling done and dry-run verified on a tiny random Llama on the CPU (`models/llm/lora/`: train → merge/convert with a logits check against ExecuTorch's model → int4 export → runner → eval); Colab notebook `models/llm/lora/train.ipynb`. **Needs a human:** access to the gated `meta-llama/Llama-3.2-1B-Instruct` + a Colab GPU session |
| 9 | Languages and voice | Not started | |
| 10 | Impact and education | Not started | |
| 11 | Evaluation | Not started | |
| 12 | Submission | Not started | |
| 13 | Pi kiosk (optional) | Not started | |

---

## Sources

- [ExecuTorch Llama README: export and benchmarks](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md)
- [ExecuTorch: Running LLMs on Android](https://docs.pytorch.org/executorch/stable/llm/run-on-android.html)
- [ExecuTorch: Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- [Prebuilt Llama 3.2 1B SpinQuant `.pte`](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET)
- [Prebuilt Llama 3.2 1B QLoRA `.pte`](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET)
- [Arm learning path: Llama Android app with ExecuTorch and XNNPACK](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/build-llama3-chat-android-app-using-executorch-and-xnnpack/4-prepare-llama-models/)
- [Ultralytics: ExecuTorch export](https://docs.ultralytics.com/integrations/executorch)
- [Ultralytics: YOLO26 end-to-end NMS-free detection](https://docs.ultralytics.com/guides/end2end-detection)
- [Ultralytics licence](https://www.ultralytics.com/license)
- [PIB: Solid Waste Management Rules 2026](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2219676&reg=3&lang=1)
- [SBM Urban: Guidelines for Source Segregation](https://sbmurban.org/storage/app/media/pdf/Guidelines%20for%20Source%20Segregation%20of%20Municipal%20Solid%20Waste.pdf)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Indic Conformer ONNX for sherpa-onnx](https://huggingface.co/meetsync/indic-conformer-onnx-sherpa)
- [DWaste: Greener AI for Waste Sorting](https://arxiv.org/html/2510.18513v1)
- [ReLeaf: MobileNetV2 app with LLM-assisted guidance](https://ijcaonline.org/archives/volume187/number116/releaf-a-mobilenetv2-based-mobile-application-for-real-time-waste-classification-with-llm-assisted-recycling-guidance/)
