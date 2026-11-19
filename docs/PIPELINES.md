# Pipelines

The app runs five separate on-device pipelines. Each has its own runtime, model and input; none
talks to a network. The knowledge base (KB) decides the bin; vision only perceives, and Llama only
explains. This file lists what runs where, so the speed and memory numbers in
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) can be read per pipeline.

All of them run on the phone's **Arm CPU**. No NPU, GPU or Hailo accelerator is used today (a GPU
or NPU option for vision is to be tried on the phone, step 62 of the plan); if one is added, it is
documented here as its own pipeline.

| # | Pipeline | Runtime on the phone | Model file | Size | Precision | Runs when |
|---|---|---|---|---|---|---|
| 1 | Detector | ExecuTorch, XNNPACK backend | YOLO26n fine-tuned on our taxonomy (`model.pte`) | ~9.4 MB | fp32 | every camera frame while scanning (paused while Llama writes) |
| 2 | Unknown-item fallback | ExecuTorch, XNNPACK backend | SigLIP 2 base image encoder (`siglip2_image_int8.pte`) + KB text embeddings (`text_embeddings.json`) | 94 MB + 0.3 MB | int8 (dynamic) | before each verdict (agreement) and when the detector is below the confidence gate |
| 3 | Knowledge base | Android SQLite | `kb.sqlite` (built by `kb/build_kb.py`) | < 1 MB | n/a | after every stable detection or voice match |
| 4 | Llama explanation | ExecuTorch `LlmModule`, XNNPACK backend | Llama 3.2 1B Instruct, prebuilt `INT4_EO8` `.pte` (SpinQuant or QLoRA; chosen by the step E experiments) | ~1.1 GB | 4-bit weights | after the verdict card is shown, or when the user asks a question |
| 5 | Speech | sherpa-onnx (ONNX Runtime) for speech → text; Android's built-in text-to-speech | Meta Omnilingual ASR 300M CTC v2 | 366 MB | int8 | push-to-talk only; loaded on demand and released after use |

Laptop numbers in this repo are functional checks only. Every speed, memory and battery number
comes from the phone.

## 1. Detector

- **Input:** camera frame (RGB), letterboxed to 640×640 with grey (114) padding, scaled to 0–1.
- **Output:** raw predictions `(1, 4 + classes, 8400)`. The Ultralytics ExecuTorch exporter drops
  YOLO26's NMS-free head, so decoding and NMS run in code.
- **Post-processing:** confidence gate (0.5, to be tuned on the validation split), then the frame
  voter: an item is stable once it wins 3 of the last 5 frames.
- **Code:** Python reference `pipeline/detect.py`, `pipeline/stability.py`; Kotlin
  `android/core/.../Detect.kt`, `Stability.kt` (same fixtures: `tests/fixtures/detect_cases.json`,
  `stability_cases.json`).
- **Export:** `models/vision/export_executorch.py`; details in [models/vision/README.md](../models/vision/README.md).

## 2. Unknown-item fallback and agreement check

- **When it runs:** when the detector is below the confidence gate, and also before every verdict:
  a stable detector item becomes a verdict only if this encoder ranks it in its top 3 for the
  same frame (agreement rule, `confirms()`); otherwise its three suggestions are shown instead.
  Measured on 2,371 BDWaste photos: right bin 91.9% with the rule, 75.2% without
  ([TECHNICAL_REPORT.md §5.3](TECHNICAL_REPORT.md)).
- **Input:** the whole frame (agreement, fallback) or the detector's crop (dirty/broken check),
  resized to 224×224, normalised with `(x − 0.5) / 0.5`.
- **Output:** an L2-normalised 768-number embedding, compared by cosine with one precomputed text
  embedding per KB item. The top 3 items are offered to the user to confirm; nothing is decided
  automatically.
- **Code:** `pipeline/fallback.py`, Kotlin `FallbackRank.kt`; export and accuracy in
  [models/fallback/README.md](../models/fallback/README.md).

## 3. Knowledge base

- Pure lookup: item + condition (dirty, broken) + city → stream, bin colour, steps, hazard,
  drop-off, notes. This is the only part that decides where an item goes.
- **Code:** `kb/build_kb.py` (validation + SQLite), `pipeline/kb.py`, Kotlin `KnowledgeBase.kt`.

## 4. Llama explanation

- **Input:** a Llama 3.2 chat prompt holding the KB facts, the item's condition, the language and
  the question (`pipeline/prompt.py`, Kotlin `Prompt.kt`). The system part never changes, so its
  KV cache can be filled once.
- **Output:** `{"why": ..., "tip": ...}`. The verifier (`pipeline/verify.py`, Kotlin `Verify.kt`)
  drops any text that contradicts the KB (wrong bin, wrong stream, wrong recyclable claim, unsafe
  advice) and falls back to KB text if the JSON is broken.
- **Off-topic questions** are declined before Llama runs (`pipeline/assistant.py`, Kotlin
  `Assistant.kt`).
- **Never blocks the verdict:** the card shows the KB answer first; Llama's words stream in after.
- Model and prompt choices and their measured quality: [eval/llm_quality/](../eval/llm_quality/)
  and `eval/results/`.

## 5. Speech

- **Speech → text:** sherpa-onnx with Omnilingual ASR 300M CTC v2 int8, push-to-talk.
  The transcript is matched to KB items by pronunciation (`pipeline/voice.py`, Kotlin `Voice.kt`);
  when the match is not confident, the top items are shown for the user to pick.
- **Text → speech:** Android's built-in engine with its offline Hindi voice. No voice model ships
  in the app (Piper Hindi voices are non-commercial or have an unverified licence).
- Details and measurements: [models/speech/README.md](../models/speech/README.md).

## Memory policy

- One heavy model at a time on phones with 6 GB of RAM or less: camera analysis (detector) pauses
  while Llama generates; the speech model and the fallback encoder load on demand and are released
  after use.
- Targets (FuturePlan section 9): app under about 60% of device RAM, detector under 50 ms per
  frame, verdict under 150 ms, Llama first token under 1.5 s, at least 15 tokens/s.

## Measured on the phone

To be filled from `eval/results/` after the phone sessions (steps 45, 61, 63, 65).

| Pipeline | p50 | p95 | Peak memory | Notes |
|---|---|---|---|---|
| Detector | pending | pending | pending | |
| Fallback encoder | pending | pending | pending | |
| KB lookup | pending | pending | pending | |
| Llama first token / tokens per second | pending | pending | pending | |
| Speech → text | pending | pending | pending | |
