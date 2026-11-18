# Roadmap

Step-by-step path from an empty repo to the full solution in [DESIGN.md](DESIGN.md), using the minimal hardware in [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md): **an Arm Android phone + the dev laptop**, with an optional Raspberry Pi 5 kiosk at the end.

For step-by-step implementation (interfaces, files, tests, acceptance checks), use [FuturePlan.md](FuturePlan.md); it wins where it differs from this file.

> **Changed since this roadmap was written:** detector is YOLO26n (not YOLO11), with NMS in app code; the fallback is SigLIP 2 (not MobileCLIP) under `models/fallback/`; speech recognition is Omnilingual ASR (not IndicConformer) and text-to-speech is Android's own; the repo is AGPL-3.0. The repository layout below is the original sketch: the real tree adds `pipeline/` (Python reference of the app logic) and `android/core/` (its tested Kotlin port). Progress per work package: [FuturePlan.md, section 12](FuturePlan.md#12-status).

Each phase lists its goal, tasks, exit criteria, edge cases, advantages, disadvantages and alternatives. Phases 0–4 produce a working MVP; everything after improves it. Follow the brief's advice: **get something working on the real phone early, then iterate.**

---

## Conventions

### Python tooling: uv

All Python code (model export, training, data, KB validation, evaluation, optional Pi app) is managed with **[uv](https://docs.astral.sh/uv/)**.

| Need | Command |
|---|---|
| Create/sync the environment | `uv sync` |
| Add a dependency | `uv add <pkg>` (dev-only: `uv add --dev <pkg>`) |
| Run a script | `uv run python models/llm/export.py` |
| Run tests | `uv run pytest` |
| Pin Python | `uv python pin 3.11` (`.python-version` committed) |

Rules:

- One root `pyproject.toml` + committed `uv.lock`. Optional groups for heavy extras (e.g. `train`, `speech`, `pi`) so a plain `uv sync` stays light.
- Never `pip install` into the project env. On Colab/Kaggle: `pip install uv` once, then `uv pip install --system -r <exported requirements>` or `uv sync`.
- CPU-only `torch` for the laptop; CUDA `torch` for Colab via a `[[tool.uv.index]]` entry scoped to the `train` group.
- `executorch` and `torch` versions are pinned together (ExecuTorch releases target a specific torch version).

The Android app uses Gradle (Kotlin); uv covers everything Python.

### Other conventions

- Model binaries (`.pte`, `.hef`, checkpoints) are **not committed**. They are published as GitHub Release assets and fetched by `scripts/download_models.sh`.
- Raw datasets are not committed; only scripts, label maps and small fixtures.
- Every benchmark result is committed as JSON/CSV under `eval/results/` with device, model and commit hash.

---

## Repo: start vs end

### Start (today)

```
waste_segregation/
└── docs/
    ├── DESIGN.md
    ├── HARDWARE_REQUIREMENTS.md
    └── roadmap.md
```

### End (target)

```
waste_segregation/
├── README.md                      # Quick start, demo GIF, architecture summary
├── LICENSE
├── pyproject.toml                 # uv project: deps + optional groups (train, speech, pi)
├── uv.lock
├── .python-version
├── .gitignore                     # models/, data/raw/, build outputs
├── .github/workflows/ci.yml       # uv sync, lint, tests, KB schema validation, Android build
│
├── docs/
│   ├── DESIGN.md
│   ├── HARDWARE_REQUIREMENTS.md
│   ├── roadmap.md
│   ├── PIPELINES.md               # Vision vs Llama vs speech pipelines, documented separately
│   ├── TECHNICAL_REPORT.md        # Final submission report
│   └── images/                    # Diagrams, screenshots
│
├── kb/                            # Knowledge base (source of truth for disposal decisions)
│   ├── schema.json
│   ├── items/*.json               # One file per item: stream, bin, prep steps, hazards, impact
│   ├── cities/*.json              # Per-city overrides (bin colours, drop-off points)
│   ├── i18n/{en,hi,ta,te,bn,mr}.json
│   └── build_kb.py                # Validates JSON → builds kb.sqlite for the app
│
├── data/
│   ├── taxonomy.yaml              # Our classes + mapping from TACO/TrashNet/GD/Roboflow labels
│   ├── scripts/                   # Download, remap, dedupe, split, augment
│   └── fixtures/                  # Small sample images for tests
│
├── models/
│   ├── llm/
│   │   ├── export.py              # Llama 3.2 → ExecuTorch .pte (INT4, XNNPACK)
│   │   ├── prompts/               # System prompt, JSON schema
│   │   ├── synth_data/            # KB → Q&A generation + verification
│   │   └── lora/                  # Fine-tuning config/notebook
│   ├── vision/
│   │   ├── train.py               # YOLO11 training (Colab)
│   │   ├── attributes/            # Clean/dirty, broken/intact head
│   │   ├── export_executorch.py   # → .pte for Android
│   │   └── export_hailo.md        # Optional Pi + AI HAT+ path
│   ├── clip/export.py             # MobileCLIP fallback → .pte
│   └── speech/                    # ASR/TTS model fetch + config (sherpa-onnx)
│
├── android/                       # Kotlin app (Gradle)
│   └── app/src/main/java/.../
│       ├── camera/                # CameraX capture + preprocessing
│       ├── vision/                # Detector, attribute head, temporal voting, confidence gate
│       ├── kb/                    # SQLite lookup + city profile
│       ├── llm/                   # ExecuTorch Llama runner, prompt builder, JSON verifier
│       ├── speech/                # ASR + TTS
│       ├── impact/                # Local log, stats, education content
│       └── ui/                    # Screens: scan, verdict, chat, impact, settings
│
├── eval/
│   ├── accuracy/                  # Per-class precision/recall, confusion matrix
│   ├── latency/                   # Per-stage + end-to-end timing harness
│   ├── llm_quality/               # Grounding/contradiction tests against KB
│   ├── offline_check.md           # Airplane-mode procedure
│   └── results/                   # Committed benchmark outputs
│
├── pi/                            # Optional Raspberry Pi 5 kiosk (Python, uv group "pi")
│
├── scripts/
│   ├── setup_dev.sh               # Toolchain setup in WSL
│   └── download_models.sh         # Fetch .pte/.hef from GitHub Releases
│
└── tests/                         # uv run pytest: KB, taxonomy, prompt/verifier, export smoke tests
```

---

## Phase 0: Foundation and tooling

**Goal:** A reproducible dev environment and repo skeleton.

**Tasks**
- `uv init`, pin Python, add base deps (`torch` CPU, `executorch`, `pytest`, `ruff`). Commit `uv.lock`.
- Repo skeleton, `.gitignore`, README stub, CI running `uv sync` + `ruff` + `pytest`.
- Android Studio + NDK + `adb`; enable developer mode and USB debugging on the phone.
- Accept the Meta Llama licence on Hugging Face and get access to Llama 3.2 1B/3B Instruct.
- Record the phone's specs (SoC, RAM, Android version, CPU features such as i8mm/dotprod) in `docs/`.

**Exit criteria:** `uv sync && uv run pytest` passes in CI; phone visible in `adb devices`.

**Edge cases**
- `adb` from WSL can't see USB devices by default → use Windows `adb.exe` or `usbipd-win` passthrough.
- ExecuTorch build tooling targets Linux/macOS → do all Python and native builds in WSL, not native Windows.
- Llama access request on Hugging Face may take time → request on day 1.
- `torch`/`executorch` version mismatch → pin both in `pyproject.toml`.

**Advantages:** uv gives fast, locked, reproducible envs for every teammate and Colab; CI catches breakage early.
**Disadvantages:** Some upfront time before any visible progress; WSL ↔ Windows USB friction.
**Alternatives:** Poetry or pip + venv (slower, weaker locking); conda (heavier, unnecessary without GPU on laptop); Docker dev container (more isolation, more setup).

---

## Phase 1: Llama on the phone (the mandatory core)

**Goal:** Llama 3.2 running on the phone via ExecuTorch, with real numbers.

**Tasks**
- Export Llama 3.2 1B Instruct to `.pte` (INT4 weights, XNNPACK, KleidiAI enabled) with `uv run` + ExecuTorch's LLM export.
- Build/run ExecuTorch's Android Llama demo app; push the `.pte` and tokenizer with `adb`.
- Benchmark: load time, time-to-first-token, tokens/s, peak RAM, temperature after 5 minutes. Save to `eval/results/`.
- If the phone has 8 GB+, repeat with 3B and decide which model to use.

**Exit criteria:** A prompt about waste returns a coherent answer on-device in airplane mode; benchmark JSON committed.

**Edge cases**
- Out-of-memory on 6 GB phones → 1B only; shorter max context; close background apps.
- Thermal throttling during long sessions → measure sustained, not just first-run, speed.
- Older CPUs without i8mm/dotprod → slower kernels; note in results.
- Android killing the app in background → keep model in a foreground service while active.

**Advantages:** De-risks the hard requirement first; numbers drive every later decision (model size, prompt length, UX).
**Disadvantages:** No visible waste functionality yet; export pipeline can be fiddly.
**Alternatives:** llama.cpp or MLC (faster to try, but violate the ExecuTorch requirement); vendor NPU backends (Qualcomm QNN, MediaTek) for Llama (faster, but device-specific and more complex; revisit in Phase 9).

---

## Phase 2: Knowledge base v1

**Goal:** A validated, offline source of truth for disposal decisions.

**Tasks**
- Define `kb/schema.json`: item id, aliases, material, SWM-2026 stream (wet/dry/sanitary/special care), bin colour, recyclable, prep steps, hazards, drop-off type, impact facts, typical weight.
- Write ~40 high-frequency items (PET bottle, chips packet, Tetra Pak, newspaper, food scraps, battery, phone, CFL bulb, medicine strip, sanitary pad, etc.).
- One default city profile + English and Hindi strings.
- `kb/build_kb.py` validates and compiles to `kb.sqlite`; tests run in CI.

**Exit criteria:** `uv run python kb/build_kb.py` produces a valid DB; every item has a stream and en/hi text.

**Edge cases**
- Composite items (Tetra Pak, blister packs) → model as their own items, not as "paper" or "plastic".
- Same item, different city rules → city profile overrides defaults.
- Condition-dependent items (greasy vs clean cardboard) → rules keyed on attributes.
- Sources disagree → cite the SWM Rules 2026 / E-Waste Rules text in each item and prefer official sources.

**Advantages:** Deterministic, auditable, no hallucinations; works instantly and offline; easy to extend per city.
**Disadvantages:** Manual effort; coverage is limited to listed items (long tail handled later by fallback).
**Alternatives:** Let Llama decide disposal directly (rejected: hallucination risk); RAG over raw PDFs of the rules (less structured, slower, harder to verify).

---

## Phase 3: Vision baseline

**Goal:** A detector that runs on the phone and outputs our item classes.

**Tasks**
- `data/taxonomy.yaml`: our ~30–40 classes and a mapping from TACO, TrashNet, Garbage Dataset and Roboflow e-waste labels.
- Data scripts: download, remap, dedupe, split (train/val/test), augment.
- Train YOLO11n on Colab/Kaggle (uv `train` group with CUDA torch).
- Export to ExecuTorch `.pte`; run on the phone via CameraX; measure per-frame latency.

**Exit criteria:** Live camera on the phone shows boxes with our class names; baseline mAP and latency committed.

**Edge cases**
- Inconsistent labels across datasets → explicit mapping; drop ambiguous classes rather than guess.
- Class imbalance (lots of bottles, few batteries) → oversampling, class weights.
- Unsupported ops in export (e.g. NMS) → keep NMS in Kotlin post-processing.
- Clear plastic and glass → track their accuracy separately from day one.
- Licence of each dataset → record in `data/taxonomy.yaml`; avoid non-commercial sets for anything shipped.

**Advantages:** Detection handles multiple items in one frame; same ExecuTorch runtime as Llama on the phone.
**Disadvantages:** Public data is mostly Western; India-specific items are under-represented until Phase 5.
**Alternatives:** Image classification only (simpler, but one item per photo); Google ML Kit / TFLite (mature, but weakens the ExecuTorch story); open-vocabulary YOLOE/YOLO-World (flexible, but heavier and slower on phone).

---

## Phase 4: End-to-end MVP

**Goal:** The full loop on the phone: camera → item → bin decision → Llama explanation.

**Tasks**
- App flow: scan screen → instant verdict card (stream, bin colour, prep steps from KB) → streaming Llama explanation below it.
- Prompt builder: KB entry + attributes + user question → Llama; JSON output schema.
- Verifier: parse JSON; override `stream`/`bin`/`hazard`/`drop_off` with KB values if they disagree; fall back to KB text if parsing fails.
- Pre-fill the system prompt once to cut time-to-first-token.
- Upload-a-photo mode in addition to live camera.

**Exit criteria:** Demo of 10 common items end-to-end in airplane mode; verdict appears before Llama finishes.

**Edge cases**
- Malformed or truncated JSON → retry once with a shorter prompt, then KB-only answer.
- Llama contradicts the KB → KB wins; log the case for fine-tuning data.
- Camera + Llama memory pressure → pause camera while generating; release detector buffers.
- No item detected → prompt the user to reframe or type the item name.

**Advantages:** A working, demo-able product early; honest latency split (instant decision, streamed explanation).
**Disadvantages:** Accuracy and coverage still limited; English/Hindi only.
**Alternatives:** Chat-only MVP without vision (faster to build, but misses the core requirement); vision-only MVP without Llama (misses the mandatory Llama component).

---

## Phase 5: Accuracy and robustness

**Goal:** Reliable results on real Indian waste.

**Tasks**
- Collect ~1–2k photos of Indian-context items with phones (varied lighting, backgrounds, in-hand, cluttered).
- Label (Roboflow/CVAT); retrain YOLO11 on combined data.
- Attribute head on crops: clean/dirty, broken/intact, hazard markings.
- Temporal voting over 5–10 frames; calibrated confidence gate ("not sure, show me the label").
- MobileCLIP fallback for low-confidence/unknown items, mapped to the nearest KB item.
- In-app "this is wrong" correction, saved locally for retraining.

**Exit criteria:** Per-class precision/recall on our held-out India test set; documented improvement on clear plastic/glass.

**Edge cases**
- Faces or personal info in photos → crop/blur before storing; keep data local.
- Overlapping or stacked items → per-box verdicts; ask to separate if boxes overlap heavily.
- Items held in hand → include hand-held images in training data.
- Over-confident wrong answers → calibrate thresholds on the validation set, not by eye.

**Advantages:** Directly targets the accuracy judging criterion and the known weakness of existing systems.
**Disadvantages:** Data collection and labelling take real time.
**Alternatives:** Synthetic data only (cheaper, weaker realism); a single larger detector instead of detector + attribute head (simpler, slower, less explainable).

---

## Phase 6: Llama specialisation

**Goal:** Better, shorter, more grounded answers from Llama.

**Tasks**
- Generate synthetic Q&A from KB entries (English + Hindi); verify every answer against the KB automatically.
- LoRA fine-tune Llama 3.2 1B (and 3B if used) on Colab; re-export with quantisation-aware settings.
- `eval/llm_quality/`: contradiction rate vs KB, JSON validity rate, answer length, Hindi quality spot checks.
- Compare base + prompt vs fine-tuned; keep whichever wins.

**Exit criteria:** Fine-tuned model beats base on contradiction rate and JSON validity without slowing down.

**Edge cases**
- Quantisation erases fine-tune gains → use QAT or re-check after INT4 export.
- Overfitting to KB phrasing → mix in general waste-education questions.
- Users asking unrelated questions → polite refusal pattern in training data.

**Advantages:** Stronger "effective use of Llama" story; fewer verifier overrides; shorter outputs = lower latency.
**Disadvantages:** Training time and Colab limits; another export to maintain.
**Alternatives:** Prompting + KB grounding only (simpler, already works from Phase 4); larger model without fine-tuning (slower on phone).

---

## Phase 7: Multilingual and voice

**Goal:** Usable by people who don't read English, by voice or text.

**Tasks**
- KB strings for Tamil, Telugu, Bengali, Marathi (human-reviewed where possible).
- IndicTrans2-distilled for free-form chat translation in languages Llama doesn't officially support.
- ASR: IndicConformer INT8 via sherpa-onnx; TTS: VITS voices via sherpa-onnx.
- Language picker + auto-detect from speech.
- Document speech/translation as separate pipelines from Llama in `docs/PIPELINES.md`.

**Exit criteria:** Full voice round-trip in Hindi and at least one more language, offline.

**Edge cases**
- Code-mixed speech (Hinglish) → test explicitly; prefer multilingual ASR models.
- Noisy environments → push-to-talk; show transcript for confirmation.
- Memory with all models loaded → load speech/translation models on demand and unload after use.
- TTS licence (MMS voices are non-commercial) → note in report; swap if needed.

**Advantages:** Inclusivity is an explicit focus of the brief; strong social-impact angle.
**Disadvantages:** Many models to manage; translation quality varies by language.
**Alternatives:** Android's built-in offline speech recognition/TTS (easier, but less control and depends on Google packs); Whisper-small (good multilingual, heavier on phone).

---

## Phase 8: Impact and education

**Goal:** Turn sorting into measurable impact and learning.

**Tasks**
- Local log: items scanned, stream, contamination flags, corrections.
- Impact screen: items by stream, estimated kg diverted (count × typical weight, labelled as estimate), estimated CO₂ avoided.
- Education: "why" cards, myth vs fact, short quizzes generated from the KB and explained by Llama.

**Exit criteria:** Impact screen and education content work offline and update after each scan.

**Edge cases**
- Estimates mistaken for measurements → always label them; cite emission factor sources.
- Repeated scans of the same item → de-duplicate within a short time window.

**Advantages:** Addresses the social/environmental impact criterion; keeps users engaged.
**Disadvantages:** Estimates are approximate without a scale.
**Alternatives:** Cloud dashboard (breaks the offline promise); skip impact features (weaker judging story).

---

## Phase 9: Optimisation and evaluation

**Goal:** Defensible numbers for every target.

**Tasks**
- Per-stage timing (capture, preprocess, detector, attribute head, KB, Llama TTFT, tokens/s, TTS), p50/p95.
- Try the phone's NPU/GPU backend for vision (QNN/MediaTek/Vulkan) vs XNNPACK.
- Accuracy report on the held-out test set; confusion matrix.
- Offline proof: airplane mode demo + network log showing no traffic.
- Battery/thermal: `dumpsys batterystats`, 15-minute sustained use test.
- Test on a second, weaker phone.

**Exit criteria:** `eval/results/` holds all numbers; report tables generated from them.

**Edge cases**
- Benchmarks on a hot phone → cool-down between runs; report sustained and cold numbers.
- Vendor backend not available on the test phone → XNNPACK fallback stays default.

**Advantages:** Honest, reproducible evidence for the accuracy/performance criterion.
**Disadvantages:** Time-consuming; results depend on the specific phone.
**Alternatives:** Report only model inference time (misleading; the brief asks for end-to-end).

---

## Phase 10 (optional): Raspberry Pi 5 kiosk

**Goal:** Smart-bin form factor reusing the same models and KB.

**Tasks**
- `pi/` Python app (uv group `pi`): camera → detector → KB → Llama (ExecuTorch on Arm CPU) → screen/voice.
- Optional AI HAT+: compile detector to `.hef` on the laptop (WSL, x86_64), run via HailoRT; document as a separate pipeline.

**Exit criteria:** Same demo flow on the Pi; separate pipeline doc if Hailo is used.

**Edge cases**
- Pi CPU vision too slow → lower input size or add AI HAT+.
- `rpi_ws281x` unsupported on Pi 5 → drive LEDs via a Pico or SPI if LEDs are added.

**Advantages:** Real-world municipal/community deployment story.
**Disadvantages:** Needs hardware; extra maintenance.
**Alternatives:** Mount the phone on a stand as a "kiosk" (zero extra hardware).

---

## Phase 11: Submission deliverables

**Goal:** Everything the brief requires, polished.

**Tasks**
- `README.md`: setup with `uv sync`, model download, APK install, architecture diagram.
- `docs/TECHNICAL_REPORT.md`: architecture, pipelines (documented separately), data, training, quantisation, results, limitations.
- Release APK + model files on GitHub Releases.
- Demo video: live scan of mixed items, Hindi voice query, airplane mode on screen, impact screen, latency overlay.

**Exit criteria:** A fresh clone + README steps reproduce the demo.

**Edge cases**
- Large model files → GitHub Release assets, not git.
- Reviewers without the same phone → include screen recordings and benchmark tables.

**Advantages:** Complete, reproducible submission aligned with all four judging criteria.
**Disadvantages:** Documentation and video take longer than expected; start drafting early.
**Alternatives:** Notebook-only submission (fails the "working application" requirement).

---

## Cross-cutting risks

| Risk | Mitigation |
|---|---|
| Llama too slow on the target phone | 1B INT4, shorter prompts, pre-filled system prompt, instant KB verdict so UX doesn't wait |
| Not enough Indian training data | Start collecting in Phase 3; augment; MobileCLIP fallback for gaps |
| Memory pressure with many models | Load on demand; one heavy model active at a time |
| Licence issues (Llama, datasets, TTS voices) | Track licences in `data/taxonomy.yaml` and the report; avoid non-commercial assets where possible |
| Scope creep | Phases 0–4 are the MVP; everything else is ordered by judging impact |
