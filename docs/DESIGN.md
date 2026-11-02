# Design: On-Device Waste Segregation Assistant

**Problem statement:** Bharat AI-SoC Challenge, PS2: Edge AI for waste classification and disposal
**Required stack:** Llama (reasoning/advice) deployed on-device with ExecuTorch, on Arm hardware
**Status:** Proposal, nothing built yet. All performance figures below are targets to validate on hardware.
**Primary platform:** Arm Android smartphone (minimal hardware, see [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)). The Raspberry Pi 5 kiosk is an optional extension using the same models.

---

## 1. Problem summary

Build a complete assistant (not just a model) that:

1. Identifies and classifies waste from a live camera or an uploaded photo.
2. Gives practical, item-specific disposal instructions.
3. Explains recycling options and environmental impact.
4. Runs fully on an Arm device with no cloud inference, using **Llama via ExecuTorch**.

Judging dimensions: technical innovation (Llama + ExecuTorch on Arm), accuracy and performance, real-world viability, and social/environmental impact. Slide targets: **92% sorting accuracy, <50 ms inference latency, 0 KB cloud bandwidth**. These are goals to measure, not claims.

If a Hailo accelerator is used, vision runs through HailoRT on the Hailo NPU and Llama runs through ExecuTorch on the Arm CPU. These are separate pipelines and must be documented as such.

---

## 2. State of the art and its gaps

| Category | Examples | What they offer | Shortcomings |
|---|---|---|---|
| Academic smart bins | YOLOv8n on Pi 5 (mAP@0.5 0.888); YOLOv5s on Pi 5 AI Kit (95.2%, ~45 ms); ASU Pi 5 + Hailo (84%, 70 ms with Hailo vs 310 ms CPU) | Fast on-device detection, sometimes servo sorting | 4–6 coarse material classes, small self-collected datasets, weak on transparent plastic and glass, no reasoning/advice/education, English only |
| Consumer apps | Oscar Pocket, What I Recycle, Sort My Trash, Recycle Mate, NANDO.Eye | Photo → bin, recycling Q&A | Cloud-dependent (e.g. Gemini), Western rule sets, no offline mode, no Indian languages |
| Kiosks | Oscar 32" display, NANDO | Hold item up, see bin | Closed, cloud-backed, costly, label with little explanation |
| LLM chat demos | Generic Llama hackathon bots | Free-form Q&A | Hallucinated disposal rules, not grounded in local regulation, slow |

**Gaps nobody covers well:**

- Alignment with India's **Solid Waste Management Rules 2026** (in force 1 April 2026), which mandate four-stream source segregation: **wet, dry, sanitary, special care**.
- Composite and multilayer items: chips packets, Tetra Pak, blister packs, medicine strips.
- Item **condition and contamination** (a greasy pizza box ≠ a clean one).
- Calibrated uncertainty: existing systems always emit a label.
- No public India-specific waste image dataset was found.
- Offline Indic voice (Tamil, Telugu, Bengali, Marathi) is rare.

---

## 3. Core idea

> **The vision model perceives, the knowledge base decides, Llama explains.**

1. **Vision** determines what the item is and its condition.
2. A **deterministic knowledge base** (SWM Rules 2026, E-Waste Rules, per-city profiles) decides the stream and bin. This is instant and cannot hallucinate.
3. **Llama on ExecuTorch** explains the decision, answers follow-ups, educates, and handles languages. It is grounded on the KB entry, and a verifier overrides it if its answer contradicts the KB.

This lets us meet the latency target honestly: the bin LED and on-screen verdict fire as soon as vision + KB finish; the Llama explanation streams afterwards. The decision never waits for the LLM.

---

## 4. Architecture

```
Camera (Pi Camera Module 3 / phone camera) or uploaded photo
   │
   ▼
[Pre-processing: resize / normalise, Arm Neon via OpenCV]
   │
   ▼
┌──────────── VISION PIPELINE (HailoRT on NPU  |  ExecuTorch on Android) ────────────┐
│ A. Detector: YOLO11n/s, ~30–40 fine-grained item classes, INT8                      │
│ B. Attribute head per crop (MobileNetV4, ExecuTorch on CPU):                        │
│    material · contaminated/clean · broken/intact · hazard markings                  │
│ C. Open-set fallback (CPU): MobileCLIP zero-shot when detector confidence < τ       │
│ D. Temporal voting over 5–10 frames + confidence gate → "not sure, show the label"  │
└──────────────────────────────────────────────────────────────────────────────────────┘
   │  {item_id, attributes, confidence, bbox}
   ▼
[Knowledge base: SQLite + JSON]
   item → SWM-2026 stream → bin colour → prep steps → hazards
   → where to take it (e-waste centre, pharmacy take-back, scrap dealer)
   → impact facts · per-city profile · strings in 6 languages
   │──────► INSTANT OUTPUT: bin LED, on-screen verdict, pre-rendered voice line
   ▼
┌──────────── LLAMA PIPELINE (ExecuTorch on Arm CPU, XNNPACK + KleidiAI) ─────────────┐
│ Llama 3.2 1B Instruct (3B on phone), INT4 SpinQuant/QAT, LoRA-tuned on waste Q&A     │
│ Prompt = KB entry + user question → schema-constrained JSON → KB verifier            │
│ Pre-filled system-prompt KV cache → lower time-to-first-token; tokens stream to UI   │
└──────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[UI: touch kiosk / Android] + [Voice: ASR in, TTS out] + [Local impact log]
```

### 4.1 Waste taxonomy

Hierarchical: **item → material → SWM-2026 stream → bin**.

| SWM 2026 stream | Problem-statement category | Examples |
|---|---|---|
| Wet | Biodegradable | Food scraps, peels, garden waste |
| Dry | Recyclables (and non-recyclable dry) | PET bottles, clean paper, cans, glass; multilayer packets (dry but low recyclability) |
| Sanitary | — | Diapers, sanitary pads |
| Special care | Hazardous, E-waste | Paint, medicines, bulbs, batteries, phones, cables, sharps |

E-waste is additionally routed per the E-Waste Rules; biomedical items get special-handling guidance.

### 4.2 Structured output schema (Llama → UI)

```json
{
  "item": "PET bottle",
  "stream": "dry",
  "bin": "blue",
  "recyclable": true,
  "prep_steps": ["Empty and rinse", "Remove cap", "Crush"],
  "hazard": null,
  "drop_off": null,
  "why": "...",
  "impact_fact": "...",
  "language": "hi"
}
```

The verifier rejects or overwrites `stream`, `bin`, `hazard` and `drop_off` if they disagree with the KB.

---

## 5. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Primary hardware | Arm Android phone (6 GB+ RAM), Kotlin app + ExecuTorch AAR | No hardware to buy; camera, mic, speaker and screen built in; both vision and Llama on ExecuTorch |
| Optional extension | Raspberry Pi 5 (8 GB), optionally + AI HAT+ (Hailo) for vision | Smart-bin kiosk story if hardware becomes available; same `.pte` models and KB |
| LLM | Llama 3.2 1B/3B Instruct, INT4, exported with `export_llm`, XNNPACK + KleidiAI | Meets the mandate; ExecuTorch ≥0.7 enables KleidiAI by default and cites Pi 5 support |
| Detector | Ultralytics YOLO11n/s → `.hef` (Hailo) or `.pte` (ExecuTorch) | YOLO11 exports to Hailo; YOLO-World/YOLOE cannot be compiled for Hailo, so open-vocab stays on CPU |
| Open-set fallback | MobileCLIP-S0 via ExecuTorch | Long-tail items without retraining |
| Knowledge base | SQLite + versioned per-city JSON | Auditable, localisable, offline |
| Translation | Pre-translated KB strings for 6 languages + IndicTrans2-distilled for free text | Llama 3.2 officially covers English and Hindi, not Tamil/Telugu/Bengali/Marathi |
| ASR | IndicConformer INT8 via sherpa-onnx | Offline; per-language models fit in memory |
| TTS | VITS/MMS voices via sherpa-onnx | IndicF5 too heavy for Pi; note MMS voices are CC-BY-NC |
| Orchestration | Python (FastAPI + WebSocket) on Pi; ExecuTorch C++ Llama runner behind pybind | Fast to build; heavy paths native |
| UI | Local web app in Chromium kiosk mode; WS2812 LED strip per bin | One UI for touchscreen and physical guidance |
| Python tooling | uv (`pyproject.toml` + `uv.lock`) | Fast, locked, reproducible environments for laptop, CI and Colab |

ASR, TTS and translation are separate pipelines from Llama and will be documented as such.

---

## 6. Data plan

- **Public sources:** TACO, TrashNet, Garbage Dataset (2026), Roboflow e-waste sets. Remap all labels to our taxonomy (datasets use inconsistent class definitions).
- **Own collection:** ~1–2k photos of Indian-context items (chips packets, Tetra Pak, kulhads, newspapers, medicine strips, sanitary waste, local e-waste) under realistic lighting and backgrounds.
- **Augmentation:** copy-paste onto bin/counter backgrounds; targeted oversampling of transparent plastic and glass.
- **LLM tuning data:** synthetic waste Q&A generated from KB entries in English and Hindi, every answer checked against the KB before use.
- **Feedback loop:** user corrections logged locally for retraining (active learning).

---

## 7. Evaluation plan

| Target | How we measure |
|---|---|
| Accuracy (92%) | Held-out India-context test set; per-class precision/recall, confusion matrix; separate report for transparent plastic and glass; calibration of the confidence gate |
| Latency (<50 ms) | Per-stage timing (capture, pre-process, vision, KB, LLM TTFT, tokens/s, TTS), p50 and p95; model inference reported separately from end-to-end; decision → LED target <100 ms |
| 0 KB cloud | Demo with Wi-Fi off; packet capture as evidence |
| Resource use | Peak RAM per pipeline; power draw via USB power meter |
| Viability | Per-city rule profiles, kiosk form factor, graceful "not sure" fallback |
| Impact | On-device log: items by stream, contamination rate, kg diverted, estimated CO₂ avoided |

---

## 8. Differentiators

1. SWM Rules 2026 four-stream alignment, not a generic recycle/trash split.
2. KB-grounded decisions with Llama as explainer: no hallucinated disposal rules.
3. Condition-aware (contamination, breakage) and composite-item handling.
4. Calibrated uncertainty instead of forced labels.
5. Honest latency split: instant verdict, streamed explanation.
6. Six-language offline voice and text.
7. Runs on a phone people already own, with an optional Pi kiosk using the same core.
8. Clearly separated and documented vision and Llama pipelines.

---

## 9. Risks and open questions (validate in week 1)

- **Llama throughput on the target phone:** benchmark on day 1; use 1B on 6 GB phones, 3B only on 8 GB+.
- **Llama throughput on Pi 5 via ExecuTorch (optional kiosk):** no published figure found. llama.cpp shows roughly 4–7 tok/s for 3B; plan on 1B for the Pi.
- **Arm requirement:** the dev laptop and x86 Android emulators are not Arm; all demos and benchmarks must run on the phone.
- **AI HAT+ 2:** price sources conflict ($130 vs $200). Its on-NPU LLM support must not be used for Llama, which has to run via ExecuTorch.
- **India-specific data:** must be collected; budget time for it.
- **TTS licensing:** MMS voices are non-commercial.
- **Recommendation from the brief:** build a small working prototype first, test on the real device early, then iterate.

---

## 10. Suggested build order

1. Phone bring-up: ExecuTorch Llama 3.2 1B INT4 running in the Android demo app and benchmarked.
2. YOLO11n baseline on public data → ExecuTorch `.pte`; end-to-end camera → label on the phone.
3. KB v1 (top ~40 items, SWM-2026 streams, English + Hindi) and full-screen bin-colour output.
4. Llama grounding, JSON schema, verifier, streaming UI.
5. Own data collection, attribute head, confidence gate, MobileCLIP fallback.
6. Voice (ASR/TTS) and remaining languages.
7. Optional: Pi 5 kiosk reusing models and KB, if hardware is available.
8. Evaluation runs, technical report, demo video.

---

## Sources

- [IIETA: YOLOv8n smart bin on Pi 5](https://iieta.org/download/file/fid/199752)
- [PNU: Pi 5 AI Kit waste sorting](https://journals.pnu.edu.ua/index.php/pcss/article/download/9871/9929/32391)
- [ASU Dalbins FURI: Pi 5 + Hailo](https://forge.engineering.asu.edu/wp-content/uploads/2025/06/Dalbins_FURI.pdf)
- [YOLOE waste detection thesis](https://www.theseus.fi/handle/10024/907379)
- [Oscar Pocket (Waste Dive)](https://www.wastedive.com/news/oscar-pocket-AI-sorting/652971/)
- [What I Recycle](https://mwm.ai/apps/what-i-recycle/6443731377)
- [Sort My Trash](https://apps.apple.com/app/id6752711795)
- [Recycle Mate (RPRA)](https://rpra.ca/?p=14736)
- [NANDO.Eye](https://www.intercleanshow.com/products-and-services/nando-re-learn/products/nando-eye)
- [ExecuTorch Llama README PR](https://github.com/pytorch/executorch/pull/6486/files)
- [LiteRT INT8 LLM benchmarks (ICCVW 2025)](https://openaccess.thecvf.com/content/ICCV2025W/AIM/supplemental/Yoon_LiteRT-Optimized_INT8_LLM_ICCVW_2025_supplemental.pdf)
- [Pi 5 LLM community benchmarks](https://bestllmfor.com/best/raspberry-pi-5-llm/)
- [ExecuTorch 1.0 release notes](https://newreleases.io/project/pypi/executorch/release/1.0.0)
- [ExecuTorch + KleidiAI (PyTorch blog)](https://pytorch.org/blog/bringing-generative-ai-to-the-masses-with-executorch-and-kleidiai/)
- [Arm learning path: Llama on Pi 5 with ExecuTorch](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/rpi-llama3/)
- [Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/)
- [CNX Software: AI HAT+ 2](https://www.cnx-software.com/2026/01/15/raspberry-pi-ai-hat-2-targets-generative-ai-llm-vlm-with-hailo-10h-accelerator/?amp=1&noamp=mobile)
- [Ultralytics Hailo export](https://docs.ultralytics.com/ko/integrations/hailo)
- [iTantra offline Indic speech packs](https://huggingface.co/datasets/Mr66/itantra-packs)
- [IndicF5](https://huggingface.co/AbhishekDelMundu/IndicF5)
- [SWM Rules 2026 overview](https://vajiramandravi.com/current-affairs/solid-waste-management-rules-2026/)
- [SWM Rules 2026: four-bin segregation](https://changemakr.asia/indias-new-solid-waste-management-rules-2026-four-bin-segregation-digital-tracking-and-stricter-enforcement-begin-april-1/)
- [TACO dataset](https://arxiv.org/abs/2003.06975v2)
- [Waste datasets review](https://github.com/AgaMiko/waste-datasets-review)
- [Garbage Dataset (2026)](https://arxiv.org/html/2602.10500v1)
- [Roboflow e-waste dataset](https://universe.roboflow.com/greenflight/e-waste-qft1o)
