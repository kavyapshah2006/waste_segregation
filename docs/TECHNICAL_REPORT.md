# Technical report: on-device waste segregation assistant

Bharat AI-SoC Challenge, PS2. **Draft outline**: sections marked *pending* need the trained detector or phone sessions. All numbers come from [eval/REPORT.md](../eval/REPORT.md), which `uv run python eval/report.py` regenerates from `eval/results/`; do not type numbers here by hand.

## 1. Problem and goal

- Household waste in India must be split into four streams under the Solid Waste Management Rules, 2026: wet (green bin), dry (blue), sanitary (red), special care (black). People get it wrong most often on mixed, dirty or hazardous items.
- Goal: point a phone at an item and get the right bin, how to prepare it, and why, fully offline, in Indian languages, using Llama through ExecuTorch on an Arm phone.

## 2. Design principle: vision perceives, the knowledge base decides, Llama explains

- The bin and the steps never come from a neural network's guess: a curated knowledge base (53 items, condition rules for dirty/broken items, city overrides, sourced from the SWM Rules and US EPA guidance) decides.
- Llama 3.2 1B writes only the explanation (`why`) and a `tip`. A verifier checks every answer against the KB and drops wrong text: wrong bin, negated bin or stream ("do not put it in the blue bin" when blue is right), wrong recyclable/compostable claim, a KB step turned into "don't", or an unsafe instruction (drain, burning). Off-topic questions are declined before Llama is called.
- Figure: pipeline diagram (camera → detector → frame voting → KB → prompt → Llama → verifier → answer card; voice and unknown-item fallback as side paths).

## 3. System

| Stage | Model / method | On the phone |
|---|---|---|
| Detection | YOLO26n (Ultralytics), exported to ExecuTorch XNNPACK; NMS in app code (the export drops the NMS-free head) | *pending: trained weights* |
| Stability | Confidence gate 0.5; verdict once one item wins 3 of the last 5 frames | Kotlin core |
| Unknown items | SigLIP 2 base image encoder, int8 `.pte` (94 MB), matched to KB text embeddings; top 3 offered to the user | Kotlin core ranking |
| Decision | Knowledge base (`kb.sqlite`), city profile | Kotlin core |
| Explanation | Llama 3.2 1B SpinQuant INT4 `.pte` (1.1 GB); prompt prefilled with `{"why": "`; verifier | Kotlin core prompt/verifier |
| Voice | Omnilingual ASR 300M CTC int8 (sherpa-onnx) → pronunciation-key item matcher; Android text-to-speech for Hindi | Kotlin core matcher |
| Impact | Scan log, kg diverted (estimate), streak; KB-only education cards and quiz | Kotlin core |

Parity: every decision function exists in Python (reference, laptop) and Kotlin (`android/core`), and both pass the same golden fixtures and case files in CI.

## 4. Results

### 4.1 Sorting accuracy (target 92%), *pending*
Stream accuracy on the held-out India test split, mAP50, per-class AP, confusion matrix (`eval/accuracy/report.py`).

### 4.2 Llama answer quality (laptop, same `.pte` as the phone)
- Asking Llama to restate all facts (7 JSON fields) gave 10% useful answers: Hindi answers ran out of tokens. Asking only for `why` + `tip`, with the answer start prefilled: 97.5% valid JSON.
- With the content checks: 61% of on-topic answers are both valid and correct; every off-topic question declined. The user always sees the KB's bin and steps; a failed explanation only drops the explanation.
- *Pending*: LoRA fine-tune on 470 KB-written examples (tooling verified end to end on a tiny model).

### 4.3 Unknown-item fallback
83.5% top-3 on 133 TACO object crops, int8 within 1 point of fp32. *Pending*: India photo set.

### 4.4 Hindi voice
Omnilingual ASR + pronunciation matching: 92.5% of item names and 87.5% of names inside sentences matched first time on synthetic speech; no false confident matches. *Pending*: native speakers.

### 4.5 Speed, memory, battery, offline proof, *pending*
Per-stage p50/p95 on the phone, Llama time to first token and tokens/s, sustained speed after 5 minutes, peak memory, airplane-mode screen recording (`eval/latency/llm_bench.md`).

## 5. Lessons

- Constrain the LLM to what only it can add; let deterministic code own the facts.
- Measure content, not format: a format-only "useful rate" of 97.5% hid answers that were wrong in a third of cases.
- Licences decide architecture: AGPL detector → AGPL repo; research-only MobileCLIP weights → SigLIP 2; non-commercial Piper voices → Android TTS.

## 6. Limitations and next steps

Unreviewed Hindi KB text; no Tamil/Telugu/Bengali/Marathi KB text yet; detector not trained on India photos yet; voice numbers are from synthetic speech; single city profile.

## Appendix

Licences: [THIRD_PARTY.md](../THIRD_PARTY.md). Built with Llama.
