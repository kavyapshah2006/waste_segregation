# v0.1.0: on-device waste segregation assistant

First release for the Bharat AI-SoC Challenge (PS2). Point an Android phone at an item and get
the right bin under India's SWM Rules 2026 (green wet, blue dry, red sanitary, black special
care), how to prepare it, and why, fully offline. Llama 3.2 1B runs on the phone's Arm CPU
through ExecuTorch.

## Files

| File | What it is |
|---|---|
| `waste-segregation.apk` | The app (arm64, Android 8+), with the knowledge base and detector inside. Signed with a debug key for sideloading; not a store build |
| `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`, `tokenizer.model` | Llama 3.2 1B Instruct, SpinQuant INT4, ExecuTorch (explanations) |
| `siglip2_image_int8.pte` | SigLIP 2 image encoder (unknown-item suggestions, dirty/broken check) |
| `omnilingual_asr_300m_ctc_v2_int8.onnx`, `omnilingual_asr_tokens.txt` | Speech recognition for voice questions (sherpa-onnx) |
| `detector_yolo26n_taco.pte`, `detector_yolo26n_taco.pt` | The detector, ExecuTorch and PyTorch weights (already inside the APK) |
| `LICENSE_*.txt`, `NOTICE.txt`, `THIRD_PARTY.md` | Licences (app AGPL-3.0, Llama 3.2 Community License, ASR model) |
| `SHA256SUMS` | Checksums: `sha256sum -c SHA256SUMS` |

## Install

1. `adb install waste-segregation.apk`
2. Copy the model files to `/sdcard/Android/data/org.wastesegregation.app/files/models/`, renaming
   `omnilingual_asr_300m_ctc_v2_int8.onnx` to `model.int8.onnx` and `omnilingual_asr_tokens.txt`
   to `tokens.txt` (for example `adb push <file> <that folder>`). Building from the repo instead:
   the README's "Run it on a phone" steps download the models and `./scripts/install_app.sh`
   installs everything.
3. Open the app; Settings lists every model as ready.

## Measured (details: `docs/TECHNICAL_REPORT.md`, `eval/REPORT.md`)

- **Llama answers**: 96.6% useful on 88 held-out questions with the shipped prompt (two worked
  examples, rules, a sampled retry when the knowledge-base checks fail), up from 69.3%. Every
  answer is checked against the knowledge base; off-topic questions are declined (126 of 126).
- **Detector** (YOLO26n, 9.8 MB, TACO test split): mAP50 0.305, mAP50-95 0.237. At the 0.5
  confidence gate it gives a verdict on 66% of images and 88.4% of verdicts land in the right
  stream; otherwise it says "not sure" and offers three suggestions.
- **Phone speed, memory and battery**: see the technical report §5.6.

## Known limits

The detector knows 14 items and was trained on TACO (European litter photos), not Indian
household waste; the other 39 knowledge-base items are reached by suggestions, voice or search.
Hindi text has not been reviewed by a native speaker; voice was measured on synthetic speech.
One city profile.

## Licences

The app is AGPL-3.0 (it builds on Ultralytics YOLO). **Built with Llama**: Llama 3.2 is licensed
under the Llama 3.2 Community License, copy included. TACO images used for training have mixed
licences (`data/taco_licences.md`). Third-party list: `THIRD_PARTY.md`.
