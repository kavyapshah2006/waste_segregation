# SwachhSathi v0.1.0: on-device waste segregation assistant

First release for the Bharat AI-SoC Challenge (PS2). Point an Android phone at an item and get the
right bin under India's SWM Rules 2026 (green wet, blue dry, red sanitary, black special care),
how to prepare it, and why, fully offline. Llama 3.2 1B, the YOLO26n detector and the SigLIP 2
encoder run on the phone's Arm CPU through ExecuTorch; the app has no internet permission.

## Files

| File | What it is |
|---|---|
| `swachhsathi.apk` | The app (arm64, Android 8+), with the knowledge base and detector inside. Signed with a debug key for sideloading; not a store build |
| `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`, `tokenizer.model` | Llama 3.2 1B Instruct, SpinQuant INT4, ExecuTorch (explanations) |
| `siglip2_image_int8.pte` | SigLIP 2 image encoder (agreement check, suggestions, dirty/broken check) |
| `omnilingual_asr_300m_ctc_v2_int8.onnx`, `omnilingual_asr_tokens.txt` | Speech recognition for voice questions (sherpa-onnx) |
| `detector_yolo26n_taco.pte`, `detector_yolo26n_taco.pt` | The detector, ExecuTorch and PyTorch weights (already inside the APK) |
| `LICENSE_*.txt`, `NOTICE.txt`, `THIRD_PARTY.md` | Licences (app AGPL-3.0, Llama 3.2 Community License, ASR model) |
| `SHA256SUMS` | Checksums: `sha256sum -c SHA256SUMS` |

## Install

1. `adb install swachhsathi.apk`
2. Copy the model files to `/sdcard/Android/data/org.wastesegregation.app/files/models/`, renaming
   `omnilingual_asr_300m_ctc_v2_int8.onnx` to `model.int8.onnx` and `omnilingual_asr_tokens.txt`
   to `tokens.txt` (for example `adb push <file> <that folder>`). Building from the repo instead:
   the README's "Run it on a phone" steps download the models and `./scripts/install_app.sh`
   installs everything.
3. Open SwachhSathi. Scanning is ready in a few seconds; Settings lists every model.
4. Optional, recommended: **Settings → Prepare all explanations** with the phone charging (about
   30 minutes on an entry-level phone). Llama writes each item's explanation once, on the phone, and
   later scans show it instantly.

## Measured (details: `docs/TECHNICAL_REPORT.md`, `eval/REPORT.md`)

- **Right bin on 2,371 South Asian photos** (BDWaste: wet, dry, special care): 91.9%, with the
  detector's verdict confirmed by SigLIP 2; 75.2% with the detector alone.
- **Llama answers**: 98.1% useful on 416 held-out questions (96.6% on the 88 the prompt was chosen
  on; baseline 69.3%), all checked against the knowledge base; off-topic questions declined.
- **Detector** (YOLO26n, 9.8 MB, TACO test split): mAP50 0.305, mAP50-95 0.237.
- **On an entry-level phone** (realme C61, Unisoc T612, 4 GB RAM): detector 0.23–0.42 s per frame;
  Llama loads in ~30 s on first use, first token 15.7 s, 4.4 tokens/s; app ~1.3 GB with Llama.
  Real items scanned by hand in airplane mode: right bin for all but a computer mouse.

## Known limits

The detector knows 14 items and was trained on TACO (European litter photos), not Indian
household waste; the other 39 knowledge-base items are reached by suggestions, voice or search, and
a second model must agree before any verdict. Llama is slow on budget phones (explanations take
15–30 s the first time; cached afterwards). Hindi text has not been reviewed by a native speaker;
voice was measured on synthetic speech. One city profile. No knowledge-base item yet for masks,
gloves or calculators.

## Licences

The app is AGPL-3.0 (it builds on Ultralytics YOLO). **Built with Llama**: Llama 3.2 is licensed
under the Llama 3.2 Community License, copy included. TACO images used for training have mixed
licences (`data/taco_licences.md`); BDWaste (CC BY 4.0) was used for evaluation only. Third-party
list: `THIRD_PARTY.md`.
