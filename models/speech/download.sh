#!/usr/bin/env bash
# Download the offline speech models into models/speech/out/ (gitignored).
#
#   ./models/speech/download.sh
#
# ASR (on the phone): Dolphin CTC multi-language, int8 (Apache-2.0, DataoceanAI/Dolphin).
#   Covers hi, ta, te, bn, mr. Base (81 MB) and small (192 MB) to compare.
# TTS (laptop tests only): Piper hi_IN "pratham", int8. Its training data is CC BY-NC-SA 4.0,
#   so it is used here to make test audio and is NOT shipped; the app uses Android's own
#   text-to-speech for Hindi.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download
for f in \
    asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2 \
    asr-models/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02.tar.bz2 \
    asr-models/sherpa-onnx-whisper-base.tar.bz2 \
    asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05.tar.bz2 \
    tts-models/vits-piper-hi_IN-pratham-medium-int8.tar.bz2; do
    name=$(basename "$f" .tar.bz2)
    if [[ -d "out/$name" ]]; then echo "have $name"; continue; fi
    # Download fully first, so a dropped connection never leaves a half-extracted folder.
    curl -fsSL --retry 3 --retry-all-errors -o "out/$name.tar.bz2.part" "$BASE/$f"
    tar -xjf "out/$name.tar.bz2.part" -C out
    rm "out/$name.tar.bz2.part"
    echo "got $name ($(du -sh "out/$name" | cut -f1))"
done
