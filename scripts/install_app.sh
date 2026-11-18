#!/usr/bin/env bash
# Install the app on the connected phone and push the large model files it reads.
#
#   ./scripts/install_app.sh                # newest APK (release, else debug) + models
#   LLM=qlora ./scripts/install_app.sh      # push the QLoRA Llama instead of SpinQuant
#   SKIP_MODELS=1 ./scripts/install_app.sh  # reinstall the APK only (models stay on the phone)
#
# Models go to the app's own folder (/sdcard/Android/data/org.wastesegregation.app/files/models),
# which is removed when the app is uninstalled. Small models (detector, KB) are inside the APK.
set -euo pipefail
cd "$(dirname "$0")/.."

PKG=org.wastesegregation.app
LLM="${LLM:-spinquant}"
APK=$(ls -t android/app/build/outputs/apk/release/*.apk android/app/build/outputs/apk/debug/*.apk 2>/dev/null | head -1 || true)
[[ -n "$APK" ]] || { echo "No APK: build it first (cd android && ./gradlew :app:assembleRelease)" >&2; exit 1; }

WIN_ADB=$(ls /mnt/c/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe 2>/dev/null | head -1 || true)
ADB="${ADB:-$(command -v adb.exe || echo "$WIN_ADB")}"
[[ -n "$ADB" ]] || { echo "adb.exe not found (see docs/DEV_SETUP.md)" >&2; exit 1; }
export ADB

echo "installing $APK"
"$ADB" install -r "$(wslpath -w "$(realpath "$APK")")"

if [[ -z "${SKIP_MODELS:-}" ]]; then
    llm_dir="models/llm/out/$LLM"
    files=("$llm_dir"/Llama-*.pte "$llm_dir/tokenizer.model")
    [[ -f models/fallback/out/siglip2_image_int8.pte ]] && files+=(models/fallback/out/siglip2_image_int8.pte)
    asr=models/speech/out/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05
    [[ -f $asr/model.int8.onnx ]] && files+=("$asr/model.int8.onnx" "$asr/tokens.txt")
    PHONE_DIR="/sdcard/Android/data/$PKG/files/models" ./scripts/push_models.sh "${files[@]}"
fi
echo "done: open the app; Settings lists which models loaded"
