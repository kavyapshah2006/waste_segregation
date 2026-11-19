#!/usr/bin/env bash
# Build everything a release needs into dist/: the APK (with the detector inside) and the model
# files the phone downloads separately, with licences and checksums.
#
#   ./scripts/build_release.sh models/vision/runs/yolo26n_taco_all_cpu/weights/best.pt
#
# Then publish (needs gh and network):
#   gh release create v0.1.0 dist/* --title "v0.1.0" --notes-file docs/RELEASE_NOTES.md
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS="${1:?usage: build_release.sh <detector best.pt>}"
DIST=dist
LLM_DIR=models/llm/out/spinquant
ASR_DIR=models/speech/out/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05
LLAMA_LICENSE_URL=https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE

if [ -z "${ANDROID_HOME:-}" ] && ! grep -qs '^sdk.dir=' android/local.properties; then
    echo "Android SDK not found: set ANDROID_HOME or sdk.dir in android/local.properties" >&2
    exit 1
fi

echo "== 1. detector: export to ExecuTorch and check against PyTorch"
check_dir=$(mktemp -d)
# An array, not `ls | head`: with pipefail, head closing the pipe early can kill the script.
test_images=(data/processed/waste/images/test/*)
cp "${test_images[@]:0:20}" "$check_dir"/
uv run --group vision python models/vision/export_executorch.py --weights "$WEIGHTS" \
    --out models/vision/out --check-images "$check_dir"
rm -r "$check_dir"

echo "== 2. KB and the SigLIP text side (fallback + dirty/broken prompts)"
uv run python kb/build_kb.py
uv run --group finetune python models/fallback/build.py --text-only
uv run --group finetune python models/attributes/build.py

echo "== 3. release APK"
(cd android && ./gradlew --console=plain :app:assembleRelease)

echo "== 4. collect"
mkdir -p "$DIST"
rm -f "$DIST"/*
cp android/app/build/outputs/apk/release/app-release.apk "$DIST/waste-segregation.apk"
cp "$WEIGHTS" "$DIST/detector_yolo26n_taco.pt"
cp models/vision/out/model.pte "$DIST/detector_yolo26n_taco.pte"
cp "$LLM_DIR"/Llama-*.pte "$LLM_DIR/tokenizer.model" "$DIST/"
cp models/fallback/out/siglip2_image_int8.pte "$DIST/"
cp "$ASR_DIR/model.int8.onnx" "$DIST/omnilingual_asr_300m_ctc_v2_int8.onnx"
cp "$ASR_DIR/tokens.txt" "$DIST/omnilingual_asr_tokens.txt"
cp "$ASR_DIR/LICENSE" "$DIST/LICENSE_omnilingual_asr.txt"
cp NOTICE "$DIST/NOTICE.txt"
cp LICENSE "$DIST/LICENSE_app_AGPL-3.0.txt"
cp THIRD_PARTY.md "$DIST/"
# The Llama 3.2 Community License requires a copy with the Llama materials: fetched verbatim.
curl -fsSL --retry 3 --retry-all-errors -o "$DIST/LICENSE_Llama_3.2.txt" "$LLAMA_LICENSE_URL"
(cd "$DIST" && sha256sum -- * > SHA256SUMS)
ls -la "$DIST"
echo "Phone install from a release: put the .pte, tokenizer.model, siglip2_image_int8.pte,"
echo "model.int8.onnx (rename from omnilingual_asr_300m_ctc_v2_int8.onnx) and tokens.txt in"
echo "/sdcard/Android/data/org.wastesegregation.app/files/models/ (see docs/PHONE_SESSION.md)."
