#!/usr/bin/env bash
# Download model files into the repo's gitignored model folders.
#
#   ./scripts/download_models.sh llama-spinquant --accept-llama-license
#   ./scripts/download_models.sh llama-qlora     --accept-llama-license
#   ./scripts/download_models.sh llama-original  --accept-llama-license   # needs `hf auth login`
#   ./scripts/download_models.sh llama-hf        --accept-llama-license   # needs `hf auth login`
#   ./scripts/download_models.sh vision <release-tag>                     # needs `gh auth login`
#
# Llama 3.2 is licensed under the Llama 3.2 Community License. Using or downloading it means
# accepting that licence, so the Llama targets refuse to run without --accept-llama-license.
# The executorch-community repos are prebuilt .pte files and are not gated; the original
# meta-llama checkpoint is gated and needs a Hugging Face login with access granted.
set -euo pipefail

cd "$(dirname "$0")/.."
target="${1:-}"

hf_download() {
    local repo="$1" dest="$2"
    shift 2
    mkdir -p "$dest"
    uvx --from "huggingface_hub>=0.34" hf download "$repo" "$@" --local-dir "$dest"
    echo "downloaded to $dest"
}

require_licence() {
    if [[ "${1:-}" != "--accept-llama-license" ]]; then
        echo "Llama 3.2 is under the Llama 3.2 Community License:" >&2
        echo "  https://www.llama.com/llama3_2/license/" >&2
        echo "Re-run with --accept-llama-license once you have read and accepted it." >&2
        exit 1
    fi
}

case "$target" in
    llama-spinquant)
        require_licence "${2:-}"
        hf_download executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET \
            models/llm/out/spinquant \
            Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte tokenizer.model params.json
        ;;
    llama-qlora)
        require_licence "${2:-}"
        hf_download executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET \
            models/llm/out/qlora \
            Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte tokenizer.model params.json
        ;;
    llama-original)
        require_licence "${2:-}"
        hf_download meta-llama/Llama-3.2-1B-Instruct models/llm/out/original \
            original/consolidated.00.pth original/params.json original/tokenizer.model
        ;;
    llama-hf)
        # Hugging Face layout, for merging a LoRA adapter (models/llm/lora/convert.py).
        require_licence "${2:-}"
        hf_download meta-llama/Llama-3.2-1B-Instruct models/llm/out/hf \
            config.json generation_config.json model.safetensors original/tokenizer.model
        ;;
    vision)
        tag="${2:?usage: download_models.sh vision <release-tag>}"
        dest="models/vision/out/release-$tag"
        mkdir -p "$dest"
        gh release download "$tag" --pattern 'model.pte' --pattern 'metadata.yaml' --dir "$dest"
        echo "downloaded to $dest"
        ;;
    *)
        sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
