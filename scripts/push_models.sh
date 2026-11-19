#!/usr/bin/env bash
# Copy model files to the connected phone, where the ExecuTorch Llama demo app reads them.
#
#   ./scripts/push_models.sh models/llm/out/spinquant/*.pte models/llm/out/spinquant/tokenizer.model
#
# Files go to /data/local/tmp/llama/ (override with PHONE_DIR=...). Works with Linux adb or
# with Windows adb.exe called from WSL (paths are converted with wslpath).
set -euo pipefail

PHONE_DIR="${PHONE_DIR:-/data/local/tmp/llama}"

if [[ $# -eq 0 ]]; then
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

# Windows platform-tools in the default SDK folder, if adb.exe is not on PATH.
# Windows SDK, or the same folder redirected into a packaged app's LocalCache (e.g. when the
# SDK was installed from the Claude desktop app).
WIN_ADB=$(ls /mnt/c/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe \
    /mnt/c/Users/*/AppData/Local/Packages/*/LocalCache/Local/Android/Sdk/platform-tools/adb.exe \
    2>/dev/null | head -1 || true)

if [[ -n "${ADB:-}" ]]; then
    case "$ADB" in *.exe) to_adb_path() { wslpath -w "$1"; } ;; *) to_adb_path() { printf '%s' "$1"; } ;; esac
elif command -v adb.exe >/dev/null 2>&1; then
    # From WSL the phone is only visible to Windows adb (USB is not passed to Linux).
    ADB=adb.exe
    to_adb_path() { wslpath -w "$1"; }
elif [[ -n "$WIN_ADB" ]]; then
    ADB="$WIN_ADB"
    to_adb_path() { wslpath -w "$1"; }
elif command -v adb >/dev/null 2>&1; then
    ADB=adb
    to_adb_path() { printf '%s' "$1"; }
else
    echo "adb not found. Install Android platform-tools (see docs/DEV_SETUP.md)." >&2
    exit 1
fi

if ! "$ADB" get-state >/dev/null 2>&1; then
    echo "No device connected. Enable USB debugging and accept the prompt on the phone." >&2
    exit 1
fi

"$ADB" shell mkdir -p "$PHONE_DIR"
for file in "$@"; do
    [[ -f "$file" ]] || { echo "not a file: $file" >&2; exit 1; }
    echo "pushing $file"
    "$ADB" push "$(to_adb_path "$(realpath "$file")")" "$PHONE_DIR/"
done
"$ADB" shell ls -l "$PHONE_DIR"
