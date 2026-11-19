#!/usr/bin/env bash
# Collect measurements from the phone during a session (docs/PHONE_SESSION.md).
#
#   ./scripts/phone_collect.sh start      # device info -> docs/DEVICE.md, reset battery stats
#   ./scripts/phone_collect.sh memory     # snapshot of the app's memory (run while Llama writes)
#   ./scripts/phone_collect.sh finish     # battery stats + pull the app's exported timings
#
# Everything lands in eval/results/phone/ (raw); eval/latency/import_phone.py turns it into
# the result files the report reads.
set -euo pipefail
cd "$(dirname "$0")/.."

PKG=org.wastesegregation.app
OUT=eval/results/phone
mkdir -p "$OUT"
# Windows SDK, or the same folder redirected into a packaged app's LocalCache (e.g. when the
# SDK was installed from the Claude desktop app).
WIN_ADB=$(ls /mnt/c/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe \
    /mnt/c/Users/*/AppData/Local/Packages/*/LocalCache/Local/Android/Sdk/platform-tools/adb.exe \
    2>/dev/null | head -1 || true)
ADB="${ADB:-$(command -v adb.exe || echo "$WIN_ADB")}"
[[ -n "$ADB" ]] || { echo "adb.exe not found (see docs/DEV_SETUP.md)" >&2; exit 1; }
"$ADB" get-state >/dev/null 2>&1 || { echo "No phone connected (USB debugging on?)" >&2; exit 1; }
stamp=$(date +%Y-%m-%d_%H%M)

case "${1:-}" in
    start)
        ADB="$ADB" ./scripts/device_info.sh > docs/DEVICE.md
        "$ADB" shell dumpsys batterystats --reset > /dev/null
        "$ADB" shell dumpsys battery | tr -d '\r' > "$OUT/battery_start_$stamp.txt"
        echo "wrote docs/DEVICE.md; battery stats reset"
        ;;
    memory)
        "$ADB" shell dumpsys meminfo "$PKG" | tr -d '\r' > "$OUT/meminfo_$stamp.txt"
        grep -E 'TOTAL PSS|TOTAL RSS|TOTAL:' "$OUT/meminfo_$stamp.txt" | head -3
        "$ADB" shell cat /proc/meminfo | tr -d '\r' | head -3 > "$OUT/device_meminfo_$stamp.txt"
        ;;
    finish)
        "$ADB" shell dumpsys battery | tr -d '\r' > "$OUT/battery_end_$stamp.txt"
        "$ADB" shell dumpsys batterystats "$PKG" | tr -d '\r' > "$OUT/batterystats_$stamp.txt"
        win_out=$(wslpath -w "$(realpath "$OUT")")
        "$ADB" pull "/sdcard/Android/data/$PKG/files/results/." "$win_out" || true
        ls -la "$OUT"
        ;;
    *)
        sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
