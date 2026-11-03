#!/usr/bin/env bash
# Print the connected Android phone's specs as Markdown, for docs/DEVICE.md.
# Usage: ./scripts/device_info.sh > docs/DEVICE.md
set -euo pipefail

if command -v adb >/dev/null 2>&1; then
    ADB=adb
elif command -v adb.exe >/dev/null 2>&1; then
    ADB=adb.exe
else
    echo "adb not found. Install Android Studio platform-tools (see docs/DEV_SETUP.md)." >&2
    exit 1
fi

if ! "$ADB" get-state >/dev/null 2>&1; then
    echo "No device connected. Enable USB debugging and accept the prompt on the phone." >&2
    exit 1
fi

prop() { "$ADB" shell getprop "$1" | tr -d '\r'; }

mem_kb=$("$ADB" shell grep MemTotal /proc/meminfo | tr -d '\r' | awk '{print $2}')
features=$("$ADB" shell grep -m1 Features /proc/cpuinfo | tr -d '\r' | cut -d: -f2)
has() { [[ " $features " == *" $1 "* ]] && echo yes || echo no; }

cat <<EOF
# Target device

| Property | Value |
|---|---|
| Model | $(prop ro.product.manufacturer) $(prop ro.product.model) |
| SoC | $(prop ro.soc.manufacturer) $(prop ro.soc.model) ($(prop ro.board.platform)) |
| ABI | $(prop ro.product.cpu.abi) |
| Android | $(prop ro.build.version.release) (API $(prop ro.build.version.sdk)) |
| RAM | $((mem_kb / 1024 / 1024)) GB ($mem_kb kB) |
| CPU dotprod (asimddp) | $(has asimddp) |
| CPU i8mm | $(has i8mm) |
| CPU SVE2 | $(has sve2) |
| CPU SME2 | $(has sme2) |

CPU features: \`$features\`
EOF
