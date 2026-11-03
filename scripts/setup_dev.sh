#!/usr/bin/env bash
# Set up the Python dev environment (run inside WSL Ubuntu) and report missing tools.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked

echo
echo "Tool check:"
for tool in uv git cmake java; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "  [ok]      $tool"
    else
        echo "  [missing] $tool"
    fi
done

# adb usually lives on the Windows side; WSL can call adb.exe directly.
if command -v adb >/dev/null 2>&1 || command -v adb.exe >/dev/null 2>&1; then
    echo "  [ok]      adb"
else
    echo "  [missing] adb (install Android Studio on Windows; see docs/DEV_SETUP.md)"
fi

echo
echo "Next: uv run pytest"
