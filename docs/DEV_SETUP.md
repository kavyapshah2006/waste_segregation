# Dev Setup (Phase 0)

Python work runs in **WSL Ubuntu** with uv. Android tooling runs on **Windows**; WSL calls `adb.exe` directly.

## 1. Python environment (WSL)

```bash
./scripts/setup_dev.sh
uv run pytest
```

Pinned: Python 3.12, `torch==2.14.0` (CPU build), `executorch==1.5.1`. Bump `torch` and `executorch` together only (ExecuTorch's `torch_pin.py` for the target tag says which torch it needs).

## 2. Android tooling (Windows)

1. Install [Android Studio](https://developer.android.com/studio).
2. In **SDK Manager → SDK Tools**, install: Android SDK Platform-Tools, NDK (Side by side), CMake.
3. Add `%LOCALAPPDATA%\Android\Sdk\platform-tools` to the Windows `PATH` so `adb.exe` is reachable from WSL.

## 3. Phone

1. **Settings → About phone →** tap **Build number** 7 times to enable Developer options.
2. **Developer options →** enable **USB debugging**.
3. Connect by USB, accept the "Allow USB debugging" prompt.
4. Check: `adb.exe devices` lists the phone.
5. Record specs: `./scripts/device_info.sh > docs/DEVICE.md` and commit it.

The phone must be 64-bit Arm with 6 GB+ RAM (see [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)).

The phone is borrowed, so before each session:

- Have the APK, `.pte` files and benchmark scripts ready on the laptop.
- Ask the owner before enabling Developer options and USB debugging.
- At the end, uninstall the app (`adb.exe uninstall <package>`) and turn off USB debugging if the owner prefers.

## 4. Llama access

Read and accept the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). After that:

- **Prebuilt `.pte` (enough for the first phone test):** `./scripts/download_models.sh llama-spinquant --accept-llama-license`. These repos are not gated, so no Hugging Face login is needed.
- **Own export and fine-tuning (WP8):** request access to the gated **meta-llama/Llama-3.2-1B-Instruct** on Hugging Face, run `uvx --from "huggingface_hub>=0.34" hf auth login`, then `./scripts/download_models.sh llama-original --accept-llama-license`. Approval can take time, so request it early.

## Phase 0 checklist

- [x] uv project, pinned deps, `uv.lock`
- [x] Repo skeleton, README, `.gitignore`
- [x] CI: `uv sync --locked`, ruff, pytest (incl. ExecuTorch export + run smoke test)
- [ ] Android Studio + platform-tools + NDK installed
- [ ] Phone in developer mode, visible in `adb.exe devices`
- [ ] `docs/DEVICE.md` committed
- [ ] Llama 3.2 access approved on Hugging Face
