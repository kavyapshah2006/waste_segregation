# Hardware Requirements

Minimal-hardware plan. The problem statement lists **Arm-based Android smartphones** as a valid target, so the primary build runs entirely on one phone and needs **nothing to be bought**. The Raspberry Pi kiosk is an optional upgrade if hardware becomes available later. See [DESIGN.md](DESIGN.md) for the architecture.

**Pipeline split (phone):** vision and Llama both run through ExecuTorch on the phone. Vision can use the phone's NPU/GPU backend where supported; Llama runs on the Arm CPU (XNNPACK + KleidiAI).

---

## Tier 0: Required (nothing to buy)

| Component | Requirement | Notes |
|---|---|---|
| **Arm Android phone** | 64-bit Arm (arm64-v8a), **6 GB RAM minimum**, 8 GB+ preferred | 6 GB is enough for Llama 3.2 1B INT4 + vision + speech; 8 GB+ allows Llama 3B. Built-in camera, mic, speaker and screen cover the whole interface |
| **USB cable** | Phone to laptop | For `adb` install, logs and benchmarking |
| **Development laptop** | Already available | See section 3 |

**Important:** the laptop is x86 (Intel), not Arm. Demos must run on the phone, not on the laptop or an x86 Android emulator, to meet the Arm requirement.

## Tier 1: Nice to have (cheap or improvised)

| Component | Why | Low-cost substitute |
|---|---|---|
| Phone stand or small tripod | Hands-free "kiosk" mode for the demo video | Books, a cardboard stand |
| Even lighting | Improves accuracy on clear plastic and glass | A desk lamp or window light |
| Second Android phone (team member's) | Test on a weaker device; show it runs on a range of phones | Borrow one |
| 4 labelled bins or boxes | Makes the demo video realistic | Cardboard boxes with coloured paper |

## Tier 2: Optional upgrade (only if hardware becomes available)

The same ExecuTorch models run on the Pi 5, so this can be added later without redesign.

| Component | Notes |
|---|---|
| Raspberry Pi 5, 8 GB | Arm Cortex-A76 CPU; runs Llama via ExecuTorch |
| Official 27 W USB-C power supply | Needed for stable operation under load |
| Active Cooler | Sustained Llama inference throttles without it |
| A2 microSD, 64 GB+ | OS and models |
| Any USB webcam | Instead of the Camera Module 3 |
| Raspberry Pi AI HAT+ (26 TOPS) | Only if vision on the Pi CPU is too slow; moves vision to the Hailo NPU |

Further kiosk extras (LED bin indicators, load cells, speakerphone, servos) are deferred until a Pi is in hand.

---

## What the phone-first plan gives up, and how we compensate

| Lost | Compensation |
|---|---|
| Hailo NPU vision pipeline | Vision on ExecuTorch with the phone's NPU/GPU backend where available (Qualcomm, MediaTek, Samsung, Vulkan) or XNNPACK on CPU |
| Bin LEDs | Full-screen bin colour + voice prompt |
| Measured kg diverted (load cell) | Item counts × typical item weight from the knowledge base (clearly labelled as estimates) |
| Power measurement with a USB meter | Android battery stats (`dumpsys batterystats`) and per-inference energy estimates |

---

## 3. Development laptop

Current dev machine: Intel Core Ultra 9 185H, Intel Arc integrated graphics, ~32 GB RAM, Windows 11 + WSL Ubuntu.

| Task | Where |
|---|---|
| Exporting Llama 1B/3B and vision models to ExecuTorch `.pte` | Laptop (32 GB RAM is sufficient) |
| Android app build (Android Studio), knowledge base, tooling | Laptop |
| Hailo model compilation (only if Tier 2 AI HAT+ is used) | Laptop, in WSL Ubuntu (x86_64) |
| YOLO training, Llama LoRA fine-tuning | Google Colab / Kaggle GPUs (Intel Arc has no CUDA). Cloud training is allowed; only inference must be on-device |
