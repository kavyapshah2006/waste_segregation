# Hardware Requirements

Hardware for the Raspberry Pi 5 smart-bin kiosk (primary form factor) and the Android companion app. See [DESIGN.md](DESIGN.md) for the architecture this supports.

**Pipeline split:** vision runs on the Hailo NPU (HailoRT); Llama runs on the Pi 5 Arm CPU (ExecuTorch).

---

## 1. Must-have (Pi 5 kiosk)

| # | Component | Notes |
|---|---|---|
| 1 | **Raspberry Pi 5, 8 GB** (16 GB if budget allows) | 8 GB fits Llama 1B + vision + speech; 16 GB leaves room to try Llama 3B on the Pi |
| 2 | **Official 27 W USB-C power supply (5V/5A)** | Phone chargers can't power the HAT, camera and USB devices together |
| 3 | **Raspberry Pi Active Cooler** | Sustained Llama inference throttles without it; fits under the AI HAT+ |
| 4 | **Raspberry Pi AI HAT+, 26 TOPS (Hailo-8)** | Vision only. AI HAT+ 2 has the same vision performance; its extra 8 GB is for on-NPU LLMs, which we can't use since Llama must run on ExecuTorch. Get the 26 TOPS version, not 13 TOPS |
| 5 | **Fast A2 microSD (64–128 GB) + USB 3 SSD (256 GB+)** | The AI HAT+ occupies the Pi's only PCIe slot, so no NVMe HAT; a USB SSD speeds up model loading |
| 6 | **Raspberry Pi Camera Module 3 Wide** + **Pi 5 camera cable (22-pin ↔ 15-pin, 300 mm)** | Wide lens covers the kiosk area; the Pi 5 has a smaller camera socket than the bundled cable |
| 7 | **Raspberry Pi Touch Display 2 (7")** | Kiosk screen |
| 8 | **USB speakerphone** (conference-call puck, mic + speaker) | Pi 5 has no audio jack; built-in echo cancellation helps speech recognition in noise |
| 9 | **Raspberry Pi Pico** (I/O co-controller) | Drives LEDs (and later servos) over USB; avoids `rpi_ws281x`, which doesn't support the Pi 5 |
| 10 | **WS2812B LED strip, 5V, ~1 m** + **74AHCT125 level shifter** + **5V supply for the strip** | One LED segment per bin; shifter converts 3.3 V logic to 5 V |
| 11 | **USB LED ring or panel light** | Consistent lighting is the cheapest accuracy gain, especially for clear plastic and glass |
| 12 | **4 colour-coded bins** (wet / dry / sanitary / special care) | Follow the local municipal colour scheme |
| 13 | **Inline USB-C power meter** | Required for power measurements in the evaluation |

## 2. Recommended

| Component | Why |
|---|---|
| **Load cell + HX711 amplifier** (per bin, or one shared scale) | Measured (not estimated) kg diverted from landfill for the impact metrics |
| **VL53L0X time-of-flight sensor** or PIR motion sensor | Wakes the system when someone approaches; saves power at idle |
| **Kiosk frame**: acrylic or 3D-printed camera arm and bin housing | Product-like demo |
| **Arm Android phone, 8 GB+ RAM** (team member's own is fine) | Android app: Llama 3B + vision on ExecuTorch |

## 3. Optional (stretch: automatic sorting)

| Component | Why |
|---|---|
| 4× MG996R servos + PCA9685 servo driver + separate 5–6 V supply | Flaps that open the correct bin automatically |

## 4. Setup accessories

- Micro-HDMI to HDMI cable (Pi 5 uses micro-HDMI)
- USB keyboard and mouse
- Ethernet cable
- microSD card reader
- Jumper wires and breadboard

---

## 5. Development laptop

Current dev machine: Intel Core Ultra 9 185H, Intel Arc integrated graphics, ~32 GB RAM, Windows 11 + WSL Ubuntu.

| Task | Where |
|---|---|
| Hailo model compilation (Dataflow Compiler needs x86_64 Linux) | Laptop, in WSL Ubuntu |
| Exporting Llama 1B/3B to ExecuTorch `.pte` | Laptop (32 GB RAM is sufficient) |
| Knowledge base, app and UI development | Laptop |
| YOLO training, Llama LoRA fine-tuning | Google Colab / Kaggle GPUs (Intel Arc has no CUDA). Cloud training is allowed; only inference must be on-device |
