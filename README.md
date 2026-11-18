# Waste Segregation Assistant

On-device AI assistant that identifies waste, decides the correct bin under India's SWM Rules 2026, and explains disposal and recycling options. Vision and Llama run fully offline on an Arm Android phone through **ExecuTorch**.

> Status: the Android app builds (CI produces the APK) with camera scan, instant KB verdict, streamed Llama explanation, photo upload, unknown-item suggestions, dirty/broken questions, Hindi/English voice, corrections and an impact screen. Llama answers: 96.6% useful on held-out questions with the shipped prompt (69.3% baseline). Detector (YOLO26n on TACO, 9.8 MB): 0.305 mAP50 on held-out images; at the 0.5 gate 88.4% of its verdicts land in the right stream ([report §5.1](docs/TECHNICAL_REPORT.md#51-sorting-accuracy)). Waiting on: the phone session (speed, memory, battery, demo video). Per-work-package status: [docs/FuturePlan.md](docs/FuturePlan.md#12-status); measured results: [eval/REPORT.md](eval/REPORT.md).

**Built with Llama.** Licensed under the GNU AGPL-3.0 ([LICENSE](LICENSE)); third-party licences in [THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE).

## Docs

- [Design](docs/DESIGN.md): problem, state of the art, architecture, tech stack
- [Hardware requirements](docs/HARDWARE_REQUIREMENTS.md): minimal, phone-first
- [Roadmap](docs/roadmap.md): phased plan from setup to submission
- [Future plan](docs/FuturePlan.md): work-package implementation guide with contracts, tests and status
- [Pipelines](docs/PIPELINES.md): the five on-device pipelines, their runtimes, models and memory policy
- [Dev setup](docs/DEV_SETUP.md): Python, Android and phone setup
- [Phone session runbook](docs/PHONE_SESSION.md): install, measure and record the demo in one ~90 minute sitting
- [Technical report (draft)](docs/TECHNICAL_REPORT.md) and [demo video script](docs/DEMO_SCRIPT.md)
- [Evaluation report](eval/REPORT.md): every measured number, regenerated from `eval/results/`

## Quick start (WSL Ubuntu)

Requires [uv](https://docs.astral.sh/uv/).

```bash
./scripts/setup_dev.sh   # installs uv if missing, syncs the env, checks tools
uv run pytest            # includes an ExecuTorch export + run smoke test
uv run ruff check .
```

Python dependencies are managed only with uv (`uv add <pkg>`, `uv run <cmd>`). `torch` and `executorch` are pinned together in `pyproject.toml`.

## How it works

Point the camera at one item. The detector names it, frame voting makes sure it is stable, and the
knowledge base decides the bin (SWM Rules 2026: green wet, blue dry, red sanitary, black special
care) with the steps and any hazard. The card shows at once; Llama 3.2 1B then streams a short
explanation and tip, checked against the knowledge base before it is shown. Not sure what it is?
The app offers three suggestions instead of guessing. Ask by typing or by voice, in English or
Hindi. Everything runs on the phone, with no internet permission at all.

Pipelines and models: [docs/PIPELINES.md](docs/PIPELINES.md). Design and results:
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Run it on a phone

```bash
cd android && ./gradlew :app:assembleDebug && cd ..   # or download the APK from a release
./scripts/download_models.sh llama-spinquant --accept-llama-license   # Llama 3.2 1B .pte
uv run --group finetune python models/fallback/build.py   # SigLIP 2 image encoder .pte
./models/speech/download.sh                               # Omnilingual ASR (voice)
./scripts/install_app.sh                                  # APK + models over USB (adb)
```

Full checklist for measuring and recording the demo: [docs/PHONE_SESSION.md](docs/PHONE_SESSION.md).

## Repository layout

| Path | Purpose | Roadmap phase |
|---|---|---|
| `docs/` | Design, pipelines, technical report, phone runbook, demo script | All |
| `kb/` | Knowledge base: disposal rules per item and city | 2 |
| `data/` | Taxonomy, dataset scripts, small fixtures | 3, 5 |
| `pipeline/` | Python reference of the app's logic (KB, prompt, verifier, voting, voice, impact) | 2+ |
| `models/` | Export and training code for Llama (+ LoRA), vision, SigLIP fallback, speech | 1, 3, 5–7 |
| `android/` | Kotlin: `core/` (tested port of `pipeline/`) and `app/` (the Android app) | 1, 4+ |
| `eval/` | Accuracy, latency and offline evaluation | 9 |
| `scripts/` | Dev setup, model download, install on the phone, phone measurements, release | 0+ |
| `tests/` | `uv run pytest` | 0+ |

Model binaries and raw datasets are not committed.
