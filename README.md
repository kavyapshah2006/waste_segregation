# Waste Segregation Assistant

On-device AI assistant that identifies waste, decides the correct bin under India's SWM Rules 2026, and explains disposal and recycling options. Vision and Llama run fully offline on an Arm Android phone through **ExecuTorch**.

> Status: Phase 0 (repo and tooling setup). See [docs/roadmap.md](docs/roadmap.md).

## Docs

- [Design](docs/DESIGN.md): problem, state of the art, architecture, tech stack
- [Hardware requirements](docs/HARDWARE_REQUIREMENTS.md): minimal, phone-first
- [Roadmap](docs/roadmap.md): phased plan from setup to submission
- [Future plan](docs/FuturePlan.md): work-package implementation guide with contracts, tests and status
- [Dev setup](docs/DEV_SETUP.md): Python, Android and phone setup

## Quick start (WSL Ubuntu)

Requires [uv](https://docs.astral.sh/uv/).

```bash
./scripts/setup_dev.sh   # installs uv if missing, syncs the env, checks tools
uv run pytest            # includes an ExecuTorch export + run smoke test
uv run ruff check .
```

Python dependencies are managed only with uv (`uv add <pkg>`, `uv run <cmd>`). `torch` and `executorch` are pinned together in `pyproject.toml`.

## Repository layout

| Path | Purpose | Roadmap phase |
|---|---|---|
| `docs/` | Design, hardware, roadmap, later the technical report | All |
| `kb/` | Knowledge base: disposal rules per item and city | 2 |
| `data/` | Taxonomy, dataset scripts, small fixtures | 3, 5 |
| `models/` | Export and training code for Llama, vision, CLIP, speech | 1, 3, 5–7 |
| `android/` | Kotlin app | 1, 4+ |
| `eval/` | Accuracy, latency and offline evaluation | 9 |
| `scripts/` | Dev setup, device info, model download | 0+ |
| `tests/` | `uv run pytest` | 0+ |

Model binaries and raw datasets are not committed.
