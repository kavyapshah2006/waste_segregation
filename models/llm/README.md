# LLM: Llama 3.2 on ExecuTorch

Llama 3.2 1B Instruct, 4-bit, XNNPACK, runs on the phone's Arm CPU. It explains decisions made by the knowledge base; it never makes them (see `pipeline/verify.py`).

## Two ways to get a `.pte`

| Path | Command | Needs |
|---|---|---|
| **Prebuilt** (first bring-up) | `./scripts/download_models.sh llama-spinquant --accept-llama-license` | Accepting the Llama 3.2 Community License. The repo is not gated, so no Hugging Face login is needed |
| **Own export** (needed for the WP8 fine-tune) | `./scripts/download_models.sh llama-original --accept-llama-license`, then `uv run python models/llm/export.py --mode int4 --checkpoint models/llm/out/original/original/consolidated.00.pth --params models/llm/out/original/original/params.json` | Hugging Face login with access to the gated `meta-llama/Llama-3.2-1B-Instruct` |

`export.py` wraps `python -m executorch.extension.llm.export.export_llm` with the configs that ship in the executorch package:

| `--mode` | Config | Checkpoint |
|---|---|---|
| `int4` | `llama_xnnpack.yaml` (8-bit dynamic activations, 4-bit weights, group 128; 4-bit embeddings) | Original or fine-tuned + merged |
| `spinquant` | `llama_xnnpack_spinquant.yaml` | Meta's SpinQuant INT4 EO8 checkpoint |
| `qat_lora` | `llama_xnnpack_qat.yaml` | Meta's QAT+LoRA INT4 EO8 checkpoint |

Context defaults to 1,024 tokens; our prompts are about 400 tokens and a shorter KV cache saves memory. Use `--dry-run` to print the exact command.

**Verified 2026-11-17:** `export.py --mode int4` on a tiny random-weight Llama 3.2 config produced a working 19 MB `.pte` in 50 s with ExecuTorch 1.5.1. The real 1B export has not been run yet; it needs the gated checkpoint.

## Running on the host (functional check only)

```bash
uv run python -m pipeline.cli --item pet_bottle --lang hi \
    --llm-model models/llm/out/spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte \
    --tokenizer models/llm/out/spinquant/tokenizer.model
```

This uses ExecuTorch's `TextLLMRunner` with `num_bos=0`, because our prompt already starts with `<|begin_of_text|>`. Laptop speed is never reported.

## Running on the phone (WP1, human session)

```bash
./scripts/push_models.sh models/llm/out/spinquant/*.pte models/llm/out/spinquant/tokenizer.model
```

Then follow `eval/latency/llm_bench.md`.

## Open questions

- **Prebuilt `.pte` compatibility:** the community `.pte` files were exported in late 2024. Check that they load with the 1.5.1 runtime (host CLI above, then the phone). If they don't, use the own-export path.
- **BOS on Android:** check whether `LlmModule` adds `<|begin_of_text|>` itself (see `pipeline/prompt.py`).

## Reference numbers (ExecuTorch README, OnePlus 12, not ours)

| Model | Decode tok/s | First token | RSS |
|---|---|---|---|
| 1B SpinQuant | 50.2 | 0.3 s | 1,921 MiB |
| 1B QAT+LoRA | 45.8 | 0.3 s | 2,255 MiB |
| 3B SpinQuant | 19.7 | 0.7 s | 3,726 MiB |
