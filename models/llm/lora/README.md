# LoRA fine-tune (WP8)

Teaches Llama 3.2 1B Instruct to answer in our format with the knowledge base's facts. Training runs on a Colab GPU; everything else runs on the laptop.

| File | What it does |
|---|---|
| `train.py` | LoRA training (PEFT) on `models/llm/synth_data/out/{train,val}.jsonl`. Loss on the answer tokens only. Tokenizes with the same C++ tokenizer as the ExecuTorch runner, prompt and answer encoded separately, exactly as the phone sees them |
| `convert.py` | Merges the adapter, converts Hugging Face weights to Meta's layout (undoing HF's query/key row permutation), writes `consolidated.00.pth` + `params.json`. `--check` compares logits with ExecuTorch's own Llama model |
| `make_bundle.py` | Zips `train.py` + fresh training data for Colab |
| `train.ipynb` | Colab notebook: install, upload bundle, download the base model, train, download the adapter |
| `dry_run.py` | The whole chain on a tiny random Llama on the CPU (~2 minutes) |

## Dry run (no GPU, no gated model)

```bash
uv run --group finetune python models/llm/lora/dry_run.py
```

Verified 2026-11-17: training lowers the loss, the converted weights reproduce the Hugging Face logits in ExecuTorch's model (max difference 2.4e-7; skipping the query/key fix gives 6.6e-3, which `--check` rejects), the int4 `.pte` exports (39 MB for the tiny model), loads in the runner and is scored by the eval. The answers are nonsense, as expected from random weights.

## Real run

Needs a human: access to the gated `meta-llama/Llama-3.2-1B-Instruct` on Hugging Face, and a Colab GPU.

1. Laptop: `uv run python models/llm/lora/make_bundle.py`
2. Colab: run `train.ipynb`, upload `models/llm/lora/out/lora_bundle.zip`, download `run1_adapter.zip`. Unzip it into `models/llm/lora/out/run1/`.
3. Laptop: download the base model in Hugging Face format (after `hf auth login`):

   ```bash
   ./scripts/download_models.sh llama-hf --accept-llama-license
   ```

4. Merge, convert and check:

   ```bash
   uv run --group finetune python models/llm/lora/convert.py \
       --base models/llm/out/hf --adapter models/llm/lora/out/run1/adapter \
       --out models/llm/lora/out/run1/meta --check
   ```

5. Export. The Llama 3.2 licence requires a fine-tuned model's name to start with "Llama", so name the file accordingly:

   ```bash
   uv run python models/llm/export.py --mode int4 \
       --checkpoint models/llm/lora/out/run1/meta/consolidated.00.pth \
       --params models/llm/lora/out/run1/meta/params.json \
       --output models/llm/out/Llama-3.2-1B-Instruct-WasteGuide-LoRA-run1-int4.pte
   ```

6. Score against the stock model (`eval/results/llm_quality_spinquant_stock_host_checked.json`):

   ```bash
   uv run python -m eval.llm_quality.run \
       --llm-model models/llm/out/Llama-3.2-1B-Instruct-WasteGuide-LoRA-run1-int4.pte \
       --tokenizer models/llm/out/hf/original/tokenizer.model --label lora_run1_host
   ```

Ship the fine-tune only if its useful rate beats the stock SpinQuant model, and read a sample of Hindi answers by hand (FuturePlan WP8).

Notes:
- `int4` is post-training quantisation (8-bit dynamic activations, 4-bit weights). SpinQuant and QAT need Meta's own training recipes, so a fine-tune of ours exports as `int4`. Compare quality after quantisation, not before.
- ExecuTorch's `llama3_2` export forces RoPE scale factor 32; `convert.py` refuses a checkpoint with other RoPE settings.
