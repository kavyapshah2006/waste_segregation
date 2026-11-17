"""Merge a LoRA adapter into Llama 3.2 and write Meta-format weights for ExecuTorch (WP8).

ExecuTorch's Llama export (models/llm/export.py) takes Meta's checkpoint layout
(consolidated.00.pth + params.json), but fine-tuning runs on the Hugging Face layout. This
script merges the adapter, renames the weights and undoes the Hugging Face reordering of the
query/key rows (HF permutes them for its rotate-half RoPE; Meta's layout interleaves pairs).

    uv run --group finetune python models/llm/lora/convert.py \\
        --base path/to/Llama-3.2-1B-Instruct --adapter models/llm/lora/out/run1/adapter \\
        --out models/llm/lora/out/run1/meta --check

--check builds ExecuTorch's own Llama model from the converted weights and compares its
logits with the Hugging Face model on a few tokens: a wrong permutation or rope setting shows
up as a large difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# models/llm/export.py exports with base.model_class=llama3_2, for which ExecuTorch forces a
# RoPE scale factor of 32 and hard-codes low_freq_factor=1, high_freq_factor=4 and an original
# context of 8192. A checkpoint with other values would export wrongly, so refuse it.
EXPECTED_ROPE = {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                 "original_max_position_embeddings": 8192}  # fmt: skip


def rope_settings(cfg) -> dict:
    """RoPE settings from a transformers config (v5: rope_parameters; v4: rope_scaling)."""
    rope = dict(getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {})
    rope.setdefault("rope_theta", getattr(cfg, "rope_theta", None))
    return rope


def params_from_config(cfg) -> dict:
    """Meta params.json for the model (same keys as Meta's original Llama 3.2 file)."""
    rope = rope_settings(cfg)
    scaled = rope.get("rope_type") == "llama3"
    if scaled:
        wrong = {k: rope.get(k) for k, v in EXPECTED_ROPE.items() if rope.get(k) != v}
        if wrong:
            raise ValueError(f"RoPE settings {wrong} differ from what the llama3_2 export uses")
    params = {
        "dim": cfg.hidden_size,
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads,
        "vocab_size": cfg.vocab_size,
        "ffn_dim_multiplier": 1.5,
        "multiple_of": 256,
        "norm_eps": cfg.rms_norm_eps,
        "rope_theta": float(rope["rope_theta"]),
        "use_scaled_rope": scaled,
    }
    # ExecuTorch derives the MLP width from dim, multiple_of and ffn_dim_multiplier
    # (model_args.py). Check it lands on the model's real width.
    hidden = int(2 * 4 * params["dim"] / 3)
    hidden = int(params["ffn_dim_multiplier"] * hidden)
    hidden = params["multiple_of"] * ((hidden + params["multiple_of"] - 1) // params["multiple_of"])
    if hidden != cfg.intermediate_size:
        raise ValueError(f"derived MLP width {hidden} != intermediate_size {cfg.intermediate_size}")
    return params


def _unpermute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of the permute in transformers' convert_llama_weights_to_hf.py."""
    dim1, dim2 = w.shape
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


_LAYER_MAP = {
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def hf_to_meta(sd: dict[str, torch.Tensor], cfg) -> dict[str, torch.Tensor]:
    out = {
        "tok_embeddings.weight": sd["model.embed_tokens.weight"],
        "norm.weight": sd["model.norm.weight"],
        # Llama 3.2 1B/3B tie the output layer to the embeddings.
        "output.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]),
    }
    for i in range(cfg.num_hidden_layers):
        hf, meta = f"model.layers.{i}.", f"layers.{i}."
        out[meta + "attention.wq.weight"] = _unpermute(
            sd[hf + "self_attn.q_proj.weight"], cfg.num_attention_heads
        )
        out[meta + "attention.wk.weight"] = _unpermute(
            sd[hf + "self_attn.k_proj.weight"], cfg.num_key_value_heads
        )
        for src, dst in _LAYER_MAP.items():
            out[meta + dst] = sd[hf + src]
    expected = len(out) - 3
    used = {k for k in sd if k.startswith("model.layers.")}
    if len(used) != expected:
        raise ValueError(f"unexpected layer weights: {sorted(used)[:5]}...")
    return out


def load_merged(base: Path | str, adapter: Path | None):
    """The base model with the adapter merged in (fp32, CPU)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model.eval()


@torch.no_grad()
def max_logit_diff(meta_sd: dict, params: dict, hf_model, tokens: list[int]) -> float:
    """Largest |logit| difference between ExecuTorch's eager Llama and the HF model."""
    from executorch.examples.models.llama.llama_transformer import construct_transformer
    from executorch.examples.models.llama.model_args import ModelArgs

    args = ModelArgs(
        **params, max_seq_len=max(64, len(tokens)), max_context_len=max(64, len(tokens))
    )
    if params["use_scaled_rope"]:
        args.rope_scale_factor = int(EXPECTED_ROPE["factor"])  # what the llama3_2 export sets
    et = construct_transformer(args)
    et.load_state_dict({k: v.float() for k, v in meta_sd.items()}, strict=True)
    et.eval()
    ids = torch.tensor([tokens])
    ours = et(ids)
    ours = ours[0] if isinstance(ours, tuple) else ours
    ref = hf_model(ids).logits[:, -1, :]
    return float((ours.reshape(ref.shape) - ref).abs().max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="HF model dir or repo id")
    p.add_argument("--adapter", type=Path, help="LoRA adapter dir from train.py (omit: base only)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--check", action="store_true", help="compare logits with ExecuTorch's model")
    args = p.parse_args(argv)

    model = load_merged(args.base, args.adapter)
    params = params_from_config(model.config)
    meta = hf_to_meta(model.state_dict(), model.config)
    if args.check:
        tokens = [128000, 128006, 882, 128007, 271, 6151, 100282, 103039, 35470, 128009]
        tokens = [t % params["vocab_size"] for t in tokens]
        diff = max_logit_diff(meta, params, model, tokens)
        print(f"check: max |logit diff| ExecuTorch vs HF = {diff:.2e}")
        if diff > 1e-3:
            print("check FAILED: the converted weights do not reproduce the HF model")
            return 1
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.to(dtype).contiguous() for k, v in meta.items()}, args.out / "consolidated.00.pth"
    )
    (args.out / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/consolidated.00.pth and params.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
