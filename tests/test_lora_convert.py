"""LoRA tooling (models/llm/lora): weight conversion and training examples.

The full chain (train -> convert -> export -> run) is exercised by
models/llm/lora/dry_run.py, which takes a few minutes; these are the fast checks.
"""

import pytest

transformers = pytest.importorskip("transformers")  # uv sync --group finetune

from models.llm.lora import convert  # noqa: E402
from models.llm.lora.train import IGNORE, build_example  # noqa: E402

ROPE = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}  # fmt: skip


def _config(**overrides):
    kwargs = dict(
        vocab_size=512, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
        rms_norm_eps=1e-5, rope_theta=500000.0, rope_scaling=dict(ROPE), tie_word_embeddings=True,
    )  # fmt: skip
    kwargs.update(overrides)
    return transformers.LlamaConfig(**kwargs)


@pytest.fixture(scope="module")
def tiny():
    import torch

    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(_config()).eval()


def test_converted_weights_reproduce_the_hf_logits(tiny):
    params = convert.params_from_config(tiny.config)
    meta = convert.hf_to_meta(tiny.state_dict(), tiny.config)
    assert convert.max_logit_diff(meta, params, tiny, [1, 7, 42, 99, 3, 500]) < 1e-5


def test_skipping_the_qk_unpermute_is_caught(tiny, monkeypatch):
    params = convert.params_from_config(tiny.config)
    monkeypatch.setattr(convert, "_unpermute", lambda w, n: w)
    meta = convert.hf_to_meta(tiny.state_dict(), tiny.config)
    assert convert.max_logit_diff(meta, params, tiny, [1, 7, 42, 99, 3, 500]) > 1e-3


def test_params_for_llama_3_2_1b_match_metas_file():
    cfg = _config(vocab_size=128256, hidden_size=2048, intermediate_size=8192,
                  num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8)  # fmt: skip
    assert convert.params_from_config(cfg) == {
        "dim": 2048, "n_layers": 16, "n_heads": 32, "n_kv_heads": 8, "vocab_size": 128256,
        "ffn_dim_multiplier": 1.5, "multiple_of": 256, "norm_eps": 1e-05,
        "rope_theta": 500000.0, "use_scaled_rope": True,
    }  # fmt: skip


def test_rope_settings_the_export_would_ignore_are_refused():
    with pytest.raises(ValueError, match="RoPE"):
        convert.params_from_config(_config(rope_scaling={**ROPE, "factor": 8.0}))


def test_mlp_width_mismatch_is_refused():
    with pytest.raises(ValueError, match="MLP width"):
        convert.params_from_config(_config(intermediate_size=320))


def test_training_example_masks_the_prompt():
    class FakeTok:
        def encode(self, text, bos, eos):
            return [ord(c) for c in text]

    ids, labels = build_example(FakeTok(), "PROMPT", 'ok"}')
    assert ids == [ord(c) for c in 'PROMPTok"}']
    assert labels == [IGNORE] * 6 + [ord(c) for c in 'ok"}']
