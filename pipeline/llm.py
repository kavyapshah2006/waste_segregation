"""Run a Llama .pte on the laptop through ExecuTorch's TextLLMRunner.

For functional checks only (does the model follow the prompt, does the verifier cope).
Speed on the x86 laptop says nothing about the phone and is never reported.

The prompt from pipeline/prompt.py already starts with <|begin_of_text|>, so the runner is
told to add no BOS token of its own (num_bos=0).

The prebuilt executorch-community .pte files use kernels outside the core runtime:
llama::custom_sdpa and llama::fast_hadamard_transform (custom_ops library) and
quantized_decomposed::embedding_byte (quantized kernels). Importing those two modules loads
their shared libraries, which registers the kernels; without them loading 'forward' fails.
"""

from __future__ import annotations

from pathlib import Path


class ExecuTorchGenerator:
    """A pipeline.assistant.Generator backed by a .pte model and its tokenizer."""

    def __init__(
        self,
        model_path: Path | str,
        tokenizer_path: Path | str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  (registers kernels)
        import executorch.kernels.quantized  # noqa: F401  (registers kernels)
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

        self._runner = TextLLMRunner(str(model_path), str(tokenizer_path))
        self._config = GenerationConfig(
            echo=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_bos=0,
            num_eos=0,
        )
        self.last_stats = None

    def __call__(self, prompt: str) -> str:
        pieces: list[str] = []
        self._runner.reset()
        self._runner.generate(
            prompt,
            self._config,
            token_callback=pieces.append,
            stats_callback=lambda stats: setattr(self, "last_stats", stats),
        )
        return "".join(pieces)
