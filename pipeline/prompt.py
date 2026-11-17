"""Llama 3.2 prompt builder (FuturePlan section 5.5).

The system message never changes, so the app can pre-fill its KV cache once and reuse it
for every scan. Everything that varies (facts, condition, language, question) goes in the
user message.

Chat format, from the Llama 3.2 Instruct tokenizer chat template:
    <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>...
Generation stops at <|eot_id|> (128009) or <|end_of_text|> (128001).

TODO(verify): whether the ExecuTorch Android runner adds <|begin_of_text|> itself. If it
does, call build_prompt(..., include_bos=False) on the phone so the BOS token is not doubled.
"""

from __future__ import annotations

import json

from pipeline.kb import Verdict

BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"

SYSTEM_PROMPT = (
    "You are a waste disposal guide for India. Use only the FACTS given. "
    "Never change the stream, bin, hazard or drop-off. "
    "Keep the answer under 80 words. "
    "Reply with one JSON object with exactly these keys: "
    '"item", "stream", "bin", "steps", "why", "tip", "language". '
    "Write why and tip in the requested LANGUAGE. Output nothing except the JSON object."
)

DEFAULT_QUESTION = "Explain how to dispose of this and why."


def _header(role: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def prompt_prefix(include_bos: bool = True) -> str:
    """The fixed part of every prompt (BOS + system message), for KV cache pre-fill."""
    return (BOS if include_bos else "") + _header("system") + SYSTEM_PROMPT + EOT


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def user_message(
    verdict: Verdict,
    attributes: dict | None,
    language_name: str,
    question: str | None = None,
    short: bool = False,
) -> str:
    facts = verdict.facts()
    if short:
        facts = {k: facts[k] for k in ("item", "stream", "bin", "steps")}
    lines = [f"FACTS: {_dumps(facts)}"]
    if not short:
        condition = {k: v for k, v in sorted((attributes or {}).items()) if v is not None}
        lines.append(f"CONDITION: {_dumps(condition)}")
    lines.append(f"LANGUAGE: {language_name}")
    lines.append(f"QUESTION: {question or DEFAULT_QUESTION}")
    return "\n".join(lines)


def build_prompt(
    verdict: Verdict,
    attributes: dict | None,
    language_name: str,
    question: str | None = None,
    *,
    short: bool = False,
    include_bos: bool = True,
) -> str:
    """Full prompt text, ending with the assistant header so the model writes the answer."""
    return (
        prompt_prefix(include_bos)
        + _header("user")
        + user_message(verdict, attributes, language_name, question, short)
        + EOT
        + _header("assistant")
    )
