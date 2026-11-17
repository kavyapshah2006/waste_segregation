"""Llama 3.2 prompt builder (FuturePlan section 5.5).

The system message never changes, so the app can pre-fill its KV cache once and reuse it
for every scan. Everything that varies (facts, condition, language, question) goes in the
user message.

Chat format, from the Llama 3.2 Instruct tokenizer chat template:
    <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>...
Generation stops at <|eot_id|> (128009) or <|end_of_text|> (128001).

BOS handling: the prompt text includes <|begin_of_text|>, so the runner must add none
(GenerationConfig.num_bos=0, which is the Python runner's default; see pipeline/llm.py).
TODO(verify): the Android LlmModule's BOS behaviour on the phone (WP1). If it adds one,
call build_prompt(..., include_bos=False) there so the BOS token is not doubled.

Answer format: Llama writes only {"why": ..., "tip": ...}. Item, stream, bin, steps, hazard
and drop-off are already in the KB verdict, so having Llama copy them back only cost tokens
(Hindi answers ran past the token limit in the stock-model baseline). The prompt ends with
ANSWER_PREFIX, the start of the JSON object, so the model's first token is already inside
the `why` string. Whatever the runner returns is the text after that prefix; join the two
with full_answer() before verifying.
"""

from __future__ import annotations

import json

from pipeline.kb import Verdict

BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"

SYSTEM_PROMPT = (
    "You are a waste disposal guide for India. Use only the FACTS given and never "
    "contradict them. The app already shows the facts to the user; your job is to explain. "
    'Reply with one JSON object with exactly two keys: "why" (answer the QUESTION in one or '
    'two short sentences) and "tip" (one short practical tip). '
    "Write both in the requested LANGUAGE. Output nothing except the JSON object."
)

DEFAULT_QUESTION = "Explain how to dispose of this and why."

# The assistant turn starts with this text; the model continues inside the "why" string.
ANSWER_PREFIX = '{"why": "'


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
    """Full prompt text, ending with the assistant header and ANSWER_PREFIX."""
    return (
        prompt_prefix(include_bos)
        + _header("user")
        + user_message(verdict, attributes, language_name, question, short)
        + EOT
        + _header("assistant")
        + ANSWER_PREFIX
    )


def full_answer(generated: str) -> str:
    """The complete answer text: ANSWER_PREFIX plus what the model generated after it."""
    return ANSWER_PREFIX + generated
