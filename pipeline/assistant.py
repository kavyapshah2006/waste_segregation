"""Ask Llama to explain a verdict, with one shorter retry, then fall back to KB text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pipeline.kb import KnowledgeBase, Verdict
from pipeline.prompt import build_prompt, full_answer
from pipeline.verify import VerifiedOutput, verify

# Takes the full prompt text, returns the text the model generated after it.
Generator = Callable[[str], str]


@dataclass
class Explanation:
    verified: VerifiedOutput
    prompts: list[str] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.prompts)


def explain(
    kb: KnowledgeBase,
    verdict: Verdict,
    attributes: dict | None,
    generate: Generator,
    question: str | None = None,
    include_bos: bool = True,
) -> Explanation:
    language_name = kb.language_name(verdict.language)
    result = Explanation(verified=None)  # type: ignore[arg-type]
    for short in (False, True):
        prompt = build_prompt(
            verdict, attributes, language_name, question, short=short, include_bos=include_bos
        )
        raw = generate(prompt)
        result.prompts.append(prompt)
        result.raw_outputs.append(raw)
        result.verified = verify(full_answer(raw), verdict)
        if result.verified.status != "fallback":
            break
    return result
