"""Ask Llama to explain a verdict, with one shorter retry, then fall back to KB text.

Free-text questions that are clearly not about waste are declined here, without calling
Llama: the stock 1B model answered every off-topic question in the baseline instead of
declining. The guard is a plain word list, not a classifier. A question passes if it uses a
waste word, names the item, or says "this" / "यह"; it only has to stop questions like
"What is the capital of France?". The prompt also tells Llama to decline, as a second line.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pipeline.kb import KnowledgeBase, Verdict
from pipeline.prompt import build_prompt, full_answer
from pipeline.verify import VerifiedOutput, kb_output, verify

# Takes the full prompt text, returns the text the model generated after it.
Generator = Callable[[str], str]

_TOPIC_EN = re.compile(
    r"\b(?:bins?|dustbins?|waste|trash|garbage|rubbish|junk|recycl\w*|compost\w*|dispos\w*|"
    r"throw\w*|thrown|dump\w*|sort\w*|segregat\w*|separat\w*|bags?|clean\w*|rinse|wash\w*|"
    r"wrap\w*|safe\w*|danger\w*|harm\w*|toxic|hazard\w*|reus\w*|sell|scrap|kabadi\w*|"
    r"collect\w*|pick\w*|burn\w*|broke\w*|break\w*|leak\w*|this|these)\b",
    re.I,
)
_TOPIC_HI = re.compile(
    r"कचर|कूड़|डिब्ब|फेंक|रीसा[यइ]कल|खाद|कम्पोस्ट|कंपोस्ट|अलग|साफ़|साफ|धो|लपेट|सुरक्षित|"
    r"ख़तर|खतर|ज़हर|जहर|दोबारा|बेच|कबाड़|टूट|(?<!\S)(?:यह|ये|इसे|इस|इन|इसको)(?!\S)"
)


def is_on_topic(question: str | None, item_names: list[str]) -> bool:
    """False only for a question that uses no waste word and does not name the item."""
    if not question or not question.strip():
        return True
    if _TOPIC_EN.search(question) or _TOPIC_HI.search(question):
        return True
    q = question.casefold()
    return any(name.casefold() in q for name in item_names if len(name) >= 3)


def declined(kb: KnowledgeBase, verdict: Verdict) -> VerifiedOutput:
    """The answer for an off-topic question: KB facts plus a fixed note, no Llama text."""
    return VerifiedOutput(
        "declined", kb_output(verdict, why=kb.message("off_topic", verdict.language)), ()
    )


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
    if not is_on_topic(question, kb.names_of(verdict.item_id)):
        return Explanation(verified=declined(kb, verdict))
    language_name = kb.language_name(verdict.language)
    result = Explanation(verified=None)  # type: ignore[arg-type]
    for short in (False, True):
        prompt = build_prompt(
            verdict,
            attributes,
            language_name,
            question,
            short=short,
            include_bos=include_bos,
            kb=kb,
        )
        raw = generate(prompt)
        result.prompts.append(prompt)
        result.raw_outputs.append(raw)
        result.verified = verify(full_answer(raw), verdict)
        if result.verified.status != "fallback":
            break
    return result
