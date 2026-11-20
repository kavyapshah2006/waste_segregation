"""Ask Llama to explain a verdict, with one shorter retry, then fall back to KB text.

Free-text questions that are clearly not about waste are declined here, without calling
Llama: the stock 1B model answered every off-topic question in the baseline instead of
declining. The guard is a plain word list, not a classifier. A question passes if it uses a
waste word, names the item, says "this" / "यह", or uses a word from any item's names in the KB
(a one-letter slip is fine: speech heard "cartin" for "carton"); it only has to stop questions
like "What is the capital of France?". The prompt also tells Llama to decline, as a second line.
The first version knew only the word list and the current item's names, and on the phone it
declined everyday questions such as "what do I do with an old plastic carton".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
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
    r"collect\w*|pick\w*|burn\w*|broke\w*|break\w*|leak\w*|this|these|put\w*|go|goes|"
    r"going|keep\w*|stor\w*|handl\w*|rid|old|empty|used|plastic\w*)\b",
    re.I,
)
_TOPIC_HI = re.compile(
    r"कचर|कूड़|डिब्ब|डब्ब|फेंक|रीसा[यइ]कल|खाद|कम्पोस्ट|कंपोस्ट|अलग|साफ़|साफ|धो|लपेट|सुरक्षित|"
    r"ख़तर|खतर|ज़हर|जहर|दोबारा|बेच|कबाड़|टूट|डाल|रख|पुरान|प्लास्टिक|"
    r"(?<!\S)(?:यह|ये|इसे|इस|इन|इसको)(?!\S)"
)

# KB name words too general to show a question is about waste ("how much water should I drink").
_VOCAB_STOP = frozenset(
    "item items small cold light soft water control pieces drink household band cell strip "
    "pack with from other पानी किया".split()
)
_VOCAB_MIN = 4  # code points
_FUZZY_MIN = 5  # words this long may differ by one letter


def words(text: str) -> list[str]:
    """Lower-cased runs of letters, marks and digits (keeps Hindi vowel signs in the word)."""
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in "LMN":
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def topic_vocabulary(names: Iterable[str]) -> frozenset[str]:
    """Words of the KB's item names and aliases that count as waste words."""
    return frozenset(
        w for name in names for w in words(name) if len(w) >= _VOCAB_MIN and w not in _VOCAB_STOP
    )


def _one_edit_apart(a: str, b: str) -> bool:
    """True if a and b differ by at most one inserted, deleted or replaced letter."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return edits + (len(b) - j) + (len(a) - i) <= 1


def is_on_topic(
    question: str | None, item_names: list[str], vocabulary: Iterable[str] = ()
) -> bool:
    """False only for a question with no waste word, no item name and no KB name word.

    vocabulary: topic_vocabulary(kb.all_names()), the words of every item's names.
    """
    if not question or not question.strip():
        return True
    if _TOPIC_EN.search(question) or _TOPIC_HI.search(question):
        return True
    q = question.casefold()
    if any(name.casefold() in q for name in item_names if len(name) >= 3):
        return True
    vocab = vocabulary if isinstance(vocabulary, (set, frozenset)) else frozenset(vocabulary)
    for w in words(question):
        if len(w) < _VOCAB_MIN:
            continue
        if w in vocab:
            return True
        if len(w) >= _FUZZY_MIN and any(
            len(v) >= _FUZZY_MIN and _one_edit_apart(w, v) for v in vocab
        ):
            return True
    return False


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


# Extra tries with sampling when the greedy answer fails the checks. In the experiments
# (variant shots_kind+rules+retry_greedy_sample) this took SpinQuant 1B from 95.5% to 96.6%
# useful answers for 1.09 generations per question on average.
SAMPLED_RETRIES = 2


def explain(
    kb: KnowledgeBase,
    verdict: Verdict,
    attributes: dict | None,
    generate: Generator,
    question: str | None = None,
    include_bos: bool = True,
    sample: Generator | None = None,
) -> Explanation:
    """Greedy answer first; unparseable -> one short-prompt retry; still not "ok" and a
    sampling generator given -> up to SAMPLED_RETRIES sampled tries, first "ok" wins."""
    if not is_on_topic(question, kb.names_of(verdict.item_id), kb.topic_vocabulary()):
        return Explanation(verified=declined(kb, verdict))
    language_name = kb.language_name(verdict.language)
    result = Explanation(verified=None)  # type: ignore[arg-type]

    def attempt(gen: Generator, short: bool) -> VerifiedOutput:
        prompt = build_prompt(
            verdict,
            attributes,
            language_name,
            question,
            short=short,
            include_bos=include_bos,
            kb=kb,
        )
        raw = gen(prompt)
        result.prompts.append(prompt)
        result.raw_outputs.append(raw)
        return verify(full_answer(raw), verdict)

    best = attempt(generate, False)
    if best.status == "fallback":
        second = attempt(generate, True)
        if second.status != "fallback":
            best = second
    if best.status != "ok" and sample is not None:
        for _ in range(SAMPLED_RETRIES):
            tried = attempt(sample, False)
            if tried.status == "ok":
                best = tried
                break
            if best.status == "fallback" and tried.status == "corrected":
                best = tried  # some checked Llama text beats none
    result.verified = best
    return result
