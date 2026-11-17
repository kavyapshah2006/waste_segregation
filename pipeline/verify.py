"""Check Llama's answer against the knowledge base (FuturePlan section 5.5, verifier rules).

The knowledge base always wins on facts: item, stream, bin, steps, hazard and drop-off are
taken from the Verdict. Llama is asked only for `why` and `tip` (pipeline/prompt.py), and
they are dropped if they name a different bin. If Llama adds `stream`, `bin` or `steps` keys
anyway, they are checked too; a wrong stream or bin also tightens the free-text check.
`why` / `tip` are also dropped if a clause negates the verdict's own bin or stream or one
of its steps, if a recyclable or compostable claim disagrees with the KB flag, or if a
clause tells the user to put waste in a drain, toilet, road or river, or to burn it.
Every disagreement is recorded as a contradiction so the raw contradiction rate can be
measured.

verify() takes the complete answer text; for runner output use prompt.full_answer() first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pipeline.kb import Verdict

MAX_FREE_TEXT = 400

# "declined": the question was off-topic, so Llama was not asked (pipeline/assistant.py).
Status = Literal["ok", "corrected", "fallback", "declined"]

# Bin mentions in free text, mapped to colour ids. Hindi uses both direct and oblique forms.
_COLOUR_WORDS = {
    "green": "green",
    "blue": "blue",
    "red": "red",
    "black": "black",
    "yellow": "yellow",
    "white": "white",
    "grey": "grey",
    "gray": "grey",
    "हरा": "green",
    "हरे": "green",
    "नीला": "blue",
    "नीले": "blue",
    "लाल": "red",
    "काला": "black",
    "काले": "black",
    "पीला": "yellow",
    "पीले": "yellow",
    "सफ़ेद": "white",
    "सफेद": "white",
}
_EN_BIN = re.compile(r"\b(green|blue|red|black|yellow|white|grey|gray)\s+(?:dust)?bins?\b", re.I)
_HI_BIN = re.compile(r"(हरा|हरे|नीला|नीले|लाल|काला|काले|पीला|पीले|सफ़ेद|सफेद)\s+(?:डिब्बे|डिब्बा|कूड़ेदान)")


@dataclass(frozen=True)
class VerifiedOutput:
    status: Status
    output: dict
    contradictions: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "output": self.output,
            "contradictions": list(self.contradictions),
        }


_SPECIAL_TOKENS = ("<|eot_id|>", "<|end_of_text|>")


def _try_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_json(text: str) -> dict | None:
    """Return the first JSON object in the text, ignoring prose and code fences.

    One repair is made: an object that ends after a complete value but lacks its final "}"
    (seen from the stock 1B model) is closed. Text cut off inside a string is not repaired,
    because the cut-off sentence would be shown to the user.
    """
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "")
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    value = _try_object(text[start : i + 1])
                    if value is not None:
                        return value
                    break
        else:
            if depth == 1 and not in_string:
                value = _try_object(text[start:].rstrip() + "}")
                if value is not None:
                    return value
        start = text.find("{", start + 1)
    return None


def bins_mentioned(text: str) -> set[str]:
    """Colour ids of bins named in free text, e.g. 'blue bin' or 'नीले डिब्बे'."""
    found = {_COLOUR_WORDS[m.group(1).lower()] for m in _EN_BIN.finditer(text)}
    found |= {_COLOUR_WORDS[m.group(1)] for m in _HI_BIN.finditer(text)}
    return found


# Content checks on why/tip, clause by clause. Deliberately strict: a dropped sentence only
# costs the explanation (the KB facts are still shown), a wrong one misleads the user.
_CLAUSE = re.compile(
    r"[.;:!?,।\n]+|\s+(?:and|but|or|so\s+that|because|और|लेकिन|या|ताकि|क्योंकि|जिससे)\s+", re.I
)
_NEG_EN = re.compile(
    r"\b(?:not|never|no|don['’]t|doesn['’]t|shouldn['’]t|cannot|can['’]t|won['’]t|isn['’]t|"
    r"aren['’]t|avoid)\b",
    re.I,
)
_NEG_HI = re.compile(r"नहीं|मत|(?<!\S)न(?!\S)")
_STREAM_WORDS = {
    "wet": (r"\bwet\s+waste\b", r"गील[ाे]\s+कचर[ाे]"),
    "dry": (r"\bdry\s+waste\b", r"सूख[ाे]\s+कचर[ाे]"),
    "sanitary": (r"\bsanitary\s+waste\b", r"सैनिटरी\s+कचर[ाे]"),
    "special_care": (r"\bspecial[\s-]+care\s+waste\b", r"विशेष\s+देखभाल\s+(?:वाले\s+)?कचर[ाे]"),
}
_RECYCL_HI = r"रीसा[यइ]कल"
_NOT_RECYCLABLE_HI = re.compile(rf"{_RECYCL_HI}\s+(?:\S+\s+){{0,3}}?(?:नहीं|न)(?!\S)")
_HARD = re.compile(
    r"मुश्किल|कठिन|शायद\s+ही|\bdifficult\b|\bhard\s+to\b|\brarely\b|\bseldom\b|\bhardly\b", re.I
)
_NOT_RECYCLABLE_EN = re.compile(
    r"\b(?:not|non-?|never|cannot|can['’]t|isn['’]t|aren['’]t)\s*(?:be\s+|being\s+)?recycl", re.I
)
_RECYCLABLE_EN = re.compile(
    r"\b(?:is|are|it['’]s)\s+recyclable\b|\bcan\s+be\s+recycled\b|\b(?:is|are|gets?)\s+recycled\b",
    re.I,
)
_RECYCLABLE_HI = re.compile(rf"{_RECYCL_HI}.*(?:हो\s+सकत|किया\s+जा\s+सकत|होत[ाीे]|हो\s+जात)")


def _clauses(text: str) -> list[str]:
    return [c for c in _CLAUSE.split(text) if c.strip()]


def _negated(clause: str) -> bool:
    return bool(_NEG_EN.search(clause) or _NEG_HI.search(clause))


_INSTEAD_OF_EN = re.compile(r"\binstead\s+of\b", re.I)  # negates what follows
_INSTEAD_OF_HI = re.compile(r"के\s+बजाय|की\s+जगह")  # negates what comes before


def _names_verdict(part: str, verdict: Verdict) -> bool:
    en, hi = _STREAM_WORDS[verdict.stream]
    return (
        verdict.bin_colour in bins_mentioned(part)
        or bool(re.search(en, part, re.I))
        or bool(re.search(hi, part))
    )


def negates_verdict(text: str, verdict: Verdict) -> bool:
    """True if a clause says not to use the verdict's own bin or stream ('do not put it in the
    blue bin' or '... instead of the blue bin' when blue is right)."""
    for clause in _clauses(text):
        if _negated(clause) and _names_verdict(clause, verdict):
            return True
        if (m := _INSTEAD_OF_EN.search(clause)) and _names_verdict(clause[m.end() :], verdict):
            return True
        if (m := _INSTEAD_OF_HI.search(clause)) and _names_verdict(clause[: m.start()], verdict):
            return True
    return False


_RECYCLE_TOPIC = re.compile(rf"recycl|{_RECYCL_HI}", re.I)


def _claims(text: str, negative: re.Pattern, positive: re.Pattern) -> set[bool]:
    """Every claim the clauses make: False for 'not X', True for 'can be X'. Negated or hedged
    clauses ('rarely', 'hard to') make no positive claim."""
    claims: set[bool] = set()
    for clause in _clauses(text):
        if negative.search(clause):
            claims.add(False)
        elif not _negated(clause) and not _HARD.search(clause) and positive.search(clause):
            claims.add(True)
    return claims


def recyclable_claims(text: str) -> set[bool]:
    """Recyclability claims in the text. A bare leading 'No,' / 'नहीं,' or 'Yes,' / 'हाँ,'
    counts as a claim when the text is about recycling ('No, glass can be recycled' makes
    both claims, so it contradicts the KB either way)."""
    claims = _claims(
        text,
        re.compile(f"{_NOT_RECYCLABLE_EN.pattern}|{_NOT_RECYCLABLE_HI.pattern}", re.I),
        re.compile(f"{_RECYCLABLE_EN.pattern}|{_RECYCLABLE_HI.pattern}", re.I),
    )
    clauses = _clauses(text)
    if clauses and _RECYCLE_TOPIC.search(text):
        lead = clauses[0].strip().casefold()
        if lead in {"no", "नहीं"}:
            claims.add(False)
        elif lead in {"yes", "हाँ", "हां"}:
            claims.add(True)
    return claims


_NOT_COMPOSTABLE = re.compile(
    r"\b(?:not|non-?|never|cannot|can['’]t|isn['’]t|aren['’]t)\s*(?:be\s+)?compost"
    r"|(?:खाद|कम्पोस्ट|कंपोस्ट)\s+(?:\S+\s+){0,3}?(?:नहीं|न)(?!\S)",
    re.I,
)
_COMPOSTABLE = re.compile(
    r"\b(?:is|are|it['’]s)\s+compostable\b|\bcan\s+be\s+composted\b"
    r"|(?:खाद|कम्पोस्ट|कंपोस्ट)\s+(?:बन\s+सकत|बनाई\s+जा\s+सकत|बनत)",
    re.I,
)

# Telling the user to put waste in a drain, toilet, road or river, or to burn it.
_UNSAFE_PLACE = re.compile(
    r"\b(?:drains?|sewers?|toilets?|nalas?|rivers?|lakes?|roadside|on\s+the\s+road|open\s+ground)\b"
    r"|नाल[ाीियोंे]*|सीवर|शौचालय|सड़क|नदी|तालाब",
    re.I,
)
_PUT = re.compile(
    r"\b(?:put|pour|flush|throw|dump|tip|empty|drop|leave)\b"
    r"|डालें|डाल\s+दें|बहाएँ|बहा\s+दें|फेंकें|फेंक\s+दें|छोड़ें|छोड़\s+दें",
    re.I,
)
_BURN = re.compile(
    r"(?:^|\byou\s+(?:can|should|may)\s+|\bplease\s+)burn\b|जलाएँ|जलाइए|जला\s+दें", re.I
)


def unsafe_instruction(text: str, verdict: Verdict) -> bool:
    """True if a clause tells the user to put waste somewhere unsafe or to burn it. Clauses
    that repeat the KB's own text are allowed (diapers: empty solid waste into the toilet)."""
    kb_clauses = [
        _content_words(c)
        for t in (*verdict.steps, *verdict.notes, verdict.hazard or "")
        for c in _clauses(t)
    ]
    for clause in _clauses(text):
        if _negated(clause) or _repeats(clause, kb_clauses):
            continue
        if (_UNSAFE_PLACE.search(clause) and _PUT.search(clause)) or _BURN.search(clause.strip()):
            return True
    return False


def compostable_claims(text: str) -> set[bool]:
    """Compostability claims in the text (no yes/no lead rule: the yes/no question asked is
    about recycling)."""
    return _claims(text, _NOT_COMPOSTABLE, _COMPOSTABLE)


_STOP = {
    "the", "and", "or", "a", "an", "it", "its", "them", "they", "this", "that", "with",
    "from", "into", "onto", "before", "after", "your", "you", "out", "off", "for", "of",
    "to", "in", "on", "at", "be", "is", "are", "if", "can", "then", "so", "do", "not",
    "को", "से", "में", "और", "के", "की", "का", "है", "हैं", "यह", "इसे", "इस", "पहले",
    "दें", "करें", "कर", "या", "पर", "ही", "भी", "न", "नहीं", "मत",
}  # fmt: skip


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[\wऀ-ॿ]+", text.casefold())
    return {w for w in words if w not in _STOP and len(w) >= 3}


def _repeats(clause: str, kb_clauses: list[set[str]], share: float = 0.8) -> bool:
    """True if the clause carries at least `share` of some KB clause's content words."""
    said = _content_words(clause)
    return any(len(words) >= 2 and len(words & said) >= share * len(words) for words in kb_clauses)


def negates_step(text: str, verdict: Verdict) -> bool:
    """True if a negated clause matches a KB step ('Do not remove stickers' when a step says
    'Remove stickers from fruit'). Steps that are themselves negative are skipped."""
    step_words = [
        _content_words(clause)
        for step in verdict.steps
        for clause in _clauses(step)
        if not _negated(clause)
    ]
    return any(_negated(c) and _repeats(c, step_words) for c in _clauses(text))


def _norm(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", value.strip().casefold())


def _stream_key(value: str) -> str:
    return _norm(value).removesuffix("_waste").removesuffix("_कचरा")


def _matches_stream(value, verdict: Verdict) -> bool:
    return _stream_key(str(value)) in {verdict.stream, _stream_key(verdict.stream_name)}


def _matches_bin(value, verdict: Verdict) -> bool:
    v = _norm(str(value))
    for suffix in ("_dustbin", "_bin", "_डिब्बा", "_डिब्बे"):
        v = v.removesuffix(suffix)
    return v in {verdict.bin_colour, _norm(verdict.bin_colour_name)} or (
        _COLOUR_WORDS.get(v) == verdict.bin_colour
    )


def _clean_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not value:
        return None
    if len(value) > MAX_FREE_TEXT:
        value = value[:MAX_FREE_TEXT].rsplit(" ", 1)[0] + "…"
    return value


def kb_output(verdict: Verdict, why: str | None = None, tip: str | None = None) -> dict:
    """The answer shown to the user: KB facts plus Llama's free text where it is allowed."""
    return {
        "item": verdict.name,
        "stream": verdict.stream,
        "bin": verdict.bin_colour,
        "steps": list(verdict.steps),
        "hazard": verdict.hazard,
        "drop_off": verdict.drop_off_text,
        "why": why,
        "tip": tip,
        "language": verdict.language,
    }


def verify(raw: str, verdict: Verdict) -> VerifiedOutput:
    data = extract_json(raw)
    if data is None:
        return VerifiedOutput("fallback", kb_output(verdict), ("unparseable",))

    contradictions: list[str] = []
    if "stream" in data and not _matches_stream(data["stream"], verdict):
        contradictions.append("stream")
    if "bin" in data and not _matches_bin(data["bin"], verdict):
        contradictions.append("bin")

    # Bins Llama may name in its text: the verdict's bin, plus any bin the KB entry itself
    # mentions (greasy cardboard: green for the soiled part, blue for the clean part).
    # Once Llama has the stream or bin wrong, only the verdict's bin is allowed.
    allowed = {verdict.bin_colour}
    if not contradictions:
        allowed |= bins_mentioned(" ".join((*verdict.steps, *verdict.notes)))

    steps = data.get("steps")
    if isinstance(steps, list):
        step_text = " ".join(s for s in steps if isinstance(s, str))
        if bins_mentioned(step_text) - allowed:
            contradictions.append("steps_bin")

    free: dict[str, str | None] = {}
    for key in ("why", "tip"):
        value = _clean_text(data.get(key))
        if value and bins_mentioned(value) - allowed:
            contradictions.append(f"{key}_bin")
            value = None
        if value and negates_verdict(value, verdict):
            contradictions.append(f"{key}_negates")
            value = None
        if value and (not verdict.recyclable) in recyclable_claims(value):
            contradictions.append(f"{key}_recyclable")
            value = None
        if value and (not verdict.compostable) in compostable_claims(value):
            contradictions.append(f"{key}_compostable")
            value = None
        if value and negates_step(value, verdict):
            contradictions.append(f"{key}_negates_step")
            value = None
        if value and unsafe_instruction(value, verdict):
            contradictions.append(f"{key}_unsafe")
            value = None
        free[key] = value

    # When one field gets the facts wrong, the other usually repeats the mistake in other
    # words (seen in the stock-model run), so neither is shown.
    if any(c.startswith(("why_", "tip_")) for c in contradictions):
        free = {"why": None, "tip": None}

    status: Status = "corrected" if contradictions else "ok"
    return VerifiedOutput(status, kb_output(verdict, **free), tuple(contradictions))
