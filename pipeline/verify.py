"""Check Llama's answer against the knowledge base (FuturePlan section 5.5, verifier rules).

The knowledge base always wins on facts: item, stream, bin, steps, hazard and drop-off are
taken from the Verdict. Only `why` and `tip` come from Llama, and they are dropped if they
name a different bin. Every disagreement is recorded as a contradiction so the raw
contradiction rate can be measured.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pipeline.kb import Verdict

MAX_FREE_TEXT = 400

Status = Literal["ok", "corrected", "fallback"]

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


def extract_json(text: str) -> dict | None:
    """Return the first balanced JSON object in the text, ignoring prose and code fences."""
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
                    try:
                        value = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    return None


def bins_mentioned(text: str) -> set[str]:
    """Colour ids of bins named in free text, e.g. 'blue bin' or 'नीले डिब्बे'."""
    found = {_COLOUR_WORDS[m.group(1).lower()] for m in _EN_BIN.finditer(text)}
    found |= {_COLOUR_WORDS[m.group(1)] for m in _HI_BIN.finditer(text)}
    return found


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
        free[key] = value

    status: Status = "corrected" if contradictions else "ok"
    return VerifiedOutput(status, kb_output(verdict, **free), tuple(contradictions))
