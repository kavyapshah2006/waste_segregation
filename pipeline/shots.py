"""Worked examples (few-shot turns) for the Llama prompt, written from the KB.

Chosen by the answer-quality experiments (eval/llm_quality/experiments.py, variant
"shots_kind+rules"): before the real question, the prompt shows two finished exchanges about
two fixed items (newspaper, AA battery) with the SAME kind of question and language. On the
stock SpinQuant 1B this lifted the useful-answer rate on held-out items from 69% to 95.5%.

The example answers come from the same templates as the fine-tuning data
(models/llm/synth_data/generate.py imports them from here), and both example items are in the
training split, never in validation.

The Kotlin port is `Shots` in android/core; the golden fixtures pin the exact text.
"""

from __future__ import annotations

import json
import re

from pipeline.kb import KnowledgeBase, Verdict

# Canonical question per kind and language. "explain" uses the default question.
QUESTIONS = {
    "explain": {"en": None, "hi": None},
    "which_bin": {"en": "Which bin does this go in?", "hi": "यह किस डिब्बे में जाएगा?"},
    "recyclable": {"en": "Is this recyclable?", "hi": "क्या यह रीसायकल हो सकता है?"},
    "prepare": {
        "en": "How should I prepare this before throwing it away?",
        "hi": "इसे फेंकने से पहले कैसे तैयार करूँ?",
    },
    "off_topic": {"en": "What is the capital of France?", "hi": "आज मौसम कैसा रहेगा?"},
}

SHOT_ITEMS = ("newspaper", "aa_battery")
LANGUAGES = ("en", "hi")  # the languages the example answers are written in

# Hindi oblique colour forms ("नीले डिब्बे", not "नीला डिब्बे").
HI_COLOUR_OBLIQUE = {"green": "हरे", "blue": "नीले", "red": "लाल", "black": "काले"}

TIPS = {
    "wet": {
        "en": "Keep a small lidded bin in the kitchen for wet waste.",
        "hi": "गीले कचरे के लिए रसोई में ढक्कन वाला छोटा डिब्बा रखें।",
    },
    "dry": {
        # No "can be recycled" here: dry waste is not always recyclable (chips packets).
        "en": "Keep dry waste clean and dry so it is easy to sort.",
        "hi": "सूखे कचरे को साफ़ और सूखा रखें ताकि उसे छाँटना आसान हो।",
    },
    "sanitary": {
        "en": "Always wrap sanitary waste before putting it in the bin.",
        "hi": "सैनिटरी कचरे को डिब्बे में डालने से पहले हमेशा लपेटें।",
    },
    "special_care": {
        "en": "Store special care waste safely until the next collection drive.",
        "hi": "विशेष देखभाल कचरे को अगले संग्रह अभियान तक सुरक्षित रखें।",
    },
}

_KINDS = (
    ("recyclable", re.compile(r"recycl|रीसा[यइ]कल", re.IGNORECASE)),
    ("prepare", re.compile(r"prepar|before\s+(?:throw|dispos|bin)|rinse|तैयार|पहले", re.IGNORECASE)),
    ("which_bin", re.compile(r"\b(?:dust)?bins?\b|डिब्ब|कूड़ेदान", re.IGNORECASE)),
)


def question_kind(question: str | None) -> str:
    """explain (no question), recyclable, prepare, which_bin, or other, from the words used."""
    if not question or not question.strip():
        return "explain"
    for kind, pattern in _KINDS:
        if pattern.search(question):
            return kind
    return "other"


def shot_kind(question: str | None) -> str:
    """Which examples to show: the question's own kind, or "explain" for anything else."""
    kind = question_kind(question)
    return kind if kind in ("which_bin", "recyclable", "prepare") else "explain"


def _bin_phrase(v: Verdict) -> str:
    if v.language == "hi":
        return f"{HI_COLOUR_OBLIQUE.get(v.bin_colour, v.bin_colour_name)} डिब्बे"
    return f"{v.bin_colour_name} bin"


def why_for(kind: str, v: Verdict, kb: KnowledgeBase) -> str:
    hi = v.language == "hi"
    if kind == "off_topic":
        # The app declines these before asking Llama; training on them is a second line.
        return kb.message("off_topic", v.language)
    if kind == "which_bin":
        lead = (
            f"यह {_bin_phrase(v)} ({v.stream_name}) में जाएगा।"
            if hi
            else f"It goes in the {_bin_phrase(v)} ({v.stream_name.lower()})."
        )
    elif kind == "recyclable":
        if hi:
            lead = (
                "हाँ, इसे रीसायकल किया जा सकता है।"
                if v.recyclable
                else "नहीं, इसे रीसायकल नहीं किया जा सकता।"
            )
        else:
            lead = "Yes, it can be recycled." if v.recyclable else "No, it cannot be recycled."
    elif kind == "prepare":
        lead = (f"पहला कदम: {v.steps[0]}।" if hi else f"First: {v.steps[0]}.").replace("।।", "।")
    else:
        lead = ""
    parts = [lead, *v.notes, v.impact]
    return " ".join(p for p in parts if p).strip()


def answer_for(kind: str, v: Verdict, kb: KnowledgeBase) -> str:
    """The whole answer object as JSON text: {"why": ..., "tip": ...}."""
    tip = TIPS[v.stream][v.language] if v.hazard is None else v.hazard
    return json.dumps({"why": why_for(kind, v, kb), "tip": tip}, ensure_ascii=False)


def shot_turns(kb: KnowledgeBase, question: str | None, language: str) -> list[tuple[str, str]]:
    """(user message, full assistant answer) for each example item, for this question's kind.
    Examples exist in English and Hindi only; other languages get none."""
    # Imported here: pipeline.prompt imports this module.
    from pipeline.prompt import user_message

    if language not in LANGUAGES:
        return []
    kind = shot_kind(question)
    turns = []
    for item_id in SHOT_ITEMS:
        verdict = kb.resolve(item_id, {}, language)
        user = user_message(verdict, {}, kb.language_name(language), QUESTIONS[kind][language])
        turns.append((user, answer_for(kind, verdict, kb)))
    return turns
