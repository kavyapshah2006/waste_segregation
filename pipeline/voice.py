"""Match a speech transcript to KB items (FuturePlan WP9). Reference for the Kotlin app.

Offline speech recognisers often write Hindi item names in Latin letters ("gatte ka dibba"
for गत्ते का डिब्बा) or with small spelling slips (दबा for दवा). Exact KB search misses both. This
compares a rough pronunciation key instead: Devanagari is romanised, both sides are simplified
(long vowels, doubled letters, w/v, ai/e ...), and items are ranked by edit-distance similarity
against every name and alias in every language.

The result is a ranked suggestion list; the app shows the top items for the user to confirm
when the best score is below `CONFIRM_BELOW`.
"""

from __future__ import annotations

import re
import unicodedata

from pipeline.kb import KnowledgeBase

CONFIRM_BELOW = 0.85  # below this similarity, ask the user to pick from the top 3
MIN_SCORE = 0.5  # below this, no suggestion at all

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "ळ": "l",
}  # fmt: skip
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}  # fmt: skip
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o",
}  # fmt: skip
_NUKTA_MAP = {"ड": "r", "ढ": "rh", "फ": "f", "ज": "j", "क": "k", "ख": "kh", "ग": "g"}
_VIRAMA, _NUKTA = "्", "़"
_NASAL = {"ं": "n", "ँ": "n", "ः": "h"}


# Recognisers sometimes slip into another script mid-word. Bengali, Gurmukhi, Gujarati, Odia,
# Tamil, Telugu, Kannada and Malayalam share Devanagari's Unicode layout at fixed offsets
# (U+0980 ... U+0D7F), so they map back by position; Urdu gets a rough letter table.
_BRAHMIC_START, _BRAHMIC_END = 0x0980, 0x0D7F
_URDU = {
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch",
    "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f", "ق": "k",
    "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "v", "ہ": "h",
    "ھ": "h", "ة": "h", "ی": "i", "ي": "i", "ے": "e", "ئ": "i",
}  # fmt: skip


def _to_devanagari(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if _BRAHMIC_START <= cp <= _BRAHMIC_END:
            ch = chr(0x0900 + (cp - _BRAHMIC_START) % 0x80)
        out.append(_URDU.get(ch, ch))
    return "".join(out)


def romanise(text: str) -> str:
    """Rough, lossy romanisation of Devanagari (other Indic scripts and Urdu first mapped)."""
    text = unicodedata.normalize("NFD", _to_devanagari(unicodedata.normalize("NFC", text)))
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _CONSONANTS:
            sound = _CONSONANTS[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == _NUKTA:
                sound = _NUKTA_MAP.get(ch, sound)
                i += 1
                nxt = text[i + 1] if i + 1 < len(text) else ""
            after = text[i + 2] if i + 2 < len(text) else ""
            prev_vowel = bool(out) and out[-1][-1:] in "aeiou"
            out.append(sound)
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                i += 1
            elif nxt == _VIRAMA:
                i += 1
            elif not ("ऀ" <= nxt <= "ॿ") or nxt in _NASAL:
                # End of word (or a nasal mark): Hindi drops the final inherent vowel.
                pass
            elif prev_vowel and nxt in _CONSONANTS and after in _MATRAS:
                # Medial schwa deletion: vowel + C(a) + C + vowel sign -> the "a" is silent
                # (बैटरी is "baitri", not "baitari").
                pass
            else:
                out.append("a")
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _NASAL:
            out.append(_NASAL[ch])
        elif ch in _MATRAS:  # stray matra
            out.append(_MATRAS[ch])
        elif ch == _NUKTA or ch == _VIRAMA:
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_SIMPLIFY = [
    ("aa", "a"), ("ii", "i"), ("ee", "i"), ("uu", "u"), ("oo", "u"), ("ai", "e"), ("au", "o"),
    ("ph", "f"), ("w", "v"), ("z", "j"), ("q", "k"), ("c", "k"), ("y", "i"),
]  # fmt: skip


def key(text: str) -> str:
    """Pronunciation key: romanised, lower case, letters only, simplified, no doubled letters."""
    s = romanise(text).lower()
    s = re.sub(r"[^a-z]", "", s)
    s = s.replace("chh", "ch").replace("kh", "k").replace("gh", "g").replace("th", "t")
    s = s.replace("dh", "d").replace("bh", "b").replace("sh", "s")
    for a, b in _SIMPLIFY:
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance between two keys."""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


MIN_WINDOW_NAME = 5  # shorter keys ("rike", "pani") turn up inside unrelated sentences
CONFIDENT_MARGIN = 0.1  # the top item must also beat the next one by this much


def window_similarity(heard: str, name: str) -> float:
    """Best similarity of `name` to any stretch of `heard` about as long as the name, for
    sentences ("purani batri ka kya karu"). Short names need a full match: they would fit
    somewhere in almost any sentence."""
    full = similarity(heard, name)
    if len(name) < MIN_WINDOW_NAME or len(heard) <= len(name):
        return full
    best = full
    for width in (len(name) - 1, len(name), len(name) + 1):
        for start in range(0, len(heard) - width + 1):
            best = max(best, similarity(heard[start : start + width], name))
    return best


def match(kb: KnowledgeBase, transcript: str, k: int = 3) -> list[dict]:
    """Items ranked by the best similarity of the transcript (or a stretch of it) to any of
    their names. Items scoring below MIN_SCORE are left out."""
    heard = key(transcript)
    if not heard:
        return []
    best: dict[str, float] = {}
    for item_id in kb.item_ids:
        best[item_id] = max(window_similarity(heard, key(n)) for n in kb.names_of(item_id))
    ranked = sorted(best.items(), key=lambda x: (-x[1], x[0]))
    # Confident only if clearly above the rest; a near tie goes to the user to pick.
    clear = len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= CONFIDENT_MARGIN
    return [
        {
            "item_id": i,
            "score": round(s, 3),
            "confirm": not (rank == 0 and clear and s >= CONFIRM_BELOW),
        }
        for rank, (i, s) in enumerate(ranked[:k])
        if s >= MIN_SCORE
    ]
