"""Offline Hindi speech round trip on the laptop (FuturePlan WP9).

    bash models/speech/download.sh
    uv run --group speech python models/speech/check_hindi.py --label laptop_tts_roundtrip

1. Text-to-speech (Piper hi_IN "pratham", test-only voice) reads every KB item's Hindi name and
   a few spoken questions.
2. Dolphin CTC (base and small, int8) transcribes the audio.
3. Scores: character error rate; whether exact KB search on the transcript finds the right
   item; and whether pipeline/voice.py (pronunciation-key matching) does, which is what the
   app uses.

Synthetic speech is clean and from one voice, so these numbers are an upper bound on real
users, not a measurement of them; repeat with recordings from native speakers. Speeds are
laptop speeds and are not reported for the phone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "models" / "speech" / "out"
RESULTS = REPO / "eval" / "results"


def _dolphin(so, d: Path, threads: int):
    return so.OfflineRecognizer.from_dolphin_ctc(
        model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads
    )


def _whisper(so, d: Path, threads: int):
    # Whisper can be told the language; the CTC models guess it per utterance.
    return so.OfflineRecognizer.from_whisper(
        encoder=str(d / "base-encoder.int8.onnx"), decoder=str(d / "base-decoder.int8.onnx"),
        tokens=str(d / "base-tokens.txt"), language="hi", task="transcribe", num_threads=threads,
    )  # fmt: skip


def _omnilingual(so, d: Path, threads: int):
    return so.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads
    )


ASR = {
    "dolphin_base_int8": (
        _dolphin,
        OUT / "sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",
    ),
    "dolphin_small_int8": (
        _dolphin,
        OUT / "sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02",
    ),
    "whisper_base_int8_hi": (_whisper, OUT / "sherpa-onnx-whisper-base"),
    "omnilingual_300m_ctc_v2_int8": (
        _omnilingual,
        OUT / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05",
    ),
}
TTS = OUT / "vits-piper-hi_IN-pratham-medium-int8"
SENTENCE = "मेरे पास {} है, इसे कहाँ डालूँ?"
# Spoken questions that name no item: the matcher should not be confident about any.
QUESTIONS = [
    "यह किस डिब्बे में जाएगा?",
    "क्या यह रीसायकल हो सकता है?",
    "इसे फेंकने से पहले कैसे तैयार करूँ?",
    "इसे कहाँ फेंकना चाहिए?",
    "क्या यह खतरनाक है?",
]


def normalise(text: str) -> str:
    """For scoring: NFC, no tags like <hi>, no punctuation, single spaces."""
    text = unicodedata.normalize("NFC", re.sub(r"<[^>]*>", "", text))
    text = re.sub(r"[।॥?!,.;:\"'()\-]", " ", text)
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    a, b = normalise(ref).replace(" ", ""), normalise(hyp).replace(" ", "")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)

    import sherpa_onnx as so

    from kb.build_kb import build
    from pipeline import voice
    from pipeline.kb import KnowledgeBase

    db = OUT / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)

    tts = so.OfflineTts(
        so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(
                    model=str(TTS / "hi_IN-pratham-medium.onnx"),
                    tokens=str(TTS / "tokens.txt"),
                    data_dir=str(TTS / "espeak-ng-data"),
                ),
                num_threads=args.threads,
            )
        )
    )
    # Each item twice: the bare Hindi name, and inside a spoken sentence.
    samples = [("name", item_id, kb.item(item_id)["names"]["hi"]) for item_id in kb.item_ids]
    samples += [
        ("sentence", item_id, SENTENCE.format(kb.item(item_id)["names"]["hi"]))
        for item_id in kb.item_ids
    ]
    samples += [("question", None, q) for q in QUESTIONS]
    audio, tts_seconds, audio_seconds = [], 0.0, 0.0
    for kind, item_id, text in samples:
        t = time.time()
        gen = tts.generate(text, sid=0, speed=1.0)
        tts_seconds += time.time() - t
        wav = np.asarray(gen.samples, dtype=np.float32)
        audio_seconds += len(wav) / gen.sample_rate
        audio.append((kind, item_id, text, wav, gen.sample_rate))

    n_items = len(kb.item_ids)
    report: dict = {
        "audio": "synthetic: Piper hi_IN pratham (test-only voice)",
        "utterances": len(audio),
        "items": n_items,
        "sentence_template": SENTENCE,
        "tts_realtime_factor_laptop": round(tts_seconds / audio_seconds, 3),
    }
    for name, (make, path) in ASR.items():
        if not path.exists():
            print(f"skip {name}: {path} missing (run download.sh)")
            continue
        rec = make(so, path, args.threads)
        cers, t0, examples = [], time.time(), []
        hits = {
            k: {"exact_top1": 0, "match_top1": 0, "match_top3": 0} for k in ("name", "sentence")
        }
        confident_on_questions = 0
        for kind, item_id, text, wav, sr in audio:
            stream = rec.create_stream()
            stream.accept_waveform(sr, wav)
            rec.decode_stream(stream)
            hyp = stream.result.text
            cers.append(cer(text, hyp))
            matches = voice.match(kb, hyp)
            matched = [m["item_id"] for m in matches]
            if item_id is not None:
                exact = kb.search(normalise(hyp), lang="hi")
                hits[kind]["exact_top1"] += bool(exact) and exact[0] == item_id
                hits[kind]["match_top1"] += bool(matched) and matched[0] == item_id
                hits[kind]["match_top3"] += item_id in matched
            elif matches and not matches[0]["confirm"]:
                confident_on_questions += 1  # a question that names no item
            if kind != "sentence" and (len(examples) < 6 or item_id is None):
                examples.append({"said": text, "heard": normalise(hyp), "matched": matched[:1]})
        report[name] = {
            "model_mb": round(sum(f.stat().st_size for f in path.glob("*int8.onnx")) / 1e6, 1),
            "median_cer": round(float(np.median(cers)), 3),
            **{
                f"{kind}_{k}": round(v / n_items, 3)
                for kind, d in hits.items()
                for k, v in d.items()
            },
            "confident_match_on_item_free_questions": confident_on_questions,
            "asr_realtime_factor_laptop": round((time.time() - t0) / audio_seconds, 3),
            "examples": examples,
        }
        print(name, {k: v for k, v in report[name].items() if k != "examples"})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.label:
        out = RESULTS / f"speech_{args.label}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
