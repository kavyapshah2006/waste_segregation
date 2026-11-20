"""Score Llama answers against the knowledge base (WP8 grounding evaluation).

Cases are the held-out validation examples from models/llm/synth_data/generate.py: items the
fine-tune never sees, in English and Hindi, across all question kinds.

Usage:
    # 1. Write the cases (with exact prompts) to run on any runner, e.g. the phone:
    uv run python eval/llm_quality/run.py --write-cases cases.jsonl
    # 2a. Score answers recorded elsewhere: JSONL lines {"id": ..., "raw": ...}
    uv run python eval/llm_quality/run.py --answers answers.jsonl --label phone_base
    # 2b. Or generate on the laptop (functional check only, not a benchmark):
    uv run python eval/llm_quality/run.py --llm-model m.pte --tokenizer tokenizer.model --label host

`raw` is the text the model generated after the prompt (the prompt ends with {"why": ").
With --label, the raw answers are also saved next to the report so they can be rescored.

The app declines clearly off-topic questions before calling Llama (pipeline/assistant.py),
and so does this script: those cases are not sent to the model and are reported separately
(off_topic_declined_rate; false_declines counts on-topic questions the guard wrongly stopped).

Metrics over the cases sent to Llama: JSON validity (not "fallback"), clean rate ("ok": no
contradiction), useful rate (clean, has a `why`, in the requested language: the headline
number), contradiction counts by type, answer language matches the request, and mean length
of `why` in words. Contradictions include content checks (a why/tip that negates the
verdict's bin or stream, or gets "recyclable" wrong), but these are word rules: read a
sample of answers by hand before quoting the useful rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kb.build_kb import build  # noqa: E402
from models.llm.synth_data.generate import QUESTIONS, Example, generate  # noqa: E402
from pipeline.assistant import is_on_topic  # noqa: E402
from pipeline.kb import KnowledgeBase  # noqa: E402
from pipeline.prompt import full_answer  # noqa: E402
from pipeline.verify import verify  # noqa: E402

RESULTS = REPO / "eval" / "results"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def load_cases(kb: KnowledgeBase, split: str = "val") -> list[Example]:
    return [e for e in generate(kb) if e.split == split]


def language_ok(text: str | None, language: str) -> bool | None:
    if not text:
        return None
    has_devanagari = bool(DEVANAGARI.search(text))
    return has_devanagari if language == "hi" else not has_devanagari


def guarded(kb: KnowledgeBase, case: Example) -> bool:
    """True if the app would decline this case's question without asking Llama."""
    question = QUESTIONS[case.question_kind][case.language]
    return not is_on_topic(question, kb.names_of(case.item_id), kb.topic_vocabulary())


def score(
    kb: KnowledgeBase,
    cases: list[Example],
    answers: dict[str, str],
    join: Callable[[str], str] = full_answer,
) -> dict:
    """Score answers. join turns a stored answer into the full JSON text (default: the
    generated text after ANSWER_PREFIX; pass an identity function for full answers)."""
    status = Counter()
    contradictions = Counter()
    lang_checks = Counter()
    why_words = []
    useful = 0  # clean AND explains something; an empty why is clean but useless
    off_topic = [c for c in cases if c.question_kind == "off_topic"]
    off_topic_declined = sum(guarded(kb, c) for c in off_topic)
    false_declines = sum(guarded(kb, c) for c in cases if c.question_kind != "off_topic")
    for case in cases:
        if guarded(kb, case):
            continue
        raw = answers.get(case.id)
        if raw is None:
            status["missing"] += 1
            continue
        verdict = kb.resolve(case.item_id, case.attributes, case.language)
        result = verify(join(raw), verdict)
        status[result.status] += 1
        contradictions.update(result.contradictions)
        why = result.output["why"]
        ok = language_ok(why, case.language)
        if ok is not None:
            lang_checks["ok" if ok else "wrong"] += 1
        if why:
            why_words.append(len(why.split()))
            useful += result.status == "ok" and ok is not False
    answered = sum(status.values()) - status["missing"]
    return {
        "cases": len(cases),
        "off_topic_declined_rate": (
            round(off_topic_declined / len(off_topic), 3) if off_topic else None
        ),
        "false_declines": false_declines,
        "sent_to_llm": sum(status.values()),
        "answered": answered,
        "json_valid_rate": round((answered - status["fallback"]) / answered, 3)
        if answered
        else None,
        "clean_rate": round(status["ok"] / answered, 3) if answered else None,
        "useful_rate": round(useful / answered, 3) if answered else None,
        "status": dict(status),
        "contradictions": dict(contradictions),
        "language_match_rate": (
            round(lang_checks["ok"] / sum(lang_checks.values()), 3) if lang_checks else None
        ),
        "mean_why_words": round(sum(why_words) / len(why_words), 1) if why_words else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--write-cases", type=Path, help="write cases with prompts and stop")
    src.add_argument("--answers", type=Path, help='JSONL of {"id": ..., "raw": ...}')
    src.add_argument("--llm-model", type=Path, help="Llama .pte to run on the host")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--label", help="save results to eval/results/llm_quality_<label>.json")
    args = p.parse_args(argv)

    db = RESULTS.parent / "tmp" / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    cases = load_cases(kb, args.split)

    if args.write_cases:
        with args.write_cases.open("w", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps({"id": c.id, "prompt": c.prompt}, ensure_ascii=False) + "\n")
        print(f"wrote {len(cases)} cases to {args.write_cases}")
        return 0

    if args.answers:
        answers = {}
        for line in args.answers.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = row["raw"]
    else:
        if args.tokenizer is None:
            print("--llm-model needs --tokenizer", file=sys.stderr)
            return 2
        from pipeline.llm import ExecuTorchGenerator

        generator = ExecuTorchGenerator(args.llm_model, args.tokenizer)
        answers = {c.id: generator(c.prompt) for c in cases if not guarded(kb, c)}

    report = {"kb_version": kb.version, "split": args.split, **score(kb, cases, answers)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.label:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"llm_quality_{args.label}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raw_out = out.with_suffix(".answers.jsonl")
        with raw_out.open("w", encoding="utf-8") as fh:
            for c in cases:
                if c.id in answers:
                    row = {"id": c.id, "raw": answers[c.id]}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out} and {raw_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
