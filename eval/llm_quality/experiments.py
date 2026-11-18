"""Compare ways of getting better answers out of Llama, all scored the same way.

Each variant changes one or more of: the system prompt, how the facts are written, worked
examples (few-shot turns), a KB-written opening for the answer, sampling, retries on a
failed check, and the model file. Every variant runs on the same on-topic cases and is scored
with eval/llm_quality/run.py's score() (format and content checks).

Laptop runs are functional checks of answer quality, never speed numbers.

Usage:
    uv run python eval/llm_quality/experiments.py --model spinquant --variants base,rules
    uv run python eval/llm_quality/experiments.py --summary   # writes the markdown table

Results go to eval/results/llm_experiments/<split>/<model>__<variant>.json plus the raw
answers (.answers.jsonl). A variant whose result file exists is skipped, so an interrupted
run can be restarted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval.llm_quality.run import guarded, language_ok, load_cases, score  # noqa: E402
from kb.build_kb import build  # noqa: E402
from models.llm.synth_data.generate import QUESTIONS, Example, completion_for  # noqa: E402
from pipeline.kb import KnowledgeBase, Verdict  # noqa: E402
from pipeline.prompt import (  # noqa: E402
    ANSWER_PREFIX,
    BOS,
    DEFAULT_QUESTION,
    EOT,
    SYSTEM_PROMPT,
    user_message,
)
from pipeline.prompt import _header as header  # noqa: E402
from pipeline.verify import verify  # noqa: E402

OUT = REPO / "eval" / "results" / "llm_experiments"
MODELS = {
    "spinquant": "models/llm/out/spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
    "qlora": "models/llm/out/qlora/Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte",
}

RULES = (
    " Rules: the bin and stream in FACTS are always right, so name only that bin. Say the "
    "item can be recycled only if FACTS has recyclable: true. Do not add disposal steps that "
    "are not in FACTS. Never suggest burning, drains, toilets, rivers or open dumping."
)

# ---------------------------------------------------------------- question kind and lead


_KINDS = (
    ("recyclable", re.compile(r"recycl|रीसा[यइ]कल", re.I)),
    ("prepare", re.compile(r"prepar|before\s+(?:throw|dispos|bin)|rinse|तैयार|पहले", re.I)),
    ("which_bin", re.compile(r"\b(?:dust)?bins?\b|डिब्ब|कूड़ेदान", re.I)),
)

_HI_COLOUR_OBLIQUE = {"green": "हरे", "blue": "नीले", "red": "लाल", "black": "काले"}


def question_kind(question: str | None) -> str:
    """What the question asks, from its words: explain, recyclable, prepare, which_bin, other."""
    if not question or not question.strip():
        return "explain"
    for kind, pattern in _KINDS:
        if pattern.search(question):
            return kind
    return "other"


def answer_lead(kind: str, verdict: Verdict, explain_too: bool = False) -> str:
    """A first sentence written from the KB, which Llama then continues. "" for no lead."""
    hi = verdict.language == "hi"
    bin_phrase = (
        f"{_HI_COLOUR_OBLIQUE.get(verdict.bin_colour, verdict.bin_colour_name)} डिब्बे"
        if hi
        else f"{verdict.bin_colour_name} bin"
    )
    if kind == "which_bin" or (kind == "explain" and explain_too):
        if hi:
            return f"यह {bin_phrase} ({verdict.stream_name}) में जाएगा। "
        return f"It goes in the {bin_phrase} ({verdict.stream_name.lower()}). "
    if kind == "recyclable":
        if hi:
            return (
                "हाँ, इसे रीसायकल किया जा सकता है। "
                if verdict.recyclable
                else "नहीं, इसे रीसायकल नहीं किया जा सकता। "
            )
        return "Yes, it can be recycled. " if verdict.recyclable else "No, it cannot be recycled. "
    if kind == "prepare" and verdict.steps:
        step = verdict.steps[0].rstrip("।.")
        return f"पहला कदम: {step}। " if hi else f"First: {step}. "
    return ""


# ---------------------------------------------------------------- prompt variants


@dataclass(frozen=True)
class Variant:
    name: str
    rules: bool = False  # stricter system prompt
    facts: str = "json"  # json | lines | minimal
    shots: str = "none"  # none | fixed (2 worked examples) | kind (2 of the same kind)
    lead: str = "none"  # none | question (KB lead for bin/recyclable/prepare) | all
    temperature: float = 0.0
    retries: int = 0  # extra attempts when the check fails
    retry_lead: bool = False  # retries use the KB lead
    retry_temperature: float | None = None  # sampling for retries only (first try greedy)
    max_new_tokens: int = 200


VARIANTS = {
    v.name: v
    for v in (
        Variant("base"),
        Variant("rules", rules=True),
        Variant("facts_lines", facts="lines"),
        Variant("facts_minimal", facts="minimal"),
        Variant("shots_fixed", shots="fixed"),
        Variant("shots_kind", shots="kind"),
        Variant("lead_question", lead="question"),
        Variant("lead_all", lead="all"),
        Variant("temp03", temperature=0.3),
        Variant("retry_sample3", temperature=0.7, retries=2),
        Variant("retry_lead", retries=1, retry_lead=True),
        Variant("retry_greedy_sample", retries=2, retry_temperature=0.7),
    )
}

# Worked examples come from train-split items only, so they never show a validation item.
SHOT_ITEMS = {
    "fixed": [("newspaper", "en", "explain"), ("food_scraps", "hi", "recyclable")],
    "kind": [("newspaper", None, None), ("aa_battery", None, None)],
}

MINIMAL_KEYS = ("item", "stream", "bin", "recyclable", "steps", "hazard")


def _facts_text(verdict: Verdict, style: str) -> str:
    facts = verdict.facts()
    if style == "minimal":
        facts = {k: facts[k] for k in MINIMAL_KEYS}
    if style in ("json", "minimal"):
        return json.dumps(facts, ensure_ascii=False, separators=(", ", ": "))
    yes_no = {True: "yes", False: "no", None: "unknown"}
    lines = [
        f"Item: {facts['item']}",
        f"Stream: {facts['stream']}",
        f"Bin: {facts['bin']}",
        f"Recyclable: {yes_no[facts['recyclable']]}",
        "Steps: " + " ".join(f"{i}) {s}" for i, s in enumerate(facts["steps"], 1)),
    ]
    for key, label in (("hazard", "Hazard"), ("drop_off", "Drop-off"), ("impact", "Impact")):
        if facts[key]:
            lines.append(f"{label}: {facts[key]}")
    if facts["notes"]:
        lines.append("Notes: " + " ".join(facts["notes"]))
    return "\n" + "\n".join(lines)


def _user_turn(v: Variant, verdict: Verdict, attributes: dict, lang_name: str, q: str | None):
    if v.facts == "json":
        return user_message(verdict, attributes, lang_name, q)
    condition = {k: val for k, val in sorted(attributes.items()) if val is not None}
    return "\n".join(
        [
            f"FACTS: {_facts_text(verdict, v.facts)}",
            f"CONDITION: {json.dumps(condition, separators=(', ', ': '))}",
            f"LANGUAGE: {lang_name}",
            f"QUESTION: {q or DEFAULT_QUESTION}",
        ]
    )


def _json_fragment(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def make_prompt(
    v: Variant, kb: KnowledgeBase, case: Example, lead: bool | None = None
) -> tuple[str, str]:
    """(prompt, prefix): the full prompt text ending in prefix, and that answer prefix."""
    verdict = kb.resolve(case.item_id, case.attributes, case.language)
    question = QUESTIONS[case.question_kind][case.language]
    kind = question_kind(question)
    system = SYSTEM_PROMPT + (RULES if v.rules else "")
    parts = [BOS, header("system"), system, EOT]
    if v.shots != "none":
        for item_id, lang, shot_kind in SHOT_ITEMS[v.shots]:
            lang = lang or case.language
            shot_kind = shot_kind or case.question_kind
            shot_verdict = kb.resolve(item_id, {}, lang)
            shot_q = QUESTIONS[shot_kind][lang]
            parts += [
                header("user"),
                _user_turn(v, shot_verdict, {}, kb.language_name(lang), shot_q),
                EOT,
                header("assistant"),
                ANSWER_PREFIX + completion_for(shot_kind, shot_verdict, kb),
            ]
    use_lead = (v.lead != "none") if lead is None else lead
    # No trailing space: after ". " the model tends to start "tip" without closing "why".
    lead_raw = answer_lead(kind, verdict, explain_too=v.lead == "all") if use_lead else ""
    lead_text = lead_raw.rstrip()
    prefix = ANSWER_PREFIX + _json_fragment(lead_text)
    parts += [
        header("user"),
        _user_turn(v, verdict, case.attributes, kb.language_name(case.language), question),
        EOT,
        header("assistant"),
        prefix,
    ]
    return "".join(parts), prefix


# ---------------------------------------------------------------- running


class Runner:
    """One loaded model; generate(prompt, temperature, max_new_tokens) -> (text, tokens)."""

    def __init__(self, model: str) -> None:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
        import executorch.kernels.quantized  # noqa: F401
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

        path = REPO / MODELS[model]
        self._config = GenerationConfig
        self._runner = TextLLMRunner(str(path), str(path.parent / "tokenizer.model"))

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> tuple[str, int]:
        config = self._config(
            echo=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_bos=0,
            num_eos=0,
        )
        pieces: list[str] = []
        stats: list = []
        self._runner.reset()
        try:
            self._runner.generate(
                prompt, config, token_callback=pieces.append, stats_callback=stats.append
            )
        except UnicodeDecodeError:
            # The Python binding passes each token as str; a token that ends inside a
            # multi-byte character (Hindi) cannot be converted and generation stops. Keep
            # the text so far: it is scored like any truncated answer, and counted.
            self.decode_errors += 1
            return "".join(pieces), len(pieces)
        return "".join(pieces), (stats[-1].num_generated_tokens if stats else 0)

    decode_errors = 0


def run_case(
    v: Variant, kb: KnowledgeBase, case: Example, gen: Callable[..., tuple[str, int]]
) -> dict:
    verdict = kb.resolve(case.item_id, case.attributes, case.language)
    attempts, tokens = 0, 0
    best = None
    for attempt in range(1 + v.retries):
        lead = True if (attempt > 0 and v.retry_lead) else None
        prompt, prefix = make_prompt(v, kb, case, lead=lead)
        if attempt == 0:
            temperature = v.temperature
        elif v.retry_temperature is not None:
            temperature = v.retry_temperature
        else:
            temperature = 0.0 if v.retry_lead else v.temperature
        raw, n = gen(prompt, temperature, v.max_new_tokens)
        attempts += 1
        tokens += n
        full = prefix + raw
        result = verify(full, verdict)
        lang = language_ok(result.output["why"], case.language)
        good = result.status == "ok" and lang is not False and bool(result.output["why"])
        if best is None or good:
            lead_text = json.loads('"' + prefix[len(ANSWER_PREFIX) :] + '"')
            best = {"full": full, "lead": lead_text, "good": good}
        if good:
            break
    return {"id": case.id, **best, "attempts": attempts, "generated_tokens": tokens}


def llm_words(row: dict, verdict: Verdict) -> int:
    """Words of `why` that Llama wrote itself, after any KB lead."""
    why = verify(row["full"], verdict).output["why"] or ""
    lead = row["lead"].strip()
    if lead and why.startswith(lead):
        why = why[len(lead) :]
    return len(why.split())


def evaluate(kb: KnowledgeBase, cases: list[Example], rows: dict[str, dict]) -> dict:
    answers = {cid: row["full"] for cid, row in rows.items()}
    sent = [c for c in cases if c.id in rows]
    report = score(kb, cases, answers, join=lambda text: text)
    for lang in ("en", "hi"):
        sub = [c for c in cases if c.language == lang]
        report[f"useful_rate_{lang}"] = score(kb, sub, answers, join=lambda t: t)["useful_rate"]
    n = len(sent)
    report["mean_attempts"] = round(sum(r["attempts"] for r in rows.values()) / n, 2)
    report["mean_generated_tokens"] = round(
        sum(r["generated_tokens"] for r in rows.values()) / n, 1
    )
    report["kb_lead_rate"] = round(sum(bool(r["lead"]) for r in rows.values()) / n, 3)
    # Useful AND Llama added at least 3 words of its own: a lead-only answer is correct but
    # is the KB talking, not Llama explaining.
    with_text = 0
    for case in sent:
        row = rows[case.id]
        verdict = kb.resolve(case.item_id, case.attributes, case.language)
        with_text += row["good"] and llm_words(row, verdict) >= 3
    report["useful_with_llm_text_rate"] = round(with_text / n, 3)
    return report


def run_variant(model: str, v: Variant, kb: KnowledgeBase, cases: list[Example], runner, out):
    started = time.time()
    errors_before = runner.decode_errors
    # Answers are saved as they come, so a stopped run resumes where it left off.
    partial = out.with_suffix(".partial.jsonl")
    rows = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
        print(f"{v.name}: resuming with {len(rows)} saved answers", flush=True)
    with partial.open("a", encoding="utf-8") as fh:
        for case in cases:
            if guarded(kb, case) or case.id in rows:
                continue
            rows[case.id] = run_case(v, kb, case, runner.generate)
            fh.write(json.dumps(rows[case.id], ensure_ascii=False) + "\n")
            fh.flush()
    rows = {c.id: rows[c.id] for c in cases if c.id in rows}  # case order, as before
    report = {
        "model": model,
        "variant": asdict(v),
        "kb_version": kb.version,
        **evaluate(kb, cases, rows),
        "decode_errors": runner.decode_errors - errors_before,
        "host_minutes": round((time.time() - started) / 60, 1),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".answers.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.unlink()
    return report


# ---------------------------------------------------------------- summary


COLUMNS = (
    ("useful_rate", "useful"),
    ("useful_with_llm_text_rate", "useful + own text"),
    ("useful_rate_en", "useful en"),
    ("useful_rate_hi", "useful hi"),
    ("json_valid_rate", "valid JSON"),
    ("mean_why_words", "why words"),
    ("mean_generated_tokens", "tokens"),
    ("mean_attempts", "attempts"),
    ("kb_lead_rate", "KB lead"),
)


def summary(split: str) -> str:
    rows = []
    for path in sorted((OUT / split).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append(r)
    rows.sort(key=lambda r: -(r["useful_rate"] or 0))
    head = "| model | variant | " + " | ".join(c[1] for c in COLUMNS) + " | top contradictions |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 3)]
    for r in rows:
        top = sorted(r["contradictions"].items(), key=lambda kv: -kv[1])[:3]
        cells = [str(r.get(key)) for key, _ in COLUMNS]
        lines.append(
            f"| {r['model']} | {r['variant']['name']} | "
            + " | ".join(cells)
            + " | "
            + ", ".join(f"{k} {n}" for k, n in top)
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--variants", help="comma-separated names from VARIANTS")
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--force", action="store_true", help="rerun variants that have results")
    p.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    p.add_argument("--summary", action="store_true", help="print the results table")
    args = p.parse_args(argv)

    if args.summary:
        print(summary(args.split))
        return 0
    if not (args.model and args.variants):
        p.error("--model and --variants are needed unless --summary")

    # One KB file per model: two runs in parallel must not rebuild the same file.
    db = REPO / "eval" / "tmp" / f"kb_{args.model}.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    cases = load_cases(kb, args.split)[: args.limit]
    # Smoke tests (--limit) go to the ignored eval/tmp, not next to the real results.
    out_dir = OUT / args.split if args.limit is None else REPO / "eval" / "tmp" / "llm_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model)
    for name in args.variants.split(","):
        v = VARIANTS[name] if name in VARIANTS else _combo(name)
        out = out_dir / f"{args.model}__{v.name}.json"
        if out.exists() and not args.force:
            print(f"skip {out.name} (exists)")
            continue
        report = run_variant(args.model, v, kb, cases, runner, out)
        keys = ("useful_rate", "useful_rate_en", "useful_rate_hi", "mean_generated_tokens")
        print(v.name, {k: report[k] for k in keys}, report["contradictions"], flush=True)
    return 0


def _combo(name: str) -> Variant:
    """'a+b+c' merges the settings of variants a, b and c (later ones win)."""
    base = VARIANTS["base"]
    merged = base
    for part in name.split("+"):
        part_v = VARIANTS[part]
        changes = {
            f: getattr(part_v, f)
            for f in base.__dataclass_fields__
            if f != "name" and getattr(part_v, f) != getattr(base, f)
        }
        merged = replace(merged, **changes)
    return replace(merged, name=name)


if __name__ == "__main__":
    sys.exit(main())
