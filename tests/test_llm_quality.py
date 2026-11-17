import json

import pytest

from eval.llm_quality.run import language_ok, load_cases, score
from models.llm.synth_data.generate import generate
from pipeline.prompt import EOT, full_answer


@pytest.fixture(scope="module")
def examples(kb):
    return generate(kb)


def test_every_target_answer_passes_the_verifier_and_has_a_split(examples):
    # generate() raises if any completion fails verification; check coverage here.
    assert len(examples) >= 400
    assert {e.language for e in examples} == {"en", "hi"}
    assert {e.split for e in examples} == {"train", "val"}
    assert len({e.id for e in examples}) == len(examples)


def test_validation_items_never_appear_in_training(examples):
    train = {e.item_id for e in examples if e.split == "train"}
    val = {e.item_id for e in examples if e.split == "val"}
    assert val and not train & val


def test_completions_are_why_and_tip_only_and_use_kb_facts(kb, examples):
    e = next(x for x in examples if x.id == "aa_battery__plain__en__which_bin")
    assert e.completion.endswith(EOT)
    answer = json.loads(full_answer(e.completion).removesuffix(EOT))
    assert set(answer) == {"why", "tip"}
    assert answer["why"].startswith("It goes in the black bin")
    assert answer["tip"] == kb.resolve("aa_battery").hazard


def test_perfect_answers_score_clean(kb):
    cases = load_cases(kb)
    answers = {c.id: c.completion for c in cases}
    report = score(kb, cases, answers)
    assert report["answered"] == len(cases)
    assert report["clean_rate"] == 1.0 and report["json_valid_rate"] == 1.0
    assert report["useful_rate"] == 1.0
    assert report["language_match_rate"] == 1.0


def test_empty_answers_are_clean_but_not_useful(kb):
    cases = load_cases(kb)[:5]
    report = score(kb, cases, {c.id: '"}' for c in cases})  # {"why": ""}
    assert report["clean_rate"] == 1.0
    assert report["useful_rate"] == 0.0


def test_bad_answers_are_counted(kb):
    cases = load_cases(kb)[:4]
    wrong_bin = "red" if kb.resolve(cases[1].item_id).bin_colour != "red" else "green"
    answers = {
        cases[0].id: "cut off mid-sent",
        cases[1].id: f'Put it in the {wrong_bin} bin.", "tip": "x"}}',
        cases[2].id: cases[2].completion,
    }  # cases[3] has no answer
    report = score(kb, cases, answers)
    assert report["status"] == {"fallback": 1, "corrected": 1, "ok": 1, "missing": 1}
    assert report["json_valid_rate"] == round(2 / 3, 3)
    assert report["contradictions"] == {"unparseable": 1, "why_bin": 1}


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("यह सही है", "hi", True),
        ("This is right", "hi", False),
        ("Fine", "en", True),
        (None, "en", None),
    ],
)
def test_language_ok(text, lang, expected):
    assert language_ok(text, lang) is expected
