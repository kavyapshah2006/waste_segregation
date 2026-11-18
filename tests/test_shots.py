"""pipeline/shots.py. tests/fixtures/question_kinds.json is shared with the Kotlin core
(ShotsTest); the exact prompt text with examples is pinned by the golden fixtures."""

import json
from pathlib import Path

import pytest

from pipeline.prompt import ANSWER_PREFIX, build_prompt
from pipeline.shots import SHOT_ITEMS, question_kind, shot_kind, shot_turns
from pipeline.verify import verify

CASES = json.loads((Path(__file__).parent / "fixtures" / "question_kinds.json").read_text("utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[str(c["question"]) for c in CASES])
def test_question_kind(case):
    assert question_kind(case["question"]) == case["kind"]
    assert shot_kind(case["question"]) == case["shots"]


@pytest.mark.parametrize("language", ["en", "hi"])
@pytest.mark.parametrize("question", [None, "Is this recyclable?", "Which bin?", "Do I rinse it?"])
def test_example_answers_pass_the_verifier(kb, language, question):
    # An example that the verifier would reject would teach the model the wrong thing.
    for (user, answer), item_id in zip(shot_turns(kb, question, language), SHOT_ITEMS, strict=True):
        verdict = kb.resolve(item_id, {}, language)
        assert verify(answer, verdict).status == "ok", (item_id, answer)
        assert "QUESTION:" in user


def test_examples_only_in_english_and_hindi(kb):
    assert shot_turns(kb, None, "ta") == []
    plain = build_prompt(kb.resolve("pet_bottle"), {}, "English")
    with_examples = build_prompt(kb.resolve("pet_bottle"), {}, "English", kb=kb)
    assert with_examples.endswith(ANSWER_PREFIX)
    assert with_examples.count("<|start_header_id|>user") == 3
    assert plain.count("<|start_header_id|>user") == 1
    # Short mode (the retry after unparseable output) stays without examples.
    short = build_prompt(kb.resolve("pet_bottle"), {}, "English", short=True, kb=kb)
    assert short.count("<|start_header_id|>user") == 1
