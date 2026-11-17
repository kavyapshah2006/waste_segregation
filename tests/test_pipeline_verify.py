import json

import pytest

from pipeline.assistant import explain
from pipeline.prompt import EOT, build_prompt, prompt_prefix
from pipeline.verify import bins_mentioned, extract_json, verify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('noise {"a": "}"} more', {"a": "}"}),
        ('{"a": "say \\"hi\\" {x}"}', {"a": 'say "hi" {x}'}),
        ('[1, 2] then {"b": 2}', {"b": 2}),
        ('{broken} {"c": 3}', {"c": 3}),
        ('{"a": 1', None),
        ("no json here", None),
        ("", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_bins_mentioned_english_and_hindi():
    assert bins_mentioned("Put it in the Blue bin, not the green dustbin.") == {"blue", "green"}
    assert bins_mentioned("इसे नीले डिब्बे में डालें") == {"blue"}
    assert bins_mentioned("Keep it dry and blue skies") == set()


def test_free_text_naming_the_right_bin_is_kept(kb):
    verdict = kb.resolve("pet_bottle")
    raw = json.dumps({"stream": "dry", "bin": "blue", "why": "The blue bin is right."})
    result = verify(raw, verdict)
    assert result.status == "ok"
    assert result.output["why"] == "The blue bin is right."


def test_bins_named_by_the_kb_entry_are_allowed_when_llama_agrees(kb):
    verdict = kb.resolve("cardboard_box", {"contaminated": True})  # green, steps mention blue
    raw = json.dumps(
        {
            "stream": "wet",
            "bin": "green",
            "steps": ["Put clean parts in the blue bin"],
            "why": "Soiled parts go in the green bin, clean parts in the blue bin.",
        }
    )
    result = verify(raw, verdict)
    assert result.status == "ok"
    assert result.output["why"].startswith("Soiled parts")


def test_after_a_wrong_bin_only_the_verdict_bin_may_be_named(kb):
    verdict = kb.resolve("cardboard_box", {"contaminated": True})
    raw = json.dumps({"stream": "dry", "bin": "blue", "why": "Clean parts: blue bin."})
    result = verify(raw, verdict)
    assert result.contradictions == ("stream", "bin", "why_bin")
    assert result.output["why"] is None


def test_long_free_text_is_truncated(kb):
    verdict = kb.resolve("pet_bottle")
    result = verify(json.dumps({"why": "word " * 200}), verdict)
    assert len(result.output["why"]) <= 401
    assert result.output["why"].endswith("…")


def test_non_string_free_text_is_dropped(kb):
    result = verify(json.dumps({"why": ["a"], "tip": 5}), kb.resolve("pet_bottle"))
    assert result.output["why"] is None and result.output["tip"] is None


def test_prompt_shape(kb):
    verdict = kb.resolve("pet_bottle", language="hi")
    prompt = build_prompt(verdict, {}, "Hindi")
    assert prompt.startswith(prompt_prefix())
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert prompt.count(EOT) == 2
    assert "LANGUAGE: Hindi" in prompt
    assert not build_prompt(verdict, {}, "Hindi", include_bos=False).startswith("<|begin")


def test_system_prefix_is_the_same_for_every_item_and_language(kb):
    a = build_prompt(kb.resolve("pet_bottle"), {}, "English")
    b = build_prompt(kb.resolve("syringe", language="hi"), {"broken": True}, "Hindi")
    prefix = prompt_prefix()
    assert a.startswith(prefix) and b.startswith(prefix)


def test_explain_retries_once_with_a_short_prompt(kb):
    verdict = kb.resolve("aa_battery")
    outputs = iter(["not json", json.dumps({"stream": "special_care", "bin": "black"})])
    result = explain(kb, verdict, {}, lambda prompt: next(outputs))
    assert result.attempts == 2
    assert "CONDITION" in result.prompts[0] and "CONDITION" not in result.prompts[1]
    assert result.verified.status == "ok"


def test_explain_falls_back_after_two_failures(kb):
    verdict = kb.resolve("aa_battery")
    result = explain(kb, verdict, {}, lambda prompt: "still not json")
    assert result.attempts == 2
    assert result.verified.status == "fallback"
    assert result.verified.output["steps"] == list(verdict.steps)


def test_explain_stops_after_first_good_answer(kb):
    verdict = kb.resolve("aa_battery")
    result = explain(kb, verdict, {}, lambda prompt: '{"stream": "dry"}')
    assert result.attempts == 1
    assert result.verified.status == "corrected"
