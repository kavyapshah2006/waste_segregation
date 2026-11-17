"""Verifier, prompt and assistant behaviour. Per-function cases live in
tests/fixtures/check_cases.json (tests/test_check_cases.py), shared with the Kotlin core."""

import json

from pipeline.assistant import explain
from pipeline.prompt import ANSWER_PREFIX, EOT, build_prompt, full_answer, prompt_prefix
from pipeline.verify import verify


def test_a_content_error_in_one_field_drops_both(kb):
    verdict = kb.resolve("fruit_veg_peels")
    raw = json.dumps(
        {"why": "Do not remove stickers from fruit.", "tip": "Leave the stickers on the fruit."}
    )
    result = verify(raw, verdict)
    assert result.status == "corrected"
    assert result.contradictions == ("why_negates_step",)
    assert result.output["why"] is None and result.output["tip"] is None
    assert result.output["steps"] == list(verdict.steps)


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
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n" + ANSWER_PREFIX)
    assert prompt.count(EOT) == 2
    assert "LANGUAGE: Hindi" in prompt
    assert not build_prompt(verdict, {}, "Hindi", include_bos=False).startswith("<|begin")


def test_system_prefix_is_the_same_for_every_item_and_language(kb):
    a = build_prompt(kb.resolve("pet_bottle"), {}, "English")
    b = build_prompt(kb.resolve("syringe", language="hi"), {"broken": True}, "Hindi")
    prefix = prompt_prefix()
    assert a.startswith(prefix) and b.startswith(prefix)


def test_full_answer_joins_the_prefix_and_the_generated_text(kb):
    generated = 'Batteries leak metals.", "tip": "Tape the ends."}' + EOT
    result = verify(full_answer(generated), kb.resolve("aa_battery"))
    assert result.status == "ok"
    assert (result.output["why"], result.output["tip"]) == (
        "Batteries leak metals.",
        "Tape the ends.",
    )
    assert result.output["bin"] == "black"


def test_off_topic_question_is_declined_without_calling_llama(kb):
    verdict = kb.resolve("pet_bottle", language="hi")

    def never(prompt):
        raise AssertionError("Llama must not be called")

    result = explain(kb, verdict, {}, never, question="आज मौसम कैसा रहेगा?")
    assert result.attempts == 0
    assert result.verified.status == "declined"
    assert result.verified.output["why"] == kb.message("off_topic", "hi")
    assert result.verified.output["bin"] == "blue" and result.verified.output["steps"]


def test_explain_retries_once_with_a_short_prompt(kb):
    verdict = kb.resolve("aa_battery")
    outputs = iter(["cut off mid-sent", 'Batteries leak metals.", "tip": "Tape the ends."}'])
    result = explain(kb, verdict, {}, lambda prompt: next(outputs))
    assert result.attempts == 2
    assert "CONDITION" in result.prompts[0] and "CONDITION" not in result.prompts[1]
    assert result.raw_outputs[1].startswith("Batteries")
    assert result.verified.status == "ok"


def test_explain_falls_back_after_two_failures(kb):
    verdict = kb.resolve("aa_battery")
    result = explain(kb, verdict, {}, lambda prompt: "still not json")
    assert result.attempts == 2
    assert result.verified.status == "fallback"
    assert result.verified.output["steps"] == list(verdict.steps)


def test_explain_stops_after_first_good_answer(kb):
    verdict = kb.resolve("aa_battery")
    result = explain(kb, verdict, {}, lambda prompt: 'Use the green bin.", "tip": "x"}')
    assert result.attempts == 1
    assert result.verified.status == "corrected"
    assert result.verified.contradictions == ("why_bin",)
