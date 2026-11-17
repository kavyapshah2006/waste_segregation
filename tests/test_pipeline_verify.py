import json

import pytest

from pipeline.assistant import explain, is_on_topic
from pipeline.prompt import ANSWER_PREFIX, EOT, build_prompt, full_answer, prompt_prefix
from pipeline.verify import (
    bins_mentioned,
    compostable_claim,
    extract_json,
    negates_step,
    negates_verdict,
    recyclable_claim,
    unsafe_instruction,
    verify,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('noise {"a": "}"} more', {"a": "}"}),
        ('{"a": "say \\"hi\\" {x}"}', {"a": 'say "hi" {x}'}),
        ('[1, 2] then {"b": 2}', {"b": 2}),
        ('{broken} {"c": 3}', {"c": 3}),
        ('{"a": 1', {"a": 1}),  # missing final brace is repaired
        ('{"a": "x", "b": "y"<|eot_id|>', {"a": "x", "b": "y"}),
        ('{"a": "cut off mid-sen', None),  # never repaired: would show a cut-off sentence
        ('{"a": {"b": 1}', {"a": {"b": 1}}),
        ('{"a": 1,', None),
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Do not put plastic bags in the blue bin.", True),
        ("Never mix it with dry waste.", True),
        ("इसे सूखे कचरे में न डालें।", True),
        ("थैली को नीले डिब्बे में मत डालें", True),
        ("Put it in the blue bin.", False),
        ("Do not put it in the green bin.", False),  # a different bin
        ("It is dry waste, not wet waste.", False),
        ("Do not burn it; put it in the blue bin.", False),
    ],
)
def test_negates_verdict(kb, text, expected):
    verdict = kb.resolve("plastic_carry_bag")  # dry, blue
    assert negates_verdict(text, verdict) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It is not recyclable.", False),
        ("Chips packets are non-recyclable.", False),
        ("नहीं, यह रीसायकल नहीं हो सकता है।", False),
        ("Aluminium can be recycled endlessly.", True),
        ("Glass can be recycled forever and quality does not drop.", True),
        ("एल्युमिनियम को बार-बार रीसायकल किया जा सकता है और उसकी गुणवत्ता कम नहीं होती।", True),
        ("Thermocol is rarely recycled.", None),
        ("थर्मोकोल शायद ही रीसायकल होता है।", None),
        ("It is hard to recycle.", None),
        ("Keep it dry.", None),
    ],
)
def test_recyclable_claim(text, expected):
    assert recyclable_claim(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Peels cannot be composted.", False),
        ("Peels can be composted at home.", True),
        ("तेल वाला काग़ज़ खाद बन सकता है।", True),
        ("डायपर घर पर न खाद बन सकते हैं।", None),
    ],
)
def test_compostable_claim(text, expected):
    assert compostable_claim(text) is expected


def test_negates_step_only_for_positive_kb_steps(kb):
    peels = kb.resolve("fruit_veg_peels")
    assert negates_step("Do not remove stickers from fruit.", peels)
    assert not negates_step("Remove stickers from fruit.", peels)
    cfl = kb.resolve("cfl_bulb", {"broken": True})  # step: "Do not vacuum or sweep; ..."
    assert not negates_step("Do not vacuum or sweep.", cfl)
    glass = kb.resolve("broken_glass", language="hi")
    assert not negates_step("मोटे काग़ज़ या डिब्बे में लपेटें ताकि यह न किसी चोट का कारण बने।", glass)


@pytest.mark.parametrize(
    ("item", "text", "expected"),
    [
        ("plastic_carry_bag", "थैलियों को नाली में डाल दें।", True),
        ("plastic_carry_bag", "Just throw them in the drain.", True),
        ("plastic_carry_bag", "You can burn them.", True),
        ("plastic_carry_bag", "Plastic bags clog drains and harm animals.", False),
        ("plastic_carry_bag", "Never burn plastic.", False),
        ("paint_can", "नाली में डाले गए पेंट और रसायन नदियों तक पहुँच जाते हैं।", False),
        ("diaper", "Empty solid waste into the toilet.", False),  # the KB's own step
    ],
)
def test_unsafe_instruction(kb, item, text, expected):
    assert unsafe_instruction(text, kb.resolve(item)) is expected


def test_content_checks_drop_only_the_bad_field(kb):
    verdict = kb.resolve("plastic_carry_bag")
    raw = json.dumps({"why": "It is not recyclable.", "tip": "Carry a cloth bag."})
    result = verify(raw, verdict)
    assert result.contradictions == ("why_recyclable",)
    assert result.output["why"] is None and result.output["tip"] == "Carry a cloth bag."


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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (None, True),
        ("Which bin does this go in?", True),
        ("Is it recyclable?", True),
        ("Can I throw this in the kitchen bin?", True),
        ("What about the PET plastic bottle cap?", True),  # names the item
        ("यह किस डिब्बे में जाएगा?", True),
        ("इसे फेंकने से पहले कैसे तैयार करूँ?", True),
        ("What is the capital of France?", False),
        ("Tell me a joke", False),
        ("आज मौसम कैसा रहेगा?", False),
    ],
)
def test_is_on_topic(kb, question, expected):
    assert is_on_topic(question, kb.names_of("pet_bottle")) is expected


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
