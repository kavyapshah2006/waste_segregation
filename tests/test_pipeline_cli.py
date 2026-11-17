import json

from pipeline.cli import main


def run(capsys, *args):
    code = main(list(args))
    return code, json.loads(capsys.readouterr().out)


def test_item_in_hindi_without_any_model(capsys, kb_path):
    code, out = run(capsys, "--item", "pet_bottle", "--lang", "hi", "--kb", str(kb_path))
    assert code == 0
    assert out["verdict"]["bin_colour_name"] == "नीला"
    assert "LANGUAGE: Hindi" in out["prompt"]


def test_text_lookup_with_condition(capsys, kb_path):
    code, out = run(capsys, "--text", "pizza box", "--contaminated", "--kb", str(kb_path))
    assert code == 0
    assert out["matches"][0] == "cardboard_box"
    assert out["verdict"]["stream"] == "wet"


def test_unknown_text(capsys, kb_path):
    code, out = run(capsys, "--text", "spaceship", "--kb", str(kb_path))
    assert code == 1 and out["error"] == "no matching item"


def test_recorded_llm_output_is_verified(capsys, kb_path, tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text('{"stream": "dry", "bin": "blue", "why": "metal"}', encoding="utf-8")
    code, out = run(capsys, "--item", "aa_battery", "--llm-output", str(raw), "--kb", str(kb_path))
    assert code == 0
    assert out["verified"]["status"] == "corrected"
    assert out["verified"]["output"]["bin"] == "black"
