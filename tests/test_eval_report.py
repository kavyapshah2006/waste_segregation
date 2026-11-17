import json

from eval import report


def test_report_builds_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    (tmp_path / "accuracy_demo.json").write_text(json.dumps({
        "split": "test", "images": 5, "objects": 4, "map50": 0.2778,
        "item_accuracy": 0.25, "stream_accuracy": 0.5, "gate": 0.5,
    }))  # fmt: skip
    (tmp_path / "llm_quality_x_host.json").write_text(json.dumps({
        "sent_to_llm": 64, "json_valid_rate": 0.953, "useful_rate": 0.609,
        "off_topic_declined_rate": 1.0, "kb_version": "abc",
    }))  # fmt: skip
    (tmp_path / "llm_TEMPLATE.json").write_text("{}")
    runs = [{"ttft_s": 0.4, "decode_tok_s": 30}, {"ttft_s": 0.6, "decode_tok_s": 28}]
    (tmp_path / "llm_pixel_2026-12-01.json").write_text(json.dumps({
        "kind": "llm_phone_benchmark", "device": {"model": "Pixel"}, "model": {"mode": "spinquant"},
        "load_time_s": 2.5, "runs": runs, "sustained_5min_decode_tok_s": 25, "peak_pss_mb": 1900,
    }))  # fmt: skip
    text = report.build()
    assert "| `demo` | test | 5 | 4 | 0.278 | 25.0% | **50.0%** | 0.5 |" in text
    assert "| `x_host` | host | 64 | 95.3% | 60.9% | 100.0% | abc |" in text
    assert "| `pixel_2026-12-01` | Pixel | spinquant | 2.5 | 0.50 | 29.0 | 25.0 | 1900 |" in text


def test_empty_results_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    text = report.build()
    assert "No results yet (`eval/accuracy/report.py`)" in text
    assert "No phone results yet" in text


def test_committed_report_is_up_to_date():
    assert report.main(["--check"]) == 0
