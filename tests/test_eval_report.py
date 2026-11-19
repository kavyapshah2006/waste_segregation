import json

from eval import report


def test_report_builds_from_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    (tmp_path / "accuracy_demo.json").write_text(json.dumps({
        "split": "test", "images": 5, "objects": 4, "map50": 0.2778,
        "item_accuracy": 0.25, "stream_accuracy": 0.5, "gate": 0.5,
        "verdict": {"gate": 0.5, "coverage": 0.6, "item_accuracy": 0.3333,
                    "stream_accuracy": 0.6667},
    }))  # fmt: skip
    (tmp_path / "detector_ultralytics_demo.json").write_text(
        json.dumps({"map50": 0.305, "map50_95": 0.237})
    )
    (tmp_path / "gate_sweep_demo.json").write_text(json.dumps({
        "split": "val", "target": 0.92, "lowest_gate": 0.5,
        "rows": [{"gate": 0.5, "coverage": 0.6, "item_accuracy": 0.6, "stream_accuracy": 0.93},
                 {"gate": 0.55, "coverage": 0.5, "item_accuracy": 0.6, "stream_accuracy": 0.94}],
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
    group = {"images": 10, "app_stream_acc": 0.9, "detector_first_stream_acc": 0.6,
             "app_item_acc": 0.7, "fallback_top3_item": 0.95, "app_verdict_coverage": 0.2,
             "app_verdict_stream_acc": None}  # fmt: skip
    (tmp_path / "bdwaste_demo.json").write_text(
        json.dumps({"gate": 0.5, "overall": group, "by_stream": {"wet": group}})
    )
    text = report.build()
    assert "| all | 10 | **90.0%** | 60.0% | 70.0% | 95.0% | 20.0% | – |" in text
    assert "| wet | 10 |" in text
    assert (
        "| `demo` | test | 5 | 4 | 0.278 | 25.0% | 50.0% | 0.5 | 60.0% | 33.3% | **66.7%** |"
        in text
    )
    assert "| `demo` | 0.305 | 0.237 |" in text
    assert "| 0.5 | 60.0% | 60.0% | 93.0% |" in text  # only whole tenths are listed
    assert "| 0.55 |" not in text and "Lowest gate reaching 92.0%: 0.5." in text
    assert "| `x_host` | host | 64 | 95.3% | 60.9% | 100.0% | abc |" in text
    assert "| `pixel_2026-12-01` | Pixel | spinquant | 2.5 | 0.50 | 29.0 | 25.0 | 1900 |" in text


def test_empty_results_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    text = report.build()
    assert "No results yet (`eval/accuracy/report.py`)" in text
    assert "No results yet (`eval/bdwaste/evaluate.py`)" in text
    assert "No phone results yet" in text


def test_committed_report_is_up_to_date():
    assert report.main(["--check"]) == 0
