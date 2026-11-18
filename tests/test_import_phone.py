"""eval/latency/import_phone.py on a synthetic phone session."""

import json

from eval.latency.import_phone import llm_runs, main


def test_llm_runs_from_stats():
    stats = json.dumps({
        "prompt_tokens": 300, "generated_tokens": 51, "model_load_start_ms": 0,
        "model_load_end_ms": 4000, "inference_start_ms": 10_000, "prompt_eval_end_ms": 11_000,
        "first_token_ms": 11_000, "inference_end_ms": 13_500,
    })  # fmt: skip
    (run,) = llm_runs([stats])
    assert run["ttft_s"] == 1.0
    assert run["prefill_tok_s"] == 300.0
    assert run["decode_tok_s"] == 20.0  # 50 tokens after the first, in 2.5 s
    assert run["load_s"] == 4.0


def test_import_session(tmp_path):
    src, out = tmp_path / "phone", tmp_path / "results"
    src.mkdir()
    out.mkdir()
    (src / "latency_2026-11-19_1000.json").write_text(json.dumps({
        "kind": "stage_latency", "date": "2026-11-19", "device": {"model": "Acme Phone 5"},
        "models": {"llm": "Llama.pte"}, "conditions": {"airplane_mode": True},
        "samples": {"detect": [10.0, 12.0]},
    }))  # fmt: skip
    stats = {"prompt_tokens": 10, "generated_tokens": 11, "inference_start_ms": 0,
             "first_token_ms": 500, "inference_end_ms": 1500}  # fmt: skip
    (src / "llm_stats_2026-11-19_1000.jsonl").write_text(json.dumps(stats) + "\n")
    (src / "meminfo_x.txt").write_text("  TOTAL PSS:   2097152 TOTAL RSS: 1\n")
    (src / "items_2026-11-19.csv").write_text(
        "item_id,placement,app_item_id,app_bin,correct_item,correct_bin,notes\n"
        "pet_bottle,table,pet_bottle,blue,y,y,\n"
        "glass_bottle,hand,pet_bottle,blue,n,y,\n"
    )
    assert main([str(src), "--out", str(out)]) == 0
    latency = json.loads((out / "latency_acme_phone_5_2026-11-19.json").read_text())
    assert latency["samples"]["detect"] == [10.0, 12.0]
    llm = json.loads((out / "llm_acme_phone_5_2026-11-19.json").read_text())
    assert llm["kind"] == "llm_phone_benchmark"
    assert llm["runs"][0]["decode_tok_s"] == 10.0
    assert llm["peak_pss_mb"] == 2048.0
    items = json.loads((out / "phone_items_acme_phone_5_2026-11-19.json").read_text())
    assert items["item_accuracy"] == 0.5
    assert items["bin_accuracy"] == 1.0
