"""Rebuild eval/REPORT.md from every result file in eval/results/ (FuturePlan WP11).

    uv run python eval/report.py            # writes eval/REPORT.md
    uv run python eval/report.py --check    # fails if REPORT.md is out of date (CI)

One command, no hand-edited numbers: the report is only ever regenerated from the JSON the
eval scripts wrote. Each table says where its numbers were measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.latency.stages import summarise  # noqa: E402

RESULTS = REPO / "eval" / "results"
REPORT = REPO / "eval" / "REPORT.md"


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _num(v, digits=3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _load(pattern: str) -> list[tuple[str, dict]]:
    found = []
    for path in sorted(RESULTS.glob(pattern)):
        if path.name.endswith("TEMPLATE.json"):
            continue
        found.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    return found


def llm_quality_section() -> list[str]:
    rows = []
    for stem, r in _load("llm_quality_*.json"):
        label = stem.removeprefix("llm_quality_")
        rows.append([
            f"`{label}`",
            "host" if "host" in label else "phone",
            str(r.get("sent_to_llm", r.get("answered"))),
            _pct(r.get("json_valid_rate")),
            _pct(r.get("useful_rate")),
            _pct(r.get("off_topic_declined_rate")),
            str(r.get("kb_version", "–")),
        ])  # fmt: skip
    lines = ["## Llama answer quality", ""]
    if not rows:
        return lines + ["No results yet (`eval/llm_quality/run.py`).", ""]
    lines += _table(
        ["Run", "Measured on", "Sent to Llama", "Valid JSON", "Useful", "Off-topic declined", "KB"],
        rows,
    )
    return lines + [
        "",
        "Useful = valid, passes every verifier check (bins, negations, recyclable/compostable "
        "claims, KB steps, unsafe instructions) and has a `why` in the requested language. "
        "Earlier runs were scored with fewer checks; rescore saved answers with "
        "`run.py --answers` to compare like with like.",
        "",
    ]


def llm_experiments_section() -> list[str]:
    """Prompt and decoding variants (eval/llm_quality/experiments.py), best first."""
    rows = []
    for path in sorted((RESULTS / "llm_experiments" / "val").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append(r)
    lines = ["## Llama prompt experiments (88 held-out questions, laptop)", ""]
    if not rows:
        return lines + ["No results yet (`eval/llm_quality/experiments.py`).", ""]
    rows.sort(key=lambda r: (-(r["useful_rate"] or 0), r["model"], r["variant"]["name"]))
    table = [
        [r["model"], f"`{r['variant']['name']}`", _pct(r["useful_rate"]), _pct(r["useful_rate_en"]),
         _pct(r["useful_rate_hi"]), _pct(r["json_valid_rate"]), _num(r["mean_generated_tokens"], 0),
         _num(r["mean_attempts"], 2)]
        for r in rows
    ]  # fmt: skip
    lines += _table(
        ["Model", "Variant", "Useful", "Useful en", "Useful hi", "Valid JSON", "Tokens", "Tries"],
        table,
    )
    return lines + [
        "",
        "Shipped: SpinQuant, `shots_kind+rules+retry_greedy_sample` "
        "(see eval/results/llm_experiments/README.md).",
        "",
    ]


def accuracy_section() -> list[str]:
    lines = ["## Detector accuracy", ""]
    standard = _load("detector_ultralytics_*.json")
    if standard:
        lines += [
            "Standard box metrics (Ultralytics val, conf 0.001, test split):",
            "",
        ]
        lines += _table(
            ["Run", "mAP50", "mAP50-95"],
            [
                [f"`{stem.removeprefix('detector_ultralytics_')}`", _num(r["map50"]),
                 _num(r["map50_95"])]
                for stem, r in standard
            ],
        )  # fmt: skip
        lines.append("")
    rows = []
    for stem, r in _load("accuracy_*.json"):
        v = r.get("verdict") or {}
        rows.append([
            f"`{stem.removeprefix('accuracy_')}`", r.get("split", "–"), str(r["images"]),
            str(r["objects"]), _num(r["map50"]), _pct(r["item_accuracy"]),
            _pct(r["stream_accuracy"]), str(r["gate"]), _pct(v.get("coverage")),
            _pct(v.get("item_accuracy")), f"**{_pct(v.get('stream_accuracy'))}**",
        ])  # fmt: skip
    if not rows:
        return lines + ["No results yet (`eval/accuracy/report.py`).", ""]
    lines += _table(
        ["Run", "Split", "Images", "Objects", "mAP50 (app conf)", "Object item acc.",
         "Object stream acc.", "Gate", "Verdict coverage", "Verdict item acc.",
         "Verdict stream acc."],
        rows,
    )  # fmt: skip
    lines += [
        "",
        "Run through the exported `.pte` with the app's pre/post-processing (conf 0.25, so mAP "
        "here is lower than the standard one). Object columns: every ground-truth object, missed "
        "ones count as wrong. Verdict columns: what the app does with a frame (top prediction "
        'or "not sure"); coverage = share of images with a verdict. Target: 92% verdict stream '
        "accuracy.",
        "",
    ]
    for stem, r in _load("gate_sweep_*.json"):
        lines += [f"Gate sweep `{stem.removeprefix('gate_sweep_')}` ({r['split']} split):", ""]
        lines += _table(
            ["Gate", "Coverage", "Verdict item acc.", "Verdict stream acc."],
            [
                [str(x["gate"]), _pct(x["coverage"]), _pct(x["item_accuracy"]),
                 _pct(x["stream_accuracy"])]
                for x in r["rows"]
                if round(x["gate"] * 100) % 10 == 0
            ],
        )  # fmt: skip
        lines += [
            "",
            f"Lowest gate reaching {_pct(r['target'])}: {r['lowest_gate']}.",
            "",
        ]
    return lines


def fallback_section() -> list[str]:
    rows = []
    for stem, r in _load("fallback_*.json"):
        for enc in ("fp32", "int8"):
            if enc in r:
                rows.append([
                    f"`{stem.removeprefix('fallback_')}`", enc, str(r["crops"]),
                    _pct(r[enc]["top1"]), _pct(r[enc]["top3"]), _pct(r[enc]["stream_top1"]),
                ])  # fmt: skip
    lines = ["## Unknown-item fallback (SigLIP 2, zero-shot)", ""]
    if not rows:
        return lines + ["No results yet (`models/fallback/check_crops.py`).", ""]
    lines += _table(["Run", "Encoder", "Crops", "Top-1 item", "Top-3 item", "Top-1 bin"], rows)
    return lines + [
        "",
        "Labelled object crops, laptop. The fallback only offers suggestions for the user to "
        "confirm; it never gives a verdict on its own.",
        "",
    ]


def speech_section() -> list[str]:
    rows = []
    for stem, r in _load("speech_*.json"):
        for model, m in r.items():
            if isinstance(m, dict) and "median_cer" in m:
                rows.append([
                    f"`{stem.removeprefix('speech_')}`", model, _num(m["model_mb"], 0),
                    _pct(m["median_cer"]), _pct(m["name_match_top1"]),
                    _pct(m["sentence_match_top1"]), _pct(m["sentence_match_top3"]),
                ])  # fmt: skip
    lines = ["## Hindi voice input", ""]
    if not rows:
        return lines + ["No results yet (`models/speech/check_hindi.py`).", ""]
    lines += _table(
        [
            "Run",
            "Recogniser",
            "MB",
            "Median CER",
            "Name top-1",
            "Sentence top-1",
            "Sentence top-3",
        ],  # fmt: skip
        rows,
    )
    return lines + [
        "",
        "Synthetic speech from one voice on the laptop: an upper bound until native speakers "
        "are recorded. Item lookup uses `pipeline/voice.py`.",
        "",
    ]


def _median(values: list) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def latency_section() -> list[str]:
    """Phone benchmark files (schema: eval/results/llm_TEMPLATE.json)."""
    rows = []
    for stem, r in _load("llm_*.json"):
        if stem.startswith("llm_quality_") or r.get("kind") != "llm_phone_benchmark":
            continue
        runs = r.get("runs", [])
        rows.append([
            f"`{stem.removeprefix('llm_')}`", r["device"].get("model") or "–",
            r["model"].get("mode") or "–", _num(r.get("load_time_s"), 1),
            _num(_median([x.get("ttft_s") for x in runs]), 2),
            _num(_median([x.get("decode_tok_s") for x in runs]), 1),
            _num(r.get("sustained_5min_decode_tok_s"), 1), _num(r.get("peak_pss_mb"), 0),
        ])  # fmt: skip
    lines = ["## Llama speed on the phone", ""]
    if not rows:
        return lines + [
            "No phone results yet (run list: `eval/latency/llm_bench.md`; laptop speed is never "
            "reported).",
            "",
        ]
    lines += _table(
        [
            "Run",
            "Device",
            "Model",
            "Load (s)",
            "First token (s, median)",
            "Decode (tok/s, median)",
            "After 5 min (tok/s)",
            "Peak PSS (MB)",
        ],  # fmt: skip
        rows,
    )
    return lines + [""]


def stage_latency_section() -> list[str]:
    """In-app per-stage timings (schema: eval/results/latency_TEMPLATE.json)."""
    lines = ["## Per-stage latency on the phone", ""]
    rows = []
    for stem, r in _load("latency_*.json"):
        if r.get("kind") != "stage_latency":
            continue
        device = r.get("device", {}).get("model") or "–"
        for stage, s in summarise(r.get("samples", {})).items():
            rows.append([
                f"`{stem.removeprefix('latency_')}`", device, stage, str(s["n"]),
                _num(s["p50"], 1), _num(s["p95"], 1), _num(s["max"], 1),
            ])  # fmt: skip
    if not rows:
        return lines + [
            "No phone timing exports yet (the app's `LatencyLog`; summary code: "
            "`eval/latency/stages.py`).",
            "",
        ]
    header = ["Session", "Device", "Stage", "Samples", "p50 (ms)", "p95 (ms)", "Max (ms)"]
    return lines + _table(header, rows) + [""]


def build() -> str:
    lines = [
        "# Evaluation report",
        "",
        "Generated by `uv run python eval/report.py` from `eval/results/`. Do not edit by hand.",
        "",
    ]
    sections = (
        accuracy_section, fallback_section, llm_quality_section, llm_experiments_section,
        speech_section,
        latency_section, stage_latency_section,
    )  # fmt: skip
    for section in sections:
        lines += section()
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="exit 1 if REPORT.md is stale")
    args = p.parse_args(argv)
    text = build()
    if args.check:
        current = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
        if current != text:
            print("eval/REPORT.md is out of date: run uv run python eval/report.py")
            return 1
        print("eval/REPORT.md is up to date")
        return 0
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
