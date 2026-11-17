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


def accuracy_section() -> list[str]:
    rows = []
    for stem, r in _load("accuracy_*.json"):
        rows.append([
            f"`{stem.removeprefix('accuracy_')}`", r.get("split", "–"), str(r["images"]),
            str(r["objects"]), _num(r["map50"]), _pct(r["item_accuracy"]),
            f"**{_pct(r['stream_accuracy'])}**", str(r["gate"]),
        ])  # fmt: skip
    lines = ["## Detector accuracy", ""]
    if not rows:
        return lines + ["No results yet (`eval/accuracy/report.py`).", ""]
    lines += _table(
        ["Run", "Split", "Images", "Objects", "mAP50", "Item acc.", "Stream acc.", "Gate"], rows
    )
    return lines + [
        "",
        "Stream accuracy (target 92%): share of ground-truth objects whose prediction lands in "
        "the right bin; missed objects count as wrong.",
        "",
    ]


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


def build() -> str:
    lines = [
        "# Evaluation report",
        "",
        "Generated by `uv run python eval/report.py` from `eval/results/`. Do not edit by hand.",
        "",
    ]
    sections = (
        accuracy_section, fallback_section, llm_quality_section, speech_section, latency_section
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
