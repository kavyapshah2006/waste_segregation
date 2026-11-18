"""Turn the raw files from a phone session into result files the report reads.

    uv run python eval/latency/import_phone.py eval/results/phone/

Reads what scripts/phone_collect.sh and the app's "Export timings" wrote:
- latency_<stamp>.json (app, kind "stage_latency") -> eval/results/latency_<device>_<date>.json
- llm_stats_<stamp>.jsonl (app, one ExecuTorch stats JSON per Llama answer)
  + meminfo_*.txt (TOTAL PSS) -> eval/results/llm_<device>_<date>.json (llm_TEMPLATE.json schema)
- items_*.csv (filled by hand, template eval/results/phone_items_TEMPLATE.csv)
  -> eval/results/phone_items_<device>_<date>.json (item and bin accuracy on real objects)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "eval" / "results"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "phone"


def commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def llm_runs(stats_lines: list[str]) -> list[dict]:
    runs = []
    for line in stats_lines:
        if not line.strip():
            continue
        s = json.loads(line)
        start, first = s["inference_start_ms"], s["first_token_ms"]
        prefill_end, end = s.get("prompt_eval_end_ms", first), s["inference_end_ms"]
        prompt, generated = s.get("prompt_tokens", 0), s.get("generated_tokens", 0)
        runs.append({
            "start_ms": start,
            "prompt_tokens": prompt,
            "generated_tokens": generated,
            "ttft_s": round((first - start) / 1000, 3),
            "prefill_tok_s": round(prompt / ((prefill_end - start) / 1000), 1)
            if prefill_end > start else None,
            "decode_tok_s": round((generated - 1) / ((end - first) / 1000), 1)
            if end > first and generated > 1 else None,
            "load_s": round((s["model_load_end_ms"] - s["model_load_start_ms"]) / 1000, 2)
            if s.get("model_load_end_ms") else None,
        })  # fmt: skip
    return runs


def peak_pss_mb(folder: Path) -> float | None:
    values = []
    for path in folder.glob("meminfo_*.txt"):
        m = re.search(r"TOTAL PSS:\s+(\d+)", path.read_text(encoding="utf-8", errors="replace"))
        if m:
            values.append(int(m.group(1)) / 1024)
    return round(max(values), 1) if values else None


def items_summary(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("item_id") and r.get("app_item_id") is not None]
    n = len(rows)
    item_ok = sum(r["app_item_id"].strip() == r["item_id"].strip() for r in rows)
    bin_ok = sum(
        (r.get("correct_bin") or "").strip().lower() in ("y", "yes", "1", "true") for r in rows
    )
    return {
        "kind": "phone_items",
        "attempts": n,
        "item_accuracy": round(item_ok / n, 3) if n else None,
        "bin_accuracy": round(bin_ok / n, 3) if n else None,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    written = []
    device_slug, date, meta = "phone", "", {}
    for path in sorted(args.folder.glob("latency_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data
        device_slug = slug(data.get("device", {}).get("model", "phone"))
        date = data.get("date", "")
        data.setdefault("commit", commit())
        out = args.out / f"latency_{device_slug}_{date or path.stem[8:]}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(out)

    stats = [ln for path in sorted(args.folder.glob("llm_stats_*.jsonl"))
             for ln in path.read_text(encoding="utf-8").splitlines()]  # fmt: skip
    runs = llm_runs(stats)
    if runs:
        t0 = runs[0]["start_ms"]
        later = [
            r["decode_tok_s"]
            for r in runs
            if r["start_ms"] - t0 >= 5 * 60 * 1000 and r["decode_tok_s"]
        ]
        report = {
            "kind": "llm_phone_benchmark",
            "date": date,
            "commit": commit(),
            "device": meta.get("device", {}),
            "model": {"file": meta.get("models", {}).get("llm"), "mode": "int4"},
            "runtime": {"app": "org.wastesegregation.app", "executorch_android": "1.5.1"},
            "conditions": meta.get("conditions", {}),
            "load_time_s": next((r["load_s"] for r in runs if r["load_s"]), None),
            "runs": [
                {
                    k: r[k]
                    for k in (
                        "prompt_tokens",
                        "generated_tokens",
                        "ttft_s",
                        "prefill_tok_s",
                        "decode_tok_s",
                    )
                }
                for r in runs
            ],  # fmt: skip
            "sustained_5min_decode_tok_s": round(sum(later) / len(later), 1) if later else None,
            "peak_pss_mb": peak_pss_mb(args.folder),
        }
        out = args.out / f"llm_{device_slug}_{date or 'session'}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        written.append(out)

    rows = []
    for path in sorted(args.folder.glob("items_*.csv")):
        with path.open(encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    if rows:
        out = args.out / f"phone_items_{device_slug}_{date or 'session'}.json"
        out.write_text(json.dumps(items_summary(rows), ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")  # fmt: skip
        written.append(out)

    for path in written:
        print(f"wrote {path}")
    if not written:
        print("nothing found to import")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
