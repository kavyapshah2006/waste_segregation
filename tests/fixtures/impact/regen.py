"""Recompute `expected` in tests/fixtures/impact/cases.json from the Python reference.

    uv run python tests/fixtures/impact/regen.py

As with the golden fixtures: only after an intended change, then read the diff by hand.
The Kotlin app must reproduce the same values (android/core ImpactTest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from kb.build_kb import build  # noqa: E402
from pipeline import education  # noqa: E402
from pipeline.impact import Scan, summary  # noqa: E402
from pipeline.kb import KnowledgeBase  # noqa: E402


def expected_for(kb: KnowledgeBase, case: dict):
    kind, inp = case["kind"], case["input"]
    if kind == "summary":
        scans = [Scan(s["time_s"], s["item_id"], s.get("attributes", {})) for s in inp["scans"]]
        return summary(kb, scans, inp["now_s"], inp.get("tz_offset_minutes", 330))
    if kind == "did_you_know":
        return education.did_you_know(kb, inp["day"], inp["language"])
    if kind == "quiz":
        return education.quiz(kb, inp["day"], inp["language"])
    if kind == "watch_out":
        return education.watch_out_cards(kb, inp["language"])
    raise ValueError(kind)


def main() -> None:
    path = HERE / "cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    out = HERE / "_kb.sqlite"
    build(out=out)
    try:
        kb = KnowledgeBase(out)
        for case in cases:
            case["expected"] = expected_for(kb, case)
    finally:
        out.unlink(missing_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {len(cases)} cases")


if __name__ == "__main__":
    main()
