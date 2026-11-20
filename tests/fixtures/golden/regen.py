"""Recompute `expected` in every golden fixture from the current Python pipeline.

Usage: uv run python tests/fixtures/golden/regen.py

Only run this after an intended change to the KB, prompt or verifier, then review the git
diff of every fixture by hand before committing. The Android app must reproduce the same
expected values, so a careless regen hides real regressions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
REPO = GOLDEN_DIR.parents[2]
sys.path.insert(0, str(REPO))

from kb.build_kb import build  # noqa: E402
from pipeline.assistant import declined, is_on_topic  # noqa: E402
from pipeline.kb import KnowledgeBase  # noqa: E402
from pipeline.prompt import build_prompt, full_answer  # noqa: E402
from pipeline.verify import verify  # noqa: E402


def expected_for(kb: KnowledgeBase, case_input: dict) -> dict:
    detection = case_input["detection"]
    if detection.get("item_id") is None:
        return {"verdict": None, "prompt": None, "verified": None}
    language = case_input.get("language", "en")
    attributes = detection.get("attributes", {})
    verdict = kb.resolve(
        detection["item_id"], attributes, language, case_input.get("city", "default_in")
    )
    question = case_input.get("question")
    if not is_on_topic(question, kb.names_of(verdict.item_id), kb.topic_vocabulary()):
        # Off-topic question: Llama is never asked, so there is no prompt.
        return {
            "verdict": verdict.to_dict(),
            "prompt": None,
            "verified": declined(kb, verdict).to_dict(),
        }
    prompt = build_prompt(verdict, attributes, kb.language_name(language), question, kb=kb)
    raw = case_input.get("raw_llm_output")
    return {
        "verdict": verdict.to_dict(),
        "prompt": prompt,
        "verified": None if raw is None else verify(full_answer(raw), verdict).to_dict(),
    }


def main() -> None:
    out = GOLDEN_DIR / "_kb.sqlite"
    build(out=out)
    kb = KnowledgeBase(out)
    try:
        for path in sorted(GOLDEN_DIR.glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            case["expected"] = expected_for(kb, case["input"])
            path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"updated {path.name}")
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
