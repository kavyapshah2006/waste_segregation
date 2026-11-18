"""The "This is wrong" log (FuturePlan WP7 step 6, checklist step 37).

When the user corrects a verdict, the app appends one JSON line to a file on the phone and keeps
the photo it scanned. Nothing leaves the phone on its own: the user can export the file (and the
photos) to help relabel and retrain. This module is the reference format plus a summary for the
laptop; Kotlin `Corrections` writes exactly the same lines (shared cases:
tests/fixtures/corrections_cases.json).

One line:
    {"time_ms": 1795000000000, "source": "detector", "predicted": "pet_bottle",
     "corrected": "glass_bottle", "confidence": 0.62, "attributes": {"broken": false,
     "contaminated": false}, "image": "corrections/1795000000000.jpg"}

`predicted` is null when nothing was recognised; `confidence` is null for voice and manual
picks; `image` is null when no photo was kept (voice).

Usage:
    uv run python -m pipeline.corrections corrections.jsonl   # summary of an exported file
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SOURCES = ("detector", "fallback", "voice", "photo")
ATTRIBUTES = ("broken", "contaminated")


@dataclass(frozen=True)
class Correction:
    time_ms: int
    source: str
    predicted: str | None
    corrected: str
    confidence: float | None = None
    attributes: dict[str, bool] = field(default_factory=dict)
    image: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if not self.corrected:
            raise ValueError("corrected item is empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        unknown = set(self.attributes) - set(ATTRIBUTES)
        if unknown:
            raise ValueError(f"unknown attributes {sorted(unknown)}")


def to_line(c: Correction) -> str:
    attributes = {k: c.attributes[k] for k in ATTRIBUTES if k in c.attributes}
    row = {
        "time_ms": c.time_ms,
        "source": c.source,
        "predicted": c.predicted,
        "corrected": c.corrected,
        "confidence": c.confidence,
        "attributes": attributes,
        "image": c.image,
    }
    return json.dumps(row, ensure_ascii=False, separators=(", ", ": "))


def parse_line(line: str) -> Correction:
    row = json.loads(line)
    return Correction(
        time_ms=int(row["time_ms"]),
        source=row["source"],
        predicted=row.get("predicted"),
        corrected=row["corrected"],
        confidence=row.get("confidence"),
        attributes=dict(row.get("attributes") or {}),
        image=row.get("image"),
    )


def summarise(corrections: list[Correction], known_items: set[str] | None = None) -> dict:
    """Counts for relabelling: which predictions users fixed, and into what."""
    pairs = Counter(f"{c.predicted or '(none)'} -> {c.corrected}" for c in corrections)
    unknown = sorted({c.corrected for c in corrections} - known_items) if known_items else []
    return {
        "corrections": len(corrections),
        "by_source": dict(sorted(Counter(c.source for c in corrections).items())),
        "pairs": dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))),
        "with_image": sum(c.image is not None for c in corrections),
        "unknown_items": unknown,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("file", type=Path, help="exported corrections .jsonl")
    p.add_argument("--kb", type=Path, help="kb.sqlite, to flag item ids the KB does not know")
    args = p.parse_args(argv)
    lines = args.file.read_text(encoding="utf-8").splitlines()
    corrections = [parse_line(line) for line in lines if line.strip()]
    known = None
    if args.kb:
        from pipeline.kb import KnowledgeBase

        known = set(KnowledgeBase(args.kb).item_ids)
    print(json.dumps(summarise(corrections, known), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
