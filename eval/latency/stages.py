"""Per-stage latency summaries for phone timing exports (FuturePlan WP11, step 60).

The app records how long each stage of a scan takes (Kotlin `LatencyLog` in android/core) and
exports one JSON file per session, saved as eval/results/latency_<device>_<date>.json:

    {"kind": "stage_latency", "device": {...}, "commit": "...", "conditions": {...},
     "samples": {"detect": [12.1, 11.8, ...], "frame_to_card": [...], ...}}   # milliseconds

Both sides summarise with the same nearest-rank percentile, so the number on the phone's debug
overlay and the one in eval/REPORT.md agree (shared cases: tests/fixtures/latency_cases.json).
"""

from __future__ import annotations

import math

# Stage names, in pipeline order. The Kotlin `Stages` object uses the same strings.
STAGES = (
    "capture",
    "preprocess",
    "detect",
    "attributes",
    "vote",
    "kb",
    "frame_to_card",
    "fallback",
    "llm_first_token",
    "asr",
    "tts_start",
)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least q of the samples at or below."""
    if not values:
        return None
    if not 0 < q <= 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summarise(samples: dict[str, list[float]]) -> dict[str, dict]:
    """{stage: {"n", "p50", "p95", "max"}} for every stage with at least one sample."""
    out = {}
    known = [s for s in STAGES if s in samples]
    extra = sorted(s for s in samples if s not in STAGES)
    for stage in known + extra:
        values = samples[stage]
        if not values:
            continue
        if any(v < 0 for v in values):
            raise ValueError(f"negative duration in {stage}")
        out[stage] = {
            "n": len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    return out
