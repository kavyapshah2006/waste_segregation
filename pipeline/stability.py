"""Confidence gate and voting over camera frames (FuturePlan WP7). Reference for Kotlin.

A single frame is noisy: motion blur, a hand in the way, the item half out of view. The app
shows a verdict only once the same item has won several recent frames with enough
confidence; until then it says it is unsure (and offers "show the label" / "type it").

Per frame, the vote is the most confident detection that maps to a KB item, if its
confidence reaches the gate; otherwise the frame votes for nothing. The state is:
- "stable": one item has at least `min_votes` of the last `window` votes;
- "unsure": something was detected in the window but no item has enough votes;
- "empty": nothing detected in the window at all.

The gate and window are starting values; tune the gate on the validation set with
eval/accuracy/report.py (higher gate: fewer wrong verdicts, more "unsure").
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

GATE = 0.5
WINDOW = 5
MIN_VOTES = 3


@dataclass(frozen=True)
class FrameVote:
    item_id: str | None  # None: no detection at or above the gate
    confidence: float  # of the best item detection in the frame, even if below the gate
    detected: bool  # the frame had any item detection


def frame_vote(detections: list[dict], gate: float = GATE) -> FrameVote:
    """detections: [{"item_id": ..., "confidence": ...}, ...] as from Detection.to_dict()."""
    items = [d for d in detections if d.get("item_id")]
    if not items:
        return FrameVote(None, 0.0, False)
    best = max(items, key=lambda d: (d["confidence"], d["item_id"]))
    item = best["item_id"] if best["confidence"] >= gate else None
    return FrameVote(item, float(best["confidence"]), True)


class FrameVoter:
    def __init__(self, window: int = WINDOW, min_votes: int = MIN_VOTES, gate: float = GATE):
        if not 1 <= min_votes <= window:
            raise ValueError("need 1 <= min_votes <= window")
        self.window, self.min_votes, self.gate = window, min_votes, gate
        self._votes: deque[FrameVote] = deque(maxlen=window)

    def reset(self) -> None:
        self._votes.clear()

    def push(self, detections: list[dict]) -> dict:
        self._votes.append(frame_vote(detections, self.gate))
        return self.state()

    def state(self) -> dict:
        counts = Counter(v.item_id for v in self._votes if v.item_id is not None)
        if counts:
            mean_conf = {
                item: sum(v.confidence for v in self._votes if v.item_id == item) / n
                for item, n in counts.items()
            }
            item = min(counts, key=lambda i: (-counts[i], -mean_conf[i], i))
            if counts[item] >= self.min_votes:
                return {
                    "state": "stable",
                    "item_id": item,
                    "votes": counts[item],
                    "confidence": round(mean_conf[item], 4),
                }
        if any(v.detected for v in self._votes):
            return {"state": "unsure", "item_id": None, "votes": 0, "confidence": None}
        return {"state": "empty", "item_id": None, "votes": 0, "confidence": None}
