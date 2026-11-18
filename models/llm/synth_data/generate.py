"""Build supervised fine-tuning data for Llama from the knowledge base (WP8).

Every example is (prompt, completion): the prompt is exactly what the app sends
(pipeline/prompt.py, ending with the answer prefix {"why": ") and the completion is the
rest of the {"why", "tip"} answer we want back, written from KB facts. Each completion must
pass pipeline/verify.py with status "ok", so the model is never trained on an answer the
app would reject.

Usage:
    uv run python models/llm/synth_data/generate.py            # writes train/val JSONL
    uv run python models/llm/synth_data/generate.py --stats    # counts only

Output (gitignored): models/llm/synth_data/out/{train,val}.jsonl. Validation items are held
out by item id, so validation measures answers about items the model never trained on.

Template text is plain and repetitive by design; Hindi templates are unreviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from kb.build_kb import build  # noqa: E402
from pipeline.kb import KnowledgeBase, Verdict  # noqa: E402
from pipeline.prompt import ANSWER_PREFIX, EOT, build_prompt, full_answer  # noqa: E402
from pipeline.shots import QUESTIONS, answer_for  # noqa: E402
from pipeline.verify import verify  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
LANGS = ("en", "hi")
VAL_SHARE = 0.15


@dataclass(frozen=True)
class Example:
    id: str
    item_id: str
    language: str
    question_kind: str
    attributes: dict
    prompt: str
    completion: str
    split: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _split(item_id: str) -> str:
    u = int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big") / 2**64
    return "val" if u < VAL_SHARE else "train"


def completion_for(kind: str, v: Verdict, kb: KnowledgeBase) -> str:
    """The text the model should generate after the prompt (which ends with ANSWER_PREFIX)."""
    answer = answer_for(kind, v, kb)
    assert answer.startswith(ANSWER_PREFIX)
    return answer[len(ANSWER_PREFIX) :] + EOT


def condition_variants(item: dict) -> list[dict]:
    variants = [{"contaminated": False, "broken": False}]
    for rule in item.get("condition_rules", []):
        variants.append({"contaminated": False, "broken": False, **rule["if"]})
    return variants


def generate(kb: KnowledgeBase) -> list[Example]:
    examples: list[Example] = []
    for item_id in kb.item_ids:
        for attributes in condition_variants(kb.item(item_id)):
            for lang in LANGS:
                verdict = kb.resolve(item_id, attributes, lang)
                for kind, questions in QUESTIONS.items():
                    prompt = build_prompt(
                        verdict, attributes, kb.language_name(lang), questions[lang], kb=kb
                    )
                    completion = completion_for(kind, verdict, kb)
                    check = verify(full_answer(completion), verdict)
                    if check.status != "ok":
                        raise AssertionError(f"{item_id}/{lang}/{kind}: {check.contradictions}")
                    cond = "_".join(k for k, val in sorted(attributes.items()) if val) or "plain"
                    examples.append(
                        Example(
                            id=f"{item_id}__{cond}__{lang}__{kind}",
                            item_id=item_id,
                            language=lang,
                            question_kind=kind,
                            attributes=attributes,
                            prompt=prompt,
                            completion=completion,
                            split=_split(item_id),
                        )
                    )
    return examples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--stats", action="store_true")
    args = p.parse_args(argv)

    db = args.out / "kb.sqlite"
    args.out.mkdir(parents=True, exist_ok=True)
    build(out=db)
    examples = generate(KnowledgeBase(db))
    counts = Counter((e.split, e.language) for e in examples)
    print(json.dumps({f"{s}/{lang}": n for (s, lang), n in sorted(counts.items())}, indent=2))
    val_items = sorted({e.item_id for e in examples if e.split == "val"})
    print(f"held-out items: {', '.join(val_items)}")
    if not args.stats:
        for split in ("train", "val"):
            path = args.out / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for e in examples:
                    if e.split == split:
                        fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
