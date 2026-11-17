"""Run the decision pipeline on the laptop.

Examples:
    uv run python -m pipeline.cli --item pet_bottle --lang hi
    uv run python -m pipeline.cli --text "pizza box" --contaminated
    uv run python -m pipeline.cli --item aa_battery --llm-output recorded.txt
    uv run python -m pipeline.cli --image photo.jpg --detector yolo26n.pte

Prints JSON: the verdict, the Llama prompt, and (with --llm-output) the verified answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.kb import DEFAULT_CITY, KnowledgeBase
from pipeline.prompt import build_prompt
from pipeline.verify import verify

REPO = Path(__file__).resolve().parents[1]
DEFAULT_KB = REPO / "kb" / "build" / "kb.sqlite"


def load_kb(path: Path) -> KnowledgeBase:
    # The default database is a build output; rebuild it every run so edits to kb/ are never
    # hidden by a stale file. An explicit --kb path is used as given.
    if path == DEFAULT_KB:
        from kb.build_kb import build

        build(out=path)
    return KnowledgeBase(path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Waste segregation pipeline (host reference).")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--item", help="KB item id, e.g. pet_bottle")
    what.add_argument("--text", help="item name or alias in any supported language")
    what.add_argument("--image", type=Path, help="photo to run the detector on")
    p.add_argument("--detector", type=Path, help="detector .pte (with --image)")
    p.add_argument("--taxonomy", type=Path, default=REPO / "data" / "taxonomy.yaml")
    p.add_argument("--contaminated", action="store_true")
    p.add_argument("--broken", action="store_true")
    p.add_argument("--lang", default="en")
    p.add_argument("--city", default=DEFAULT_CITY)
    p.add_argument("--question")
    llm = p.add_mutually_exclusive_group()
    llm.add_argument("--llm-output", help="recorded raw Llama output: a file path, or - for stdin")
    llm.add_argument(
        "--llm-model", type=Path, help="Llama .pte to run on the host (needs --tokenizer)"
    )
    p.add_argument("--tokenizer", type=Path, help="tokenizer.model for --llm-model")
    p.add_argument("--kb", type=Path, default=DEFAULT_KB)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kb = load_kb(args.kb)
    attributes = {"contaminated": args.contaminated, "broken": args.broken}
    result: dict = {"kb_version": kb.version}

    if args.item:
        item_id = args.item
    elif args.text:
        matches = kb.search(args.text)
        result["matches"] = matches
        if not matches:
            print(json.dumps({**result, "error": "no matching item"}, ensure_ascii=False, indent=2))
            return 1
        item_id = matches[0]
    else:
        if args.detector is None:
            print("--image needs --detector", file=sys.stderr)
            return 2
        from pipeline.detect import Detector

        detector = Detector(args.detector, args.taxonomy)
        detections = detector.detect_path(args.image)
        result["detections"] = [d.to_dict() for d in detections]
        known = [d for d in detections if d.item_id]
        if not known:
            print(json.dumps({**result, "error": "nothing detected"}, ensure_ascii=False, indent=2))
            return 1
        item_id = known[0].item_id

    verdict = kb.resolve(item_id, attributes, args.lang, args.city)
    result["verdict"] = verdict.to_dict()
    result["prompt"] = build_prompt(verdict, attributes, kb.language_name(args.lang), args.question)
    if args.llm_model:
        if args.tokenizer is None:
            print("--llm-model needs --tokenizer", file=sys.stderr)
            return 2
        from pipeline.assistant import explain
        from pipeline.llm import ExecuTorchGenerator

        generator = ExecuTorchGenerator(args.llm_model, args.tokenizer)
        explanation = explain(kb, verdict, attributes, generator, args.question)
        result["raw_outputs"] = explanation.raw_outputs
        result["verified"] = explanation.verified.to_dict()
    elif args.llm_output:
        raw = (
            sys.stdin.read()
            if args.llm_output == "-"
            else Path(args.llm_output).read_text(encoding="utf-8")
        )
        result["verified"] = verify(raw, verdict).to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
