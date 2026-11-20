"""Write every item's "Why it matters" ahead of time, for the app to ship.

On the phone, Llama needs ~30 s to load and ~15 s to read its prompt before the first word
(realme C61, Unisoc T612). Plain scans always ask the same thing for the same item, condition
and language, so this script asks the same Llama 3.2 1B SpinQuant model on the laptop, through
the same prompt builder and the same KB checks as the app (pipeline.assistant.explain: greedy
first, two sampled retries if the checks fail). The answers ship in the APK as
explanations_seed.jsonl; the app adds them to its explanation cache on first start, so a plain
scan shows its explanation at once. Typed and spoken questions still run Llama on the phone.

Jobs: every KB item x {no condition, each dirty/broken attribute true, each false} x language.
Keys match the app's ExplanationCache.key exactly. Already written keys are skipped, so the
script can be stopped and resumed.

    uv run python -m models.llm.seed_explanations                # en + hi
    uv run python -m models.llm.seed_explanations --languages en

Output: models/llm/explanations_seed.jsonl (collected into the APK by app/build.gradle.kts).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.assistant import explain  # noqa: E402
from pipeline.kb import KnowledgeBase  # noqa: E402

# Bump together with ExplanationCache.VERSION in the app when the prompt or checks change.
CACHE_VERSION = 1
SAMPLE_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200


def cache_key(kb_version: str, item_id: str, attributes: dict, language: str) -> str:
    """Same text as the app's ExplanationCache.key (booleans as true/false, sorted names)."""
    attrs = ",".join(
        f"{k}={'true' if v else 'false'}"
        for k, v in sorted(attributes.items())
        if isinstance(v, bool)
    )
    return f"v{CACHE_VERSION}|{kb_version}|{item_id}|{attrs}|{language}"


def jobs(kb: KnowledgeBase, attributes_by_item: dict, languages: list[str]):
    for language in languages:
        for item_id in kb.item_ids:
            yield item_id, {}, language
            for attr in sorted(attributes_by_item.get(item_id, {})):
                yield item_id, {attr: True}, language
                yield item_id, {attr: False}, language


class Runner:
    """One loaded model; greedy() and sampled() are pipeline.assistant generators."""

    def __init__(self, model: Path, tokenizer: Path) -> None:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  (registers kernels)
        import executorch.kernels.quantized  # noqa: F401  (registers kernels)
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

        self._config = GenerationConfig
        self._runner = TextLLMRunner(str(model), str(tokenizer))

    def _generate(self, prompt: str, temperature: float) -> str:
        config = self._config(
            echo=False,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            num_bos=0,
            num_eos=0,
        )
        pieces: list[str] = []
        self._runner.reset()
        try:
            self._runner.generate(prompt, config, token_callback=pieces.append)
        except UnicodeDecodeError:
            # Laptop-only: the Python binding passes each token as str, and a token ending
            # inside a multi-byte character (Hindi) cannot be converted, so generation stops.
            # Keep the text so far; the KB checks then accept it, retry, or drop it (as in
            # eval/llm_quality/experiments.py). The phone app receives bytes and is not affected.
            pass
        return "".join(pieces)

    def greedy(self, prompt: str) -> str:
        return self._generate(prompt, 0.0)

    def sampled(self, prompt: str) -> str:
        return self._generate(prompt, SAMPLE_TEMPERATURE)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    llm = REPO / "models/llm/out/spinquant"
    p.add_argument(
        "--model", type=Path, default=llm / "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
    )
    p.add_argument("--tokenizer", type=Path, default=llm / "tokenizer.model")
    p.add_argument("--kb", type=Path, default=REPO / "kb/build/kb.sqlite")
    p.add_argument(
        "--attributes", type=Path, default=REPO / "models/attributes/out/attribute_prompts.json"
    )
    p.add_argument("--languages", default="en,hi")
    p.add_argument("--out", type=Path, default=REPO / "models/llm/explanations_seed.jsonl")
    args = p.parse_args()

    kb = KnowledgeBase(args.kb)
    prompts = json.loads(args.attributes.read_text(encoding="utf-8"))
    attributes_by_item = prompts.get("items", {})
    todo = list(jobs(kb, attributes_by_item, args.languages.split(",")))

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["key"].split("|")[1] == kb.version:
                    done.add(row["key"])
    print(f"{len(todo)} jobs, {len(done)} already written", flush=True)

    runner = Runner(args.model, args.tokenizer)
    statuses: dict[str, int] = {}
    start = time.time()
    with args.out.open("a", encoding="utf-8") as out:
        for i, (item_id, attrs, language) in enumerate(todo, 1):
            key = cache_key(kb.version, item_id, attrs, language)
            if key in done:
                continue
            verdict = kb.resolve(item_id, attrs, language)
            result = explain(kb, verdict, attrs, runner.greedy, None, sample=runner.sampled)
            v = result.verified
            statuses[v.status] = statuses.get(v.status, 0) + 1
            if v.status in ("ok", "corrected"):
                row = {"key": key, **v.to_dict()}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            elapsed = time.time() - start
            print(f"[{i}/{len(todo)}] {key}: {v.status} ({elapsed:.0f} s)", flush=True)
    print("statuses:", statuses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
