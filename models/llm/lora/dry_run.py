"""End-to-end dry run of the fine-tuning chain on a tiny random-weight Llama, on the CPU.

    uv run --group finetune python models/llm/lora/dry_run.py

Builds a 2-layer Llama with the real Llama 3.2 vocabulary and RoPE settings, then runs every
step of the real pipeline on it: train.py (LoRA, a few steps on a few examples) ->
convert.py --check (merge, Meta layout, logits match ExecuTorch's own model) ->
export.py --mode int4 (.pte) -> load the .pte with the ExecuTorch runner -> score with
eval/llm_quality. The answers are nonsense (random weights); the point is that nothing in
the chain breaks before a GPU hour is spent on the real 1B model. About 3 minutes.

Needs a tokenizer.model; by default the one downloaded with the SpinQuant model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "models" / "llm" / "lora" / "out" / "dry_run"
TOKENIZER = REPO / "models" / "llm" / "out" / "spinquant" / "tokenizer.model"


def tiny_base(path: Path) -> None:
    """A random 2-layer Llama shaped like Llama 3.2 (vocab, GQA, scaled RoPE, tied output)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128256, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072,
        rms_norm_eps=1e-5, rope_theta=500000.0, tie_word_embeddings=True,
        rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        bos_token_id=128000, eos_token_id=[128001, 128009],
    )  # fmt: skip
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)


def step(name: str):
    print(f"\n=== {name} ===", flush=True)
    return time.time()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args(argv)
    if not args.tokenizer.exists():
        print(f"{args.tokenizer} missing: run scripts/download_models.sh llama-spinquant first")
        return 2

    from eval.llm_quality.run import load_cases, score
    from kb.build_kb import build
    from models.llm import export
    from models.llm.lora import convert, train
    from models.llm.synth_data import generate
    from pipeline.kb import KnowledgeBase

    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    summary: dict = {}

    t = step("1. synthetic data")
    generate.main(["--out", str(args.out / "data")])
    summary["data_s"] = round(time.time() - t, 1)

    t = step("2. tiny random base model")
    tiny_base(args.out / "base")

    t = step("3. LoRA training (CPU)")
    rc = train.main([
        "--base", str(args.out / "base"), "--tokenizer", str(args.tokenizer),
        "--data", str(args.out / "data"), "--out", str(args.out / "run"),
        "--limit", "8", "--batch", "4", "--lr", "5e-3", "--max-steps", str(args.steps),
    ])  # fmt: skip
    log = json.loads((args.out / "run" / "train_log.json").read_text())
    first, last = log["train_loss"][0]["loss"], log["train_loss"][-1]["loss"]
    summary.update(train_rc=rc, train_loss_first=round(first, 3), train_loss_last=round(last, 3),
                   train_s=round(time.time() - t, 1))  # fmt: skip
    if rc or not last < first:
        print(f"FAILED: training did not reduce the loss ({first:.3f} -> {last:.3f})")
        return 1

    t = step("4. merge + convert + logits check")
    rc = convert.main([
        "--base", str(args.out / "base"), "--adapter", str(args.out / "run" / "adapter"),
        "--out", str(args.out / "meta"), "--dtype", "fp32", "--check",
    ])  # fmt: skip
    summary.update(convert_rc=rc, convert_s=round(time.time() - t, 1))
    if rc:
        return 1

    t = step("5. export to .pte (int4, XNNPACK)")
    pte = args.out / "tiny_lora_int4.pte"
    rc = export.main([
        "--mode", "int4", "--checkpoint", str(args.out / "meta" / "consolidated.00.pth"),
        "--params", str(args.out / "meta" / "params.json"), "--output", str(pte),
    ])  # fmt: skip
    summary.update(
        export_rc=rc, export_s=round(time.time() - t, 1), pte_mb=round(pte.stat().st_size / 1e6, 1)
    )
    if rc:
        return 1

    t = step("6. run the .pte and score 4 held-out cases")
    from pipeline.llm import ExecuTorchGenerator

    db = args.out / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    cases = [c for c in load_cases(kb) if c.question_kind != "off_topic"][:4]
    gen = ExecuTorchGenerator(pte, args.tokenizer, max_new_tokens=24)
    answers = {c.id: gen(c.prompt) for c in cases}
    report = score(kb, cases, answers)
    summary.update(eval_s=round(time.time() - t, 1), eval_answered=report["answered"],
                   sample_answer=next(iter(answers.values()))[:80])  # fmt: skip
    if report["answered"] != len(cases):
        print("FAILED: the runner did not answer every case")
        return 1

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("\nDRY RUN OK\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
