"""LoRA fine-tune of Llama 3.2 Instruct on the KB-written answers (WP8).

Data: models/llm/synth_data/out/{train,val}.jsonl from synth_data/generate.py. Each example
is the exact prompt the app sends plus the completion we want (the rest of the
{"why", "tip"} answer after the prefilled `{"why": "`).

Tokenization uses the same C++ tiktoken tokenizer as ExecuTorch's runner (pytorch_tokenizers
CppTiktoken), with the prompt and the completion encoded separately, so training sees exactly
the token ids the phone will see. Loss is on the completion tokens only.

    uv run --group finetune python models/llm/lora/train.py \\
        --base path/to/Llama-3.2-1B-Instruct --tokenizer path/to/original/tokenizer.model \\
        --out models/llm/lora/out/run1

Writes <out>/adapter (PEFT LoRA adapter) and <out>/train_log.json. Then run convert.py to
merge and convert, models/llm/export.py --mode int4 to export, and
eval/llm_quality/run.py to score. Runs on CPU for the dry run; use a GPU for the real one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "models" / "llm" / "synth_data" / "out"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PAD_ID = 128004  # <|finetune_right_pad_id|>; padded positions are masked anyway
IGNORE = -100


def load_tokenizer(path: Path | str):
    from pytorch_tokenizers import CppTiktoken

    tok = CppTiktoken()
    tok.load(str(path))
    return tok


def encode(tok, text: str) -> list[int]:
    return list(tok.encode(text, 0, 0))  # no BOS/EOS: the text already carries them


def build_example(tok, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """Token ids and labels; labels are IGNORE on the prompt so only the answer is learned."""
    p, c = encode(tok, prompt), encode(tok, completion)
    return p + c, [IGNORE] * len(p) + c


def load_split(tok, data_dir: Path, split: str, max_len: int, limit: int | None = None):
    rows = [
        json.loads(line)
        for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]
    examples, skipped = [], 0
    for row in rows:
        ids, labels = build_example(tok, row["prompt"], row["completion"])
        if len(ids) > max_len:
            skipped += 1
            continue
        examples.append((ids, labels))
    return examples, skipped


def batches(examples, batch_size: int, rng: random.Random | None):
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start : start + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), PAD_ID)
        labels = torch.full((len(chunk), width), IGNORE)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (x, y) in enumerate(chunk):
            ids[row, : len(x)] = torch.tensor(x)
            labels[row, : len(y)] = torch.tensor(y)
            mask[row, : len(x)] = 1
        yield ids, labels, mask


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_loss(model, examples, batch_size: int, device: str, amp) -> float:
    model.eval()
    total, count = 0.0, 0
    for ids, labels, mask in batches(examples, batch_size, None):
        with amp():
            out = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)
            )
        n = int((labels[:, 1:] != IGNORE).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(1, count)


def data_hash(data_dir: Path) -> str:
    h = hashlib.sha256()
    for split in ("train", "val"):
        h.update((data_dir / f"{split}.jsonl").read_bytes())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="HF model dir or repo id")
    p.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.model (tiktoken)")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=1024, help="matches the exported context")
    p.add_argument("--limit", type=int, help="use only the first N examples per split (tests)")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = None
    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def amp():
        if amp_dtype is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=amp_dtype)

    tok = load_tokenizer(args.tokenizer)
    train, skipped_train = load_split(tok, args.data, "train", args.max_len, args.limit)
    val, skipped_val = load_split(tok, args.data, "val", args.max_len, args.limit)
    lengths = sorted(len(ids) for ids, _ in train)
    print(
        f"train {len(train)} (skipped {skipped_train} over {args.max_len} tokens), "
        f"val {len(val)} (skipped {skipped_val}); tokens median {lengths[len(lengths) // 2]}, "
        f"max {lengths[-1]}"
    )

    # Weights stay fp32; autocast does the low-precision maths on GPU (works on T4 and newer).
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=TARGETS,
            task_type="CAUSAL_LM",
        ),  # fmt: skip
    )
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = max(1, int(args.epochs * steps_per_epoch))
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, total // 30)
    opt = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    log = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "data_hash": data_hash(args.data),
        "device": device,
        "amp": str(amp_dtype),
        "steps": total,
        "train_loss": [],
        "val_loss": [],
    }
    log["val_loss"].append({"step": 0, "loss": eval_loss(model, val, args.batch, device, amp)})
    print(f"step 0: val loss {log['val_loss'][-1]['loss']:.4f}")

    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for ids, labels, mask in batches(train, args.batch, rng):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, args.lr, warmup)
            with amp():
                loss = model(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    labels=labels.to(device),
                ).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            log["train_loss"].append({"step": step, "loss": float(loss)})
            if step % 10 == 0 or step == total:
                print(f"step {step}/{total}: loss {float(loss):.4f} ({time.time() - t0:.0f}s)")
            if step % steps_per_epoch == 0 or step == total:
                log["val_loss"].append(
                    {"step": step, "loss": eval_loss(model, val, args.batch, device, amp)}
                )
                print(f"step {step}: val loss {log['val_loss'][-1]['loss']:.4f}")
            if step >= total:
                break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter")
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'adapter'} and train_log.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
