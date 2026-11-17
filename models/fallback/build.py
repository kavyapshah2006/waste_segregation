"""Build the unknown-item fallback: SigLIP 2 image encoder (.pte) + KB text embeddings (WP7).

    uv run --group finetune python models/fallback/build.py

When the detector stays unsure, the app runs this image encoder on the frame (or the
detector's best box), compares the embedding with one text embedding per KB item, and offers
the top 3 items for the user to confirm. It never gives a verdict on its own.

Model: google/siglip2-base-patch16-224 (Apache-2.0, not gated), pinned to a revision.
Apple's MobileCLIP was considered and rejected: its weights are licensed for non-commercial
research only.

Outputs in models/fallback/out/:
- siglip2_image_fp32.pte, siglip2_image_int8.pte: image -> L2-normalised 768-d embedding.
  Input float32 [1, 3, 224, 224]: RGB resized to 224x224 (bilinear, no aspect padding),
  scaled to [0, 1], then (x - 0.5) / 0.5.
- text_embeddings.json: model, revision, logit scale/bias, item ids, one L2-normalised
  embedding per item (mean over "a photo of <name>" for the item's English names and aliases).
- build_report.json: how closely each .pte reproduces the PyTorch embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
OUT = REPO / "models" / "fallback" / "out"
PROMPT = "a photo of {}"


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        return torch.nn.functional.normalize(pooled, dim=-1)


def item_texts(kb, item_id: str) -> list[str]:
    item = kb.item(item_id)
    names = [item["names"]["en"], *item.get("aliases", {}).get("en", [])]
    return [PROMPT.format(n) for n in dict.fromkeys(names)]


@torch.no_grad()
def text_embeddings(model, tokenizer, kb) -> tuple[list[str], np.ndarray]:
    ids, rows = kb.item_ids, []
    for item_id in ids:
        batch = tokenizer(item_texts(kb, item_id), padding="max_length", max_length=64,
                          truncation=True, return_tensors="pt")  # fmt: skip
        out = model.get_text_features(**batch)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = torch.nn.functional.normalize(emb, dim=-1).mean(0)
        rows.append(torch.nn.functional.normalize(emb, dim=0).numpy())
    return ids, np.stack(rows)


def export(encoder: torch.nn.Module, out: Path, quantize: bool) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    example = (torch.randn(1, 3, 224, 224),)
    program = torch.export.export(encoder, example)
    if quantize:
        # A few quantize/dequantize ops stay outside the XNNPACK delegate; their out-variant
        # kernels live in this library, which must be loaded before to_executorch().
        import executorch.kernels.quantized  # noqa: F401
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )
        prepared = prepare_pt2e(program.module(), quantizer)
        prepared(*example)  # dynamic activations: one pass only records shapes
        program = torch.export.export(convert_pt2e(prepared), example)
    edge = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()])
    out.write_bytes(edge.to_executorch().buffer)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    from kb.build_kb import build
    from pipeline.fallback import Fallback, preprocess
    from pipeline.kb import KnowledgeBase

    args.out.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(MODEL, revision=REVISION, local_dir=args.out / "hf",
                              allow_patterns=["*.json", "model.safetensors"])  # fmt: skip
    model = AutoModel.from_pretrained(local, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(local)

    db = args.out / "kb.sqlite"
    build(out=db)
    kb = KnowledgeBase(db)
    ids, emb = text_embeddings(model, tokenizer, kb)
    text_json = args.out / "text_embeddings.json"
    text_json.write_text(json.dumps({
        "model": MODEL, "revision": REVISION, "prompt": PROMPT, "kb_version": kb.version,
        "logit_scale": float(model.logit_scale.exp().detach()),
        "logit_bias": float(model.logit_bias.detach()),
        "item_ids": ids, "embeddings": [[round(float(x), 6) for x in row] for row in emb],
    }) + "\n", encoding="utf-8")  # fmt: skip
    print(f"wrote {text_json} ({len(ids)} items)")

    encoder = ImageEncoder(model).eval()
    report = {}
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(3)]
    for name, quantize in (("fp32", False), ("int8", True)):
        pte = args.out / f"siglip2_image_{name}.pte"
        export(encoder, pte, quantize)
        fb = Fallback(pte, text_json)
        cos = []
        for arr in samples:
            from PIL import Image

            x = preprocess(Image.fromarray(arr))
            with torch.no_grad():
                ref = encoder(torch.from_numpy(x)).numpy()[0]
            cos.append(float(np.dot(ref, fb.embed(Image.fromarray(arr)))))
        report[name] = {
            "mb": round(pte.stat().st_size / 1e6, 1),
            "min_cosine_vs_pytorch": round(min(cos), 5),
        }
        print(name, report[name])
    (args.out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
