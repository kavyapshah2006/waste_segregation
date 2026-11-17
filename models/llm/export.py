"""Export Llama 3.2 to an ExecuTorch .pte (XNNPACK, 4-bit) with ExecuTorch's export_llm.

Usage:
    uv run python models/llm/export.py --mode int4 \\
        --checkpoint path/to/original/consolidated.00.pth --params path/to/original/params.json
    uv run python models/llm/export.py --mode spinquant --checkpoint ... --params ... --dry-run

Modes use the configs bundled with the executorch package:
    int4       llama_xnnpack.yaml, 8-bit dynamic activations and 4-bit weights (post-training).
               For the original meta-llama checkpoint or our fine-tuned, merged one (WP8).
    spinquant  llama_xnnpack_spinquant.yaml, Meta's SpinQuant INT4 EO8 checkpoint.
    qat_lora   llama_xnnpack_qat.yaml, Meta's QAT + LoRA INT4 EO8 checkpoint.

Defaults keep the context at 1024 tokens: our prompts are about 400 tokens and a shorter
KV cache saves memory on 6 GB phones.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "models" / "llm" / "out"
MODES = {
    "int4": "llama_xnnpack.yaml",
    "spinquant": "llama_xnnpack_spinquant.yaml",
    "qat_lora": "llama_xnnpack_qat.yaml",
}


def bundled_config(mode: str) -> Path:
    import executorch  # a namespace package: no __file__, use __path__

    root = Path(next(iter(executorch.__path__)))
    path = root / "examples/models/llama/config" / MODES[mode]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; is executorch installed (uv sync)?")
    return path


def _flatten(d: dict, prefix: str = "") -> set[str]:
    keys = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        keys.add(key)
        if isinstance(v, dict):
            keys |= _flatten(v, key + ".")
    return keys


def build_command(
    mode: str,
    checkpoint: Path | None,
    params: Path,
    output: Path,
    max_seq_len: int = 1024,
) -> list[str]:
    """export_llm command line. Keys already in the config file are overridden as `key=v`;
    new keys must be added as `+key=v` (Hydra rule)."""
    config = bundled_config(mode)
    existing = _flatten(yaml.safe_load(config.read_text(encoding="utf-8")) or {})
    overrides = {
        "base.model_class": "llama3_2",
        "base.params": str(params),
        "export.output_name": str(output),
        "export.max_seq_length": str(max_seq_len),
        "export.max_context_length": str(max_seq_len),
    }
    if checkpoint is not None:
        overrides["base.checkpoint"] = str(checkpoint)
    args = [f"{'' if k in existing else '+'}{k}={v}" for k, v in overrides.items()]
    return [
        sys.executable,
        "-m",
        "executorch.extension.llm.export.export_llm",
        "--config",
        str(config),
        *args,
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=sorted(MODES), default="int4")
    p.add_argument("--checkpoint", type=Path, help="consolidated.00.pth (omit only for tests)")
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--output", type=Path, help="default: models/llm/out/llama3_2_<mode>.pte")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true", help="print the command and stop")
    args = p.parse_args(argv)

    output = args.output or DEFAULT_OUT / f"llama3_2_{args.mode}.pte"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args.mode, args.checkpoint, args.params, output, args.max_seq_len)
    print(shlex.join(cmd))
    if args.dry_run:
        return 0
    if args.checkpoint is None:
        print("warning: no --checkpoint, exporting RANDOM weights (pipeline test only)")
    subprocess.run(cmd, check=True, cwd=output.parent)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
