"""Zip what the Colab notebook needs: train.py and freshly generated training data.

    uv run python models/llm/lora/make_bundle.py      # -> models/llm/lora/out/lora_bundle.zip

Upload the zip in models/llm/lora/train.ipynb. The data is regenerated from the current KB,
so retrain after KB changes.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "models" / "llm" / "lora" / "out"


def main() -> int:
    from models.llm.synth_data import generate

    data = OUT / "bundle_data"
    generate.main(["--out", str(data)])
    bundle = OUT / "lora_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).with_name("train.py"), "train.py")
        for split in ("train", "val"):
            z.write(data / f"{split}.jsonl", f"data/{split}.jsonl")
    print(f"wrote {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
