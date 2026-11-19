"""Make the app's launcher icon and in-app logo from one source image (the SwachhSathi logo).

    uv run python android/app/make_icons.py android/app/logo/swachhsathi_logo.png

The source may have a dark margin around the rounded-square logo (as exported from a design
tool): the logo is found as the bounding box of non-dark pixels. Writes:
  res/drawable-nodpi/app_logo.png         512 px, rounded corners transparent (in-app logo)
  res/drawable-nodpi/ic_launcher_art.png  432 px adaptive-icon layer (art in the visible 72 dp zone)
  res/mipmap-anydpi-v26/ic_launcher{,_round}.xml  adaptive icon using it as the background
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

RES = Path(__file__).parent / "src" / "main" / "res"
ADAPTIVE = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/ic_launcher_art" />
    <foreground android:drawable="@android:color/transparent" />
</adaptive-icon>
"""


def logo_box(img: Image.Image, dark: int = 40) -> tuple[int, int, int, int]:
    """Bounding box of pixels brighter than `dark` (the logo, without its dark margin)."""
    grey = img.convert("L").point(lambda v: 255 if v > dark else 0)
    box = grey.getbbox()
    if box is None:
        raise SystemExit("no logo found: the image is all dark")
    return box


def main(src: str) -> int:
    img = Image.open(src).convert("RGBA")
    left, top, right, bottom = logo_box(img)
    side = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = side / 2
    square = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))

    # In-app logo: rounded square, transparent outside the corners.
    logo = square.resize((512, 512), Image.LANCZOS)
    mask = Image.new("L", logo.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 511, 511), radius=112, fill=255)
    logo.putalpha(mask)

    # Launcher art: inset past the rounded corners so the launcher's own mask shapes it. Adaptive
    # icons only show the middle 72 of 108 dp, so the art goes in that zone and a blurred copy
    # fills the outer ring (seen only in launcher animations).
    inset = int(side * 0.07)
    inner = square.crop((inset, inset, side - inset, side - inset)).convert("RGB")
    art = inner.resize((432, 432), Image.LANCZOS).filter(ImageFilter.GaussianBlur(18))
    visible = round(432 * 76 / 108)
    art.paste(inner.resize((visible, visible), Image.LANCZOS), ((432 - visible) // 2,) * 2)

    out = RES / "drawable-nodpi"
    out.mkdir(parents=True, exist_ok=True)
    logo.save(out / "app_logo.png", optimize=True)
    art.convert("RGB").save(out / "ic_launcher_art.png", optimize=True)
    anydpi = RES / "mipmap-anydpi-v26"
    anydpi.mkdir(parents=True, exist_ok=True)
    for name in ("ic_launcher.xml", "ic_launcher_round.xml"):
        (anydpi / name).write_text(ADAPTIVE, encoding="utf-8")
    print(f"logo box {left, top, right, bottom} -> {out}/app_logo.png, ic_launcher_art.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
