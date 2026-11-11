#!/usr/bin/env python3
"""Draw the title and chapter cards as transparent PNGs for finish.sh.

cards.py WIDTH HEIGHT OUTDIR  -> OUTDIR/card_NN.png + OUTDIR/cards.txt
(cards.txt: one "file start duration fade-out-at" line per card, seconds)
"""
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import os

HERE = os.path.dirname(os.path.abspath(__file__))
LATO = os.path.join(HERE, "assets", "fonts")  # Lato 2.015, SIL OFL 1.1
BIG_FACE = f"{LATO}/Lato-Light.ttf"
SMALL_FACE = f"{LATO}/Lato-Regular.ttf"
LOGO = os.path.join(HERE, "assets", "qtpass-logo.png")
INK = (243, 230, 211, 255)

sys.path.insert(0, HERE)
import timeline  # noqa: E402

# (start, end, layout, big line, small line), timed to the film's clock
CARDS = timeline.cards()


def tracked(draw, xy, text, font, tracking):
    """Draw text with extra letter spacing; return its width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=INK)
        x += draw.textlength(ch, font=font) + tracking
    return x - xy[0] - tracking


def width_of(draw, text, font, tracking):
    return sum(draw.textlength(c, font=font) for c in text) + tracking * (len(text) - 1)


def card(w, h, layout, big, small):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    centre = layout == "centre"
    bsize = int(h * (0.12 if centre else 0.075))
    ssize = int(h * (0.036 if centre else 0.032))
    bf = ImageFont.truetype(BIG_FACE, bsize)
    sf = ImageFont.truetype(SMALL_FACE, ssize)
    btrack = bsize * (0.03 if centre else 0.02)
    strack = ssize * 0.1
    if centre:
        lh = int(h * 0.2)
        logo = Image.open(LOGO).convert("RGBA").resize((lh, lh), Image.LANCZOS)
        img.alpha_composite(logo, ((w - lh) // 2, int(h * 0.15)))
        by = int(h * 0.37)
        bx = (w - width_of(d, big, bf, btrack)) / 2
        sx = (w - width_of(d, small, sf, strack)) / 2
        sy = by + int(bsize * 1.35)
    else:
        bx, by = int(w * 0.06), int(h * 0.1)
        sx, sy = bx + int(bsize * 0.04), by + int(bsize * 1.25)
    tracked(d, (bx, by), big, bf, btrack)
    tracked(d, (sx, sy), small, sf, strack)
    # soft shadow under the type
    alpha = img.getchannel("A").filter(ImageFilter.GaussianBlur(h * 0.006))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda v: int(v * 0.5)))
    return Image.alpha_composite(shadow, img)


def main():
    w, h, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    lines = []
    for i, (t0, t1, layout, big, small) in enumerate(CARDS):
        path = f"{out}/card_{i:02d}.png"
        card(w, h, layout, big, small).save(path)
        lines.append(f"{path} {t0} {t1 - t0:g} {t1 - t0 - 0.9:g}")
    open(f"{out}/cards.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
