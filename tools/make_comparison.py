"""Compose the qualitative comparison grid (paper Fig. 4) from per-method error maps.

Each dataset folder in ``--root`` holds ``t1.png, t2.png, gt.png, scribble.png`` and one
error map per method (TP white, TN black, FP red, FN green). Pass ``--extra NAME=PATTERN``
to append your own results, e.g. ``--extra Ours=results/{dataset}/error_map.png``.

Example:
    python tools/make_comparison.py --out assets/comparison.png
"""
import argparse
import os

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

COLUMNS = [("t1", "T1 image"), ("t2", "T2 image"), ("gt", "GT"), ("scribble", "Scribble"),
           ("irg-mcs", "IRG-McS"), ("npsg", "NPSG"), ("scasc", "SCASC"), ("gir-mrf", "GIR-MRF"),
           ("sdir", "SDIR"), ("srf", "SRF"), ("sg-srf", "SG-SRF"), ("shift-base", "SHIFT-Base"),
           ("shift-post", "SHIFT-Post"), ("sgca", "SGCA (ours)")]
LEGEND = [((255, 255, 255), "TP"), ((0, 0, 0), "TN"), ((255, 0, 0), "FP"), ((0, 255, 0), "FN")]


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="assets/comparison")
    p.add_argument("--datasets", nargs="*", default=[f"hetdata{i}" for i in range(1, 7)])
    p.add_argument("--extra", nargs="*", default=[], help="NAME=PATTERN with {dataset} placeholder")
    p.add_argument("--tile", type=int, default=150, help="tile width in pixels")
    p.add_argument("--out", default="assets/comparison.png")
    args = p.parse_args()

    cols = [(os.path.join(args.root, "{dataset}", f"{k}.png"), label) for k, label in COLUMNS]
    cols += [(pattern, name) for name, pattern in (e.split("=", 1) for e in args.extra)]

    tw, gap, left, top = args.tile, 4, 150, 46
    rows = []
    for ds in args.datasets:
        tiles = []
        for pattern, _ in cols:
            im = Image.open(pattern.format(dataset=ds)).convert("RGB")
            tiles.append(im.resize((tw, round(im.height * tw / im.width)), Image.NEAREST))
        rows.append((ds, tiles, max(t.height for t in tiles)))

    width = left + len(cols) * (tw + gap)
    height = top + sum(h + gap for *_, h in rows) + 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f_head, f_row = font(15, bold=True), font(17, bold=True)

    for j, (_, label) in enumerate(cols):
        x = left + j * (tw + gap) + tw // 2
        draw.text((x, top - 12), label, fill="black", font=f_head, anchor="ms")
    y = top
    for ds, tiles, h in rows:
        label = ds.replace("hetdata", "HetData #")
        draw.text((left - 12, y + h // 2), label, fill="black", font=f_row, anchor="rm")
        for j, t in enumerate(tiles):
            canvas.paste(t, (left + j * (tw + gap), y + (h - t.height) // 2))
        y += h + gap

    x = left
    for color, name in LEGEND:
        draw.rectangle([x, y + 18, x + 26, y + 40], fill=color, outline="black")
        draw.text((x + 34, y + 29), name, fill="black", font=f_row, anchor="lm")
        x += 110
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(f"saved {args.out} ({width}x{height})")


if __name__ == "__main__":
    main()
