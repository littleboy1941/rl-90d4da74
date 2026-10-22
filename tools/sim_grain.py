"""PC preview of the `grain` look (reference `docs/ref_grain.jpg` in fff):
black-and-white, flash-lit face out of the dark, soft focus, everything
made of coarse 1 px grain; blacks lifted to dark grey, whites slightly dim.

    python tools/sim_grain.py <image> <out.png> [--face x,y,w,h] [--ref ref.jpg]

Recipe: grey -> soft blur -> flash falloff (light drops away from the
face, so the background sinks into the dark) -> exposure locked on the
face (its mean goes to near white, ~0.27 darker is already black) -> noise before the curve (the
curve clips it, so mid tones become a mix of dark and light dots) ->
output range 0.09...0.94.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

SIDE = 400


def grain(img, face, seed=0, blur=3.0, window=0.35, noise=0.12, falloff=1.0):
    fx, fy, fw, fh = face
    g = img.convert("L").filter(ImageFilter.GaussianBlur(blur))
    g = np.asarray(g, np.float32) / 255
    # Flash: full light on the face, ~35% a face height or two away.
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]].astype(np.float32)
    d2 = ((xx - (fx + fw / 2)) / fw) ** 2 + ((yy - (fy + fh * 0.45)) / fh) ** 2
    g = g * (0.35 + 0.65 * np.exp(-d2 / (2 * falloff ** 2)))
    mid = float(g[int(fy + fh * 0.15):int(fy + fh * 0.75), int(fx + fw * 0.2):int(fx + fw * 0.8)].mean())
    rng = np.random.default_rng(seed)
    v = (g + rng.normal(0, noise, g.shape) - (mid - 0.27)) / window
    v = np.clip(v, 0, 1)
    return 0.09 + 0.85 * v


def square(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)


def main():
    src, out = sys.argv[1], sys.argv[2]
    face = (130, 80, 155, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    tiles = []
    if "--ref" in sys.argv:
        tiles.append(square(Image.open(sys.argv[sys.argv.index("--ref") + 1]).convert("RGB")))
    v = grain(square(Image.open(src).convert("RGB")), face)
    tiles.append(Image.fromarray((v * 255).astype(np.uint8)).convert("RGB"))
    sheet = Image.new("RGB", (SIDE * len(tiles), SIDE))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * SIDE, 0))
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
