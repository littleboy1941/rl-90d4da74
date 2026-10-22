"""PC preview of the `tv_static` effect (reference `docs/ref_tvstatic.jpg` in
fff): the person's silhouette is filled with TV snow on black. The snow
is made of soft blobs stretched sideways with cyan / orange fringes (the
CRT's colour misregistration); it is blown out to near white at the top
and thins into separate dark-gapped blobs towards the bottom; a glow
spills around the silhouette; faint vertical aperture-grille lines; a
few dust specks on the black.

    python tools/sim_tvstatic.py <photo> <out.png> <out.gif> [--rect x0,y0,x1,y1] [--bg camera|black] [--fg x0,y0,x1,y1]

`--bg camera` keeps the room around the person (the phone effect),
`--bg black` puts the person on black (as in the reference). `--fg` marks
a box that is surely the person (helps GrabCut).

The silhouette here comes from OpenCV GrabCut inside `--rect` (on the
phone: Vision person segmentation, both cameras). Animation: new snow
every frame (it boils, as on a TV), slight overall flicker.
"""
import sys

import cv2
import numpy as np
from PIL import Image

SIDE = 400
FRAMES = 16
FPS = 25


def silhouette(photo, rect, sure=None):
    img = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    x0, y0, x1, y1 = rect
    if sure is None:
        cv2.grabCut(img, mask, (x0, y0, x1 - x0, y1 - y0), bg, fg, 6, cv2.GC_INIT_WITH_RECT)
    else:
        # Sure person: an ellipse in the `sure` box (head) and the bottom
        # band under it (shoulders); maybe person around them; the rest of
        # `rect` maybe background, outside it surely background.
        mask[:] = cv2.GC_BGD
        mask[y0:y1, x0:x1] = cv2.GC_PR_BGD
        a0, b0, a1, b1 = sure
        centre, axes = ((a0 + a1) // 2, (b0 + b1) // 2), ((a1 - a0) // 2, (b1 - b0) // 2)
        cv2.ellipse(mask, centre, (int(axes[0] * 1.3), int(axes[1] * 1.15)), 0, 0, 360, cv2.GC_PR_FGD, -1)
        mask[b1:, x0:x1] = cv2.GC_PR_FGD
        cv2.ellipse(mask, centre, axes, 0, 0, 360, cv2.GC_FGD, -1)
        mask[(b1 + y1) // 2:, (a0 + x0) // 2:(a1 + x1) // 2] = cv2.GC_FGD
        cv2.grabCut(img, mask, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def snow(rng, h, w):
    """Blobs ~4 x 2 px (stretched sideways), smooth, 0..1."""
    n = rng.random((h // 2 + 2, w // 4 + 2)).astype(np.float32)
    n = cv2.resize(n, (w + 8, h + 4), interpolation=cv2.INTER_CUBIC)[2:2 + h, 4:4 + w]
    return np.clip((n - 0.5) * 1.6 + 0.5, 0, 1)


def frame(mask, rng, flicker, camera=None):
    h, w = mask.shape
    s = snow(rng, h, w)
    # Exposure: blown out at the top of the silhouette, thinning to the bottom.
    rows = np.where(mask.max(1) > 0.5)[0]
    top, bottom = (rows.min(), rows.max()) if len(rows) else (0, h)
    t = np.clip((np.arange(h) - top) / max(bottom - top, 1), 0, 1)[:, None]
    gain = 2.8 - 2.0 * t
    cut = 0.05 + 0.45 * t                       # below this a blob is a dark gap
    v = np.clip((s - cut) / (1 - cut), 0, 1) * gain * flicker
    # Colour misregistration: red / blue sampled 1.5 px apart -> fringes.
    r = np.roll(v, 2, axis=1) * 1.0
    g = v * 0.97
    b = np.roll(v, -2, axis=1) * 1.05
    rgb = np.stack([r, g, b], -1) * mask[..., None]
    rgb *= np.array([1.0, 0.96, 0.92], np.float32)          # warm white
    # Glow spilling around (and over) the silhouette.
    glow = cv2.GaussianBlur(np.clip(rgb, 0, 1.5), (0, 0), 9) * 0.55
    out = rgb + glow * np.array([0.9, 0.95, 1.0], np.float32)
    # Aperture grille and dust on the black.
    out *= (0.88 + 0.12 * np.cos(np.arange(w) * 2 * np.pi / 2.4))[None, :, None]
    dust = (rng.random((h, w)) > 0.9994).astype(np.float32)[..., None] * 0.8
    if camera is None:
        out = out + dust * (1 - mask[..., None])
    else:
        out = camera * (1 - mask[..., None]) + out
    return np.clip(out, 0, 1)


def main():
    src, png, gif = sys.argv[1], sys.argv[2], sys.argv[3]
    rect = (20, 5, 380, 399)
    if "--rect" in sys.argv:
        rect = tuple(int(v) for v in sys.argv[sys.argv.index("--rect") + 1].split(","))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    photo = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    sure = None
    if "--fg" in sys.argv:
        sure = tuple(int(v) for v in sys.argv[sys.argv.index("--fg") + 1].split(","))
    mask = silhouette(photo, rect, sure)
    camera = None
    if "--bg" in sys.argv and sys.argv[sys.argv.index("--bg") + 1] == "camera":
        camera = np.asarray(photo, np.float32) / 255
    rng = np.random.default_rng(7)
    frames = [frame(mask, rng, 1 + rng.normal(0, 0.04), camera) for _ in range(FRAMES)]
    ims = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    ref = Image.open(r"C:\Users\society\Desktop\fff\docs\ref_tvstatic.jpg").convert("RGB")
    rw, rh = ref.size
    ref = ref.crop((0, (rh - rw) // 2, rw, (rh + rw) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    sheet = Image.new("RGB", (SIDE * 3, SIDE))
    sheet.paste(ref, (0, 0))
    sheet.paste(photo, (SIDE, 0))
    sheet.paste(ims[0], (SIDE * 2, 0))
    sheet.save(png)
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=1000 // FPS, loop=0)
    print("wrote", png, gif)


if __name__ == "__main__":
    main()
