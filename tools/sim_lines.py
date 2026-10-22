"""PC preview of the `lines` look (reference `docs/ref_lines.jpg` in fff):
the head drawn with thin vertical lines on white paper. Line width follows
the darkness (thin where light, merging into solid black in the eye
sockets); darkness drips down each column with its own length, so dark
features run down in ragged streaks; thin lines overshoot above the head
and trail far below the chin.

    python tools/sim_lines.py <image> <out.png> [--face x,y,w,h] [--ref ref.jpg] [--skull ref.jpg]

The head is an ellipse around the face box here (on the phone: the head
mask from the face-parsing network).
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

SIDE = 400
PERIOD = 2.6                  # line spacing, px of the 400 px frame
SUPER = 4                     # horizontal supersampling for line edges


def lines(img, face, seed=1, skull=None):
    fx, fy, fw, fh = face
    r = np.random.default_rng(seed)
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(3.0)), np.float32) / 255
    h, w = g.shape
    # Darkness, contrast locked on the face: its mean -> 0.22 (thin lines),
    # shadows 0.35 darker -> solid black.
    mid = float(g[int(fy + fh * 0.2):int(fy + fh * 0.8), int(fx + fw * 0.2):int(fx + fw * 0.8)].mean())
    dark = np.clip((mid - g) / 0.35 + 0.22, 0, 1)
    # Head: ellipse a bit taller than the face box (forehead and hair).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = fx + fw / 2, fy + fh * 0.42
    e = ((xx - cx) / (fw * 0.6)) ** 2 + ((yy - cy) / (fh * 0.72)) ** 2
    head = np.clip((1.0 - e) / 0.25, 0, 1)
    if skull is not None:
        # The face replaced by a skull: its darkness map stretched over the
        # head ellipse's box (on the phone: placed by the head pose).
        # Box: 1.45 face heights tall, the skull map's own aspect (0.66).
        sh = fh * 1.45
        sw = sh * skull.size[0] / skull.size[1]
        x0, x1 = int(cx - sw / 2), int(cx + sw / 2)
        y0, y1 = int(cy - sh * 0.52), int(cy + sh * 0.48)
        sk = np.asarray(skull.resize((x1 - x0, y1 - y0), Image.LANCZOS), np.float32) / 255
        dark = np.zeros_like(g)
        ys0, xs0 = max(y0, 0), max(x0, 0)
        dark[ys0:min(y1, h), xs0:min(x1, w)] = sk[ys0 - y0:min(y1, h) - y0, xs0 - x0:min(x1, w) - x0]
        head = (dark > 0.02).astype(np.float32)     # the skull's own outline
    dark *= head
    # Per column: drip decay (dark runs down with its own length).
    # (The skull map already has its own drips: almost none on top.)
    col_decay = 1 - (r.uniform(0.2, 0.4, w) if skull is not None else r.uniform(0.008, 0.07, w))
    out = np.zeros_like(dark)
    run = np.zeros(w, np.float32)
    for y in range(h):
        run = np.maximum(dark[y], run * col_decay)
        out[y] = run
    # Hairline tails from the head outline: up to ~0.45 face heights above,
    # up to ~1.6 below, each column its own length, fading out.
    inside = head > 0.3
    has = inside.any(0)
    top = np.where(has, inside.argmax(0), h).astype(np.float32)
    bottom = np.where(has, h - 1 - inside[::-1].argmax(0), -1).astype(np.float32)
    up = r.uniform(0.05, 1, w) ** 1.2 * fh * 0.45
    down = r.uniform(0.15, 1, w) ** 0.9 * fh * 1.5
    weight = r.uniform(0.1, 0.3, w)
    t_up = np.clip((yy - (top - up)[None, :]) / (up[None, :] + 1e-3), 0, 1) * (yy < top[None, :])
    t_dn = np.clip(((bottom + down)[None, :] - yy) / (down[None, :] + 1e-3), 0, 1) * (yy > bottom[None, :])
    tails = weight[None, :] * np.maximum(t_up, t_dn) * has[None, :]
    out = np.maximum(out, tails)
    # Line screen: in each column cell of PERIOD px, ink covers width
    # PERIOD * darkness^1.5 around a slightly jittered line centre; with
    # supersampling a very thin line comes out as a light grey hairline.
    xs = (np.arange(w * SUPER) + 0.5) / SUPER
    cell = np.floor(xs / PERIOD).astype(int)
    jitter = r.uniform(-0.3, 0.3, cell.max() + 1) * PERIOD
    centre = (cell + 0.5) * PERIOD + jitter[cell]
    src_x = np.clip(centre.astype(int), 0, w - 1)
    width = out[:, src_x] ** (1.0 if skull is not None else 1.5) * PERIOD * 1.1
    ink = (np.abs(xs - centre)[None, :] < width / 2).astype(np.float32)
    ink = ink.reshape(h, w, SUPER).mean(-1)
    return np.clip(0.98 * (1 - ink) + 0.03 * ink, 0, 1)


def skull_darkness(ref):
    """Darkness map of the skull from the reference itself: the line screen
    is blurred away sideways, the head part (no fringe, no drips) cropped."""
    g = ref.convert("L").crop((90, 150, 380, 590))
    g = np.asarray(g, np.float32) / 255
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3.5) ** 2)
    k /= k.sum()
    g = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, g)
    dark = np.clip(1 - g, 0, 1)
    return Image.fromarray((dark * 255).astype(np.uint8))


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
        ref = Image.open(sys.argv[sys.argv.index("--ref") + 1]).convert("RGB")
        rw, rh = ref.size
        canvas = Image.new("RGB", (rh, rh), (255, 255, 255))
        canvas.paste(ref, ((rh - rw) // 2, 0))
        tiles.append(canvas.resize((SIDE, SIDE), Image.LANCZOS))
    img = square(Image.open(src).convert("RGB"))
    skull = None
    if "--skull" in sys.argv:
        skull = skull_darkness(Image.open(sys.argv[sys.argv.index("--skull") + 1]))
    v = lines(img, face, skull=skull)
    tiles.append(Image.fromarray((v * 255).astype(np.uint8)).convert("RGB"))
    sheet = Image.new("RGB", (SIDE * len(tiles), SIDE))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * SIDE, 0))
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
