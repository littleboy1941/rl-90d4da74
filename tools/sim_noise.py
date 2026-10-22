"""PC preview of the `noise` effect (reference `docs/ref_noise.jpg` in fff):
black-and-white, overexposed, the head smeared sideways in horizontal
bands, everything soft, tones made of coarse 1 px grain.

    python tools/sim_noise.py <image> <out.png> <out.gif> [--face x,y,w,h]

Animation: the grain boils every frame; the bands drift slowly and every
~1 s jerk hard for a few frames (signal tear); moving leaves horizontal
ghost trails (part of the previous frame kept). The test photo is swayed
sideways to stand in for head motion.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

SIDE = 400
FPS = 15
FRAMES = 36


def bands(seed, height, burst):
    """Row bands 3-40 px: horizontal shift and smear length per band."""
    r = np.random.default_rng(seed)
    shift = np.zeros(height)
    smear = np.zeros(height)
    y = 0
    while y < height:
        n = int(r.integers(3, 40))
        if burst:
            s = r.normal(0, 22) if r.random() < 0.5 else r.normal(0, 4)
            l = r.uniform(40, 160)
        else:
            s = r.normal(0, 3)
            l = r.uniform(20, 110) if r.random() < 0.7 else r.uniform(0, 12)
        shift[y:y + n] = s
        smear[y:y + n] = l
        y += n
    return shift, smear


def smear_rows(img, shift, smear, face):
    """Each row: box smear of its band length (centred), then shift.
    Bands act fully on the head rows, half elsewhere."""
    h, w = img.shape
    fx, fy, fw, fh = face
    out = np.empty_like(img)
    cs = np.cumsum(np.pad(img, ((0, 0), (1, 0)), mode="edge"), axis=1)
    x = np.arange(w)
    for y in range(h):
        head = 1.0 if fy - 0.4 * fh < y < fy + 1.3 * fh else 0.5
        l = max(1, int(smear[y] * head))
        a = np.clip(x - l // 2, 0, w)
        b = np.clip(x + 1 + l - l // 2, 0, w)
        row = (cs[y, b] - cs[y, a]) / np.maximum(b - a, 1)
        out[y] = np.interp(x - shift[y] * head, x, row)
    return out


def tone(v, lo, hi):
    """Overexposed auto levels: the darkest 12% of the frame crush to
    black, everything above the 65th percentile blows out to white (the
    backlit look of the reference); in between - grey, then grain."""
    return np.clip((v - lo) / max(hi - lo, 0.05), 0, 1)


def main():
    src, png, gif = sys.argv[1], sys.argv[2], sys.argv[3]
    face = (130, 80, 155, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    base = Image.open(src).convert("L")
    w, h = base.size
    s = min(w, h)
    base = base.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    pad = Image.new("L", (SIDE + 80, SIDE), 255)
    pad.paste(base, (40, 0))
    rng = np.random.default_rng(5)
    frames, acc = [], None
    for f in range(FRAMES):
        t = f / FPS
        sway = int(28 * np.sin(t * 2.4))                       # stand-in head motion
        cur = pad.crop((40 - sway, 0, 40 - sway + SIDE, SIDE)).filter(ImageFilter.GaussianBlur(8))
        cur = np.asarray(cur, np.float32) / 255
        epoch = int(t / 0.35)
        burst = (f % 15) in (9, 10, 11)                         # ~1 s period, 3-frame jerk
        shift, smear = bands(epoch * 7 + (f if burst else 0), SIDE, burst)
        cur = smear_rows(cur, shift, smear, face)
        acc = cur if acc is None else acc * 0.5 + cur * 0.5   # ghost trail
        # Grain before the tone curve: whites and blacks clip clean, the
        # middle tones turn into boiling grain.
        lo, hi = np.percentile(acc, [12, 65])
        v = tone(acc + rng.normal(0, 0.07, acc.shape), lo, hi)
        frames.append(Image.fromarray((v * 255).astype(np.uint8)).convert("RGB"))
    ref = Image.open(r"C:\Users\society\Desktop\fff\docs\ref_noise.jpg").convert("RGB")
    rw, rh = ref.size
    ref = ref.crop((0, (rh - rw) // 2, rw, (rh + rw) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    sheet = Image.new("RGB", (SIDE * 3, SIDE))
    sheet.paste(ref, (0, 0))
    sheet.paste(frames[4], (SIDE, 0))
    sheet.paste(frames[10], (SIDE * 2, 0))
    sheet.save(png)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    print("wrote", png, gif)


if __name__ == "__main__":
    main()
