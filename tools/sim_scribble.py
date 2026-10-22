"""PC preview of the `scribble` effect: the head crossed out with a pen
scribble that "boils" (several hand-drawn variants switched a few times a
second, as in drawn animation).

    python tools/sim_scribble.py <image> <out.png> <out.gif> [--face x,y,w,h] [--style ball|marker]

The image is center-cropped to a 400 px square; the face box is in that
square, screen pixels (top down).
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

SIDE = 400
SUPER = 3                     # supersampling for smooth thin lines
VARIANTS = 4                  # boil drawings
BOIL_FPS = 8                  # drawing switches per second


def scribble(seed, size=512, strokes=150, core=110, steps=90):
    """One scribble ball: black lines on transparent, `size` px square,
    ball radius ~0.42 of the size. Strokes are random walks: gentle turning
    (wide loops), now and then a sharp pen kink; pulled back toward the
    centre when they stray. `core` extra strokes stay within half the
    radius, so the middle is nearly solid black and the rim sparse."""
    r = np.random.default_rng(seed)
    big = size * SUPER
    im = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(im)
    c = big / 2
    radius = big * 0.42
    step = big * 0.024
    for index in range(strokes + core):
        inner = index >= strokes
        p = c + r.normal(0, radius * (0.25 if inner else 0.45), 2)
        theta = r.uniform(0, 2 * np.pi)
        omega = 0.0
        reach = radius * (r.uniform(0.35, 0.55) if inner else r.uniform(0.75, 1.2))
        points = [tuple(p)]
        for _ in range(steps):
            omega = omega * 0.85 + r.normal(0, 0.1)
            theta += omega
            if r.random() < 0.06:                  # pen kink
                theta += r.choice([-1, 1]) * r.uniform(1.5, 2.8)
            off = p - c
            dist = np.hypot(*off)
            if dist > reach:
                home = np.arctan2(-off[1], -off[0])
                diff = (home - theta + np.pi) % (2 * np.pi) - np.pi
                theta += diff * 0.4
            p = p + step * np.array([np.cos(theta), np.sin(theta)])
            points.append(tuple(p))
        d.line(points, fill=255, width=int(1.3 * SUPER), joint="curve")
    return im.resize((size, size), Image.LANCZOS)


def marker(seed, size=512, strokes=160):
    """Face scratched out with a marker (reference `docs/ref_scratch.jpg`):
    horizontal strokes of varying thickness over an oval (radius ~0.36 of
    the size, a bit taller than wide), each with a slight slope and
    wobble, ends overshooting the oval at random and tapering; dense in the
    middle (solid), separate strokes at the rim."""
    r = np.random.default_rng(seed)
    big = size * SUPER
    im = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(im)
    c = big / 2
    rx, ry = big * 0.33, big * 0.38
    for _ in range(strokes):
        v = float(np.clip(r.normal(0, 0.5), -1.0, 1.0))
        y = c + v * ry
        half = rx * np.sqrt(max(0.0, 1 - v * v)) * r.uniform(0.85, 1.45)
        shift = r.normal(0, rx * 0.06)
        x0, x1 = c - half + shift, c + half + shift
        slope = r.normal(0, 0.05)
        thick = big * r.uniform(0.006, 0.03) * (1 - 0.5 * abs(v))     # thinner at the rim
        wobble, phase = r.uniform(0, big * 0.006), r.uniform(0, 6.3)
        n = 40
        top, bottom = [], []
        for i in range(n + 1):
            t = i / n
            x = x0 + (x1 - x0) * t
            yy = y + slope * (x - c) + wobble * np.sin(phase + t * 9)
            taper = min(1.0, t / 0.08, (1 - t) / 0.08)
            top.append((x, yy - thick / 2 * taper))
            bottom.append((x, yy + thick / 2 * taper))
        d.polygon(top + bottom[::-1], fill=255)
    return im.resize((size, size), Image.LANCZOS)


def place_marker(frame, patch, face, jitter):
    """Patch over the face oval (hair stays): centred on the face box,
    side ~1.45 face heights."""
    fx, fy, fw, fh = face
    side = int(fh * 1.45)
    cx = fx + fw / 2 + jitter[0]
    cy = fy + fh * 0.45 + jitter[1]
    alpha = patch.resize((side, side), Image.LANCZOS).rotate(jitter[2], resample=Image.BICUBIC)
    ink = Image.new("RGB", (side, side), (6, 6, 6))
    out = frame.copy()
    out.paste(ink, (int(cx - side / 2), int(cy - side / 2)), alpha)
    return out


def place(frame, ball, face, jitter):
    """Ball over the whole head: centre a bit above the face centre (hair),
    diameter ~2.1 face heights."""
    fx, fy, fw, fh = face
    side = int(fh * 2.1)
    cx = fx + fw / 2 + jitter[0]
    cy = fy + fh * 0.28 + jitter[1]
    alpha = ball.resize((side, side), Image.LANCZOS).rotate(jitter[2], resample=Image.BICUBIC)
    ink = Image.new("RGB", (side, side), (8, 8, 8))
    out = frame.copy()
    out.paste(ink, (int(cx - side / 2), int(cy - side / 2)), alpha)
    return out


def main():
    src, png, gif = sys.argv[1], sys.argv[2], sys.argv[3]
    face = (130, 80, 155, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    style = sys.argv[sys.argv.index("--style") + 1] if "--style" in sys.argv else "ball"
    make, put = (marker, place_marker) if style == "marker" else (scribble, place)
    balls = [make(seed) for seed in range(VARIANTS)]
    r = np.random.default_rng(99)
    jitters = [(r.normal(0, 2), r.normal(0, 2), r.normal(0, 3)) for _ in range(VARIANTS)]
    frames = [put(im, balls[i], face, jitters[i]) for i in range(VARIANTS)]
    sheet = Image.new("RGB", (SIDE * 2, SIDE * 2))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % 2) * SIDE, (i // 2) * SIDE))
    sheet.save(png)
    frames[0].save(gif, save_all=True, append_images=frames[1:] * 1, duration=1000 // BOIL_FPS, loop=0)
    print("wrote", png, gif)


if __name__ == "__main__":
    main()
