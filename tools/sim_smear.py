"""PC preview of the `smear` face style (SmearStyle / SmearGPU, same math).

    python tools/sim_smear.py <image> <out.png> [--face x,y,w,h] [--time t ...]

The image is center-cropped to a 400 px square; the face box is in that
square, screen pixels (top down). Several --time values give a strip.
"""
import sys

import numpy as np
from PIL import Image

SIDE = 400


def hash32(x):
    """Same integer hash as `hash32` in the Metal source (uint32 wraparound)."""
    x = np.asarray(x, dtype=np.uint64) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def unit(*keys):
    """Hash of several integers -> [0, 1)."""
    h = np.uint64(0x9E3779B9)
    for k in keys:
        h = hash32((np.asarray(k, dtype=np.uint64) + h * np.uint64(31)) & 0xFFFFFFFF)
    return (h & 0xFFFF).astype(np.float32) / 65536.0


def grade(c):
    """Muted grey-green look of the reference: low saturation, lifted blacks."""
    l = (c * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
    c = l + (c - l) * 0.12
    c = c * 0.86 + 0.07
    return c * np.array([0.97, 1.0, 0.98], np.float32)


def smear(src, face, t):
    h, w, _ = src.shape
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    cam = grade(src)
    out = cam.copy()

    # Vertical pass: one thread per column, rows top down.
    x = np.arange(w)
    epoch = np.floor(t / 0.6 + unit(x, 7) * 3).astype(np.int64)    # staggered changes
    band = x // 9                                                   # neighbouring columns alike
    r_on, r_start, r_len, r_alpha, r_tint, r_gain = (
        0.6 * unit(x, epoch, k) + 0.4 * unit(band, epoch, k + 10) for k in range(6))
    u = (x - cx) / (fw * 0.55)
    head = (np.abs(u) < 1) & (r_on < 0.93)
    ellipse = 1 - np.sqrt(np.clip(1 - u * u, 0, 1))
    start = fy + fh * (-0.02 + 0.16 * r_start + 0.3 * ellipse)
    end = fy + fh * (1.0 + 1.7 * r_len ** 0.7)
    end = np.where(r_len > 0.8, h, end)
    alpha = 0.003 + 0.09 * r_alpha ** 2.5
    # Faint ghost streaks beside the head.
    ghost = (~head) & (np.abs(u) < 2.2) & (r_on < 0.06)
    start = np.where(ghost, fy + fh * r_start, start)
    end = np.where(ghost, start + fh * (0.5 + r_len), end)
    alpha = np.where(ghost, 0.02, alpha)
    active = head | ghost
    tint = np.ones((w, 3), np.float32)
    tint[r_tint < 0.08] = (0.86, 1.04, 1.12)
    tint[(r_tint >= 0.08) & (r_tint < 0.15)] = (1.1, 0.88, 1.08)
    tint[(r_tint >= 0.15) & (r_tint < 0.21)] = (1.08, 1.06, 0.84)
    gain = (0.86 + 0.28 * r_gain)[:, None] * tint
    acc = np.zeros((w, 3), np.float32)
    started = np.zeros(w, bool)
    for y in range(h):
        row = cam[y]
        inside = active & (y >= start) & (y < end)
        fresh = inside & ~started
        acc[fresh] = row[fresh]
        started |= inside
        acc = np.where(inside[:, None], acc + (row - acc) * alpha[:, None], acc)
        streak = acc * gain
        # Ragged end: last 6 px blend back to the camera.
        fade = np.clip((end - y) / 6, 0, 1)[:, None]
        mixed = row + (streak - row) * fade
        mixed = np.where(ghost[:, None], row + (mixed - row) * 0.5, mixed)
        out[y] = np.where(inside[:, None], mixed, row)

    # Horizontal pass over the hair band: rows dragged to the right.
    res = out.copy()
    for y in range(int(max(0, fy - 0.55 * fh)), int(min(h, fy + 0.35 * fh))):
        e = int(np.floor(t / 0.6 + unit(y, 9) * 3))
        if (0.6 * unit(y, e, 20) + 0.4 * unit(y // 3, e, 21)) > 0.6:
            continue
        a = 0.03 + 0.17 * unit(y, e, 22)
        x0 = int(max(0, cx - fw * 0.6))
        x1 = int(min(w, cx + fw * (0.5 + 0.8 * unit(y, e, 23))))
        acc = out[y, x0].copy()
        for xx in range(x0, x1):
            acc += (out[y, xx] - acc) * a
            res[y, xx] = acc
    return np.clip(res, 0, 1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    face = (140, 100, 120, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    times = [float(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--time"] or [0.0]
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    img = np.asarray(im, np.float32) / 255
    frames = [smear(img, face, t) for t in times]
    strip = np.concatenate(frames, axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
