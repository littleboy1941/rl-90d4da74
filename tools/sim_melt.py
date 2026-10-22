"""PC preview of the `melt` style: the MeltGPU shader and MeltStyle bands
ported to numpy, run on a still image for N frames.

    python tools/sim_melt.py <image> <out.png> [frames] [--face x,y,w,h] [--gif out.gif]

The face box is in the 400 px square, screen pixels (top down).

Mirrors FaceStyles.swift (MeltStyle / MeltGPU); keep the numbers in sync.
SplitMix and the hash are the same, so bands match the phone per epoch.
"""
import sys

import cv2
import numpy as np

SIDE = 400
BLOCK = 4
EPOCH_SECONDS = 3.5
FACE_SHARE = 0.85
BACK_SHARE = 0.4
M64 = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed):
        self.state = seed & M64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def columns(face, epoch):
    """face = (x, y, w, h) in screen pixels (top down)."""
    count = SIDE // BLOCK
    top = np.zeros(count, np.float32)
    speed = np.zeros(count, np.float32)
    refresh = np.ones(count, np.float32)
    band = np.zeros(count, np.int64)
    gen = SplitMix(epoch * 7919 + 17)
    unit = lambda: (gen.next() % 10_000) / 10_000
    fx, fy, fw, fh = face
    left, right = fx - fw * 0.3, fx + fw * 1.3
    i = 0
    while i < count:
        width = 1 + gen.next() % 6
        center = (i + width / 2) * BLOCK
        t, s, r = 0.0, 0.0, 1.0
        if left < center < right:
            if unit() < FACE_SHARE:
                t = fy - 0.15 * fh + unit() * 0.6 * fh
                s = float(1 + gen.next() % 4)
                r = 0.006
        elif unit() < BACK_SHARE:
            t = SIDE * (0.15 + 0.75 * unit())
            s = float(1 + gen.next() % 3)
            r = 0.01
        top[i:i + width], speed[i:i + width], refresh[i:i + width] = t, s, r
        band[i:i + width] = i
        i += width
    return top, speed, refresh, band


def hash3(a, b, c):
    a, b, c = (np.asarray(v, np.uint32) for v in (a, b, c))
    h = (a * np.uint32(73856093)) ^ (b * np.uint32(19349663)) ^ (c * np.uint32(83492791))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / 16777216.0


def grade(rgb):
    # CIColorControls saturation 0.55, contrast 1.12 (approximation)
    y = rgb @ np.array([0.2125, 0.7154, 0.0721], np.float32)
    rgb = y[..., None] + (rgb - y[..., None]) * 0.55
    return np.clip((rgb - 0.5) * 1.12 + 0.5, 0, 1)


def fresh(cam, frame):
    ys, xs = np.mgrid[0:SIDE, 0:SIDE]
    cell = BLOCK * 2
    qy = np.clip((ys // cell) * cell + cell // 2, 0, SIDE - 1)
    qx = np.clip((xs // cell) * cell + cell // 2, 0, SIDE - 1)
    w = np.array([0.299, 0.587, 0.114], np.float32)
    y = cam @ w
    ch = cam[qy, qx]
    cy = ch @ w
    cb = (ch[..., 2] - cy) * 0.564 * 0.8 + 0.02
    cr = (ch[..., 0] - cy) * 0.713 * 0.8 + 0.045
    e = hash3(xs // BLOCK // 2, ys // BLOCK // 2, frame // 20)
    pink, green = e > 0.9975, (e > 0.995) & (e <= 0.9975)
    cb = np.where(pink, 0.06, np.where(green, -0.22, cb))
    cr = np.where(pink, 0.2, np.where(green, -0.08, cr))
    y = y * 0.92 + 0.05
    return np.clip(np.stack([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], -1), 0, 1)


def main():
    src, out = sys.argv[1], sys.argv[2]
    frames = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 60
    gif = sys.argv[sys.argv.index("--gif") + 1] if "--gif" in sys.argv else None
    img = cv2.imread(src)
    h, w = img.shape[:2]
    s = min(h, w)
    img = cv2.resize(img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    face = (140, 100, 120, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    print("face", tuple(int(v) for v in face))
    cam = grade(img[..., ::-1].astype(np.float32) / 255)
    ys, xs = np.mgrid[0:SIDE, 0:SIDE]
    prev = None
    shots = []
    for frame in range(1, frames + 1):
        t = frame / 30
        epoch = int(t / EPOCH_SECONDS)
        top, speed, refresh, band = columns(face, epoch)
        bx = xs // BLOCK
        by = ys // BLOCK
        zone = (speed[bx] > 0) & (ys > top[bx])
        keyframe = prev is None or frame == 1 or int((frame - 1) / 30 / EPOCH_SECONDS) != epoch
        new = fresh(cam, frame)
        if keyframe:
            cur = new
        else:
            redo = ~zone | (hash3(band[bx], by // 4, frame) < refresh[bx])
            py = np.clip(ys - speed[bx].astype(np.int64), 0, SIDE - 1)  # screen: from above
            dragged = prev[py, xs].copy()
            n = hash3(xs, SIDE - 1 - ys, frame)
            dragged = np.where((n > 0.997)[..., None], dragged * 0.5 + np.array([0.95, 0.78, 0.88]) * 0.5, dragged)
            dragged = np.where((n < 0.002)[..., None], dragged * 0.8, dragged)
            cur = np.where(redo[..., None], new, dragged)
        prev = cur.astype(np.float32)
        if gif and frame % 2 == 0:
            shots.append((prev * 255).astype(np.uint8))
    cv2.imwrite(out, (prev[..., ::-1] * 255).astype(np.uint8))
    if gif:
        from PIL import Image
        ims = [Image.fromarray(a) for a in shots]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=66, loop=0)
    print("saved", out)


if __name__ == "__main__":
    main()
