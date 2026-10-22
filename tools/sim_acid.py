"""PC preview of the `acid` color pass (no hair streaks): AcidShader's
`acid()` ported to numpy, one frame of a still image.

    python tools/sim_acid.py <image> <out.png> [--face x,y,w,h] [--time t]

The face box is in the 400 px square, screen pixels (top down). Mirrors
AcidShader in FaceStyles.swift; keep the numbers in sync.
"""
import sys

import cv2
import numpy as np

SIDE = 400


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def hash3(a, b, c):
    a, b, c = (np.asarray(v).astype(np.uint32) for v in (a, b, c))
    h = (a * np.uint32(73856093)) ^ (b * np.uint32(19349663)) ^ (c * np.uint32(83492791))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / 16777216.0


def vnoise(px, py):
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x, y = (ix + 4096).astype(np.int64), (iy + 4096).astype(np.int64)
    a, b = hash3(x, y, 7), hash3(x + 1, y, 7)
    c, d = hash3(x, y + 1, 7), hash3(x + 1, y + 1, 7)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy


def hsv2rgb(h, s, v):
    k = np.stack([h + 1.0, h + 2 / 3, h + 1 / 3], -1)
    p = np.abs((k - np.floor(k)) * 6 - 3)
    return v[..., None] * (1 + (np.clip(p - 1, 0, 1) - 1) * s)


def main():
    src, out = sys.argv[1], sys.argv[2]
    face = (140, 100, 120, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    t = float(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 1.0
    img = cv2.imread(src)
    h, w = img.shape[:2]
    s = min(h, w)
    img = cv2.resize(img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    rgb = img[..., ::-1].astype(np.float32) / 255
    # CIColorControls saturation 1.3, contrast 1.12 (approximation)
    y = rgb @ np.array([0.2125, 0.7154, 0.0721], np.float32)
    cam = np.clip(((y[..., None] + (rgb - y[..., None]) * 1.3) - 0.5) * 1.12 + 0.5, 0, 1)
    lum = cam @ np.array([0.299, 0.587, 0.114], np.float32)

    sy, xs = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)   # screen rows (top down)
    fx, fy, fw, fh = face
    d = np.hypot((xs - (fx + fw / 2)) / (fw * 0.6), (sy - (fy + fh / 2)) / (fh * 0.65))
    outside = smoothstep(0.9, 1.4, d)
    wave = np.sin(sy * 0.21 + t * 3 + 2 * np.sin(xs * 0.03 + t)) * 5 * outside
    qx = np.clip(xs + wave.astype(np.int64), 0, SIDE - 1).astype(np.int64)
    qy = sy.astype(np.int64)

    def L(dx, dy):
        # dy is in texture rows (bottom up): +1 = one row up on screen
        return lum[np.clip(qy - dy, 0, SIDE - 1), np.clip(qx + dx, 0, SIDE - 1)]

    l = L(0, 0)
    # 3x3 taps 4 px apart: smooth brightness for broad color zones
    smooth = sum(L(dx, dy) for dx in (-4, 0, 4) for dy in (-4, 0, 4)) / 9
    detail = smooth + (l - smooth) * (0.8 + 1.5 * outside)
    c0 = cam[qy, qx]
    orig = np.arctan2(1.732 * (c0[..., 1] - c0[..., 2]), 2 * c0[..., 0] - c0[..., 1] - c0[..., 2]) / 6.2832 + 0.5
    hue = smooth * 2.0 + orig * 0.3 + t * 0.07
    hue -= np.floor(hue)
    value = smoothstep(0.03, 0.3, detail)
    c = hsv2rgb(hue, 1.0, value)
    edge = smoothstep(0.14, 0.3, np.abs(L(4, 0) - L(-4, 0)) + np.abs(L(0, 4) - L(0, -4)))
    c *= (1 - 0.8 * edge)[..., None]

    ex, ey = fx + fw * 0.7, fy + fh * 0.4
    exn, eyn = (xs - ex) / fw, (sy - ey) / fw
    around = 1 - smoothstep(0.12, 0.22, np.hypot(exn, eyn))
    below = (1 - smoothstep(0.1, 0.2, np.abs(exn))) * (eyn >= 0) * (1 - smoothstep(0.35, 0.6, eyn))
    mask = np.maximum(around, below)
    vx, vy = xs * 0.09, (sy - t * 6) * 0.035
    r = vnoise(vx, vy) * 0.65 + vnoise(vx * 2.3, vy * 2.3) * 0.35
    line = 1 - smoothstep(0.02, 0.07, np.abs(r - 0.5))
    k = (line * mask)[..., None]
    c = c * (1 - k) + np.array([0.5, 0.0, 0.04]) * k
    cv2.imwrite(out, (np.clip(c, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    print("saved", out)


if __name__ == "__main__":
    main()
