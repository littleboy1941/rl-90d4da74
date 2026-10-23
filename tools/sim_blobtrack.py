"""PC preview of `blob tracking` for the back camera (author 2026-10-23,
references in fff PROJECT_NOTES, "breakcore для задней камеры"; v2 with
corner brackets and `x:y` labels was "too sloppy", v3 follows the neater
reference `1979402827_1_tiktok_...mp4`):
  - the picture as a black-and-white engraving (high-pass + threshold),
    `--colour` keeps the camera colours instead;
  - thin 1 px rectangles on the textured spots only, many sizes,
    overlapping; some are inverted inside;
  - a 4-digit number in a monospace font inside each box;
  - a small red dot in the centre of most boxes;
  - long thin lines joining box centres.

    python tools/sim_blobtrack.py <clip.mp4> <out.mp4> [--colour]

Found on a 100 px copy of the frame (the phone can do the same on the
CPU): boxes are the most detailed spots of the frame (local contrast),
at two scales (big and small boxes). Tracked: matched to the
nearest of the last frame, eased, shown after 2 frames, kept 4 frames.
"""
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIDE = 400
FPS = 30
SMALL = 100
SCALE = SIDE / SMALL
FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 10)
WHITE = (235, 235, 235)
RED = (200, 20, 30)


class Tracks:
    def __init__(self, reach, ease):
        self.items = []          # dict: id, box (x0, y0, x1, y1), seen, lost
        self.reach = reach
        self.ease = ease
        self.next_id = 0

    def update(self, boxes):
        for it in self.items:
            it["hit"] = False
        for box in boxes:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, dist = None, self.reach
            for it in self.items:
                if it["hit"]:
                    continue
                b = it["box"]
                d = np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                if d < dist:
                    best, dist = it, d
            if best is None:
                self.items.append({"id": self.next_id, "box": np.array(box, float), "seen": 1, "lost": 0, "hit": True})
                self.next_id += 1
            else:
                best["box"] += (np.array(box, float) - best["box"]) * self.ease
                best["seen"] += 1
                best["lost"] = 0
                best["hit"] = True
        for it in self.items:
            if not it["hit"]:
                it["lost"] += 1
        self.items = [it for it in self.items if it["lost"] <= 4]
        return [it for it in self.items if it["seen"] >= 2]


DITHER = np.random.default_rng(5).random((SIDE, SIDE)).astype(np.float32)


def engrave(gray):
    """Stipple: dark stays black, light parts and edges turn into fine
    white specks (fixed dither noise). Brightness is stretched per frame
    (2nd..99th percentile) and bent down, so only the light parts are dense."""
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 99)
    g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    v = np.clip(g ** 2.2 * 0.9 + np.abs(hp) * 3, 0, 1)
    return (v > DITHER).astype(np.uint8) * 255


def find_boxes(gray_small):
    """Boxes on the most detailed spots: local contrast (std in a small
    window), the top 22 % of this frame, connected, at two scales."""
    g = gray_small.astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (5, 5)) - mean * mean, 0))
    found = []
    for sigma, top, limit in ((2.5, 0.12, 8), (0.8, 0.22, 14)):
        detail = cv2.GaussianBlur(std, (0, 0), sigma)
        spots = (detail >= np.quantile(detail, 1 - top)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(spots)
        level_boxes = []
        for x, y, w, h, area in stats[1:]:
            touches = x <= 0 or y <= 0 or x + w >= SMALL or y + h >= SMALL
            if 3 <= max(w, h) <= 30 and area >= 5 and not touches:
                level_boxes.append((area, (x, y, x + w, y + h)))
        level_boxes.sort(reverse=True)
        found += [box for _, box in level_boxes[:limit]]
    return found


def number(track_id):
    return f"{(track_id * 7919 + 1973) % 9000 + 1000}"


def hud(base, boxes):
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    centres = []
    for it in sorted(boxes, key=lambda it: it["id"]):
        x0, y0, x1, y1 = (it["box"] * SCALE).round().astype(int)
        x0, y0, x1, y1 = max(x0 - 3, 0), max(y0 - 3, 0), min(x1 + 3, SIDE - 1), min(y1 + 3, SIDE - 1)
        if it["id"] % 4 == 1 and x1 > x0 and y1 > y0:
            region = im.crop((x0, y0, x1, y1))
            im.paste(Image.eval(region, lambda v: 255 - v), (x0, y0))
        draw.rectangle([x0, y0, x1, y1], outline=WHITE, width=1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        centres.append((cx, cy))
        if x1 - x0 >= 22:
            draw.text((cx - 4, cy - 12), number(it["id"]), fill=WHITE, font=FONT)
        if it["id"] % 3 != 2:
            draw.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=RED)
    # Long thin lines: every box to the one two ids further.
    for a, b in zip(centres, centres[2:]):
        draw.line([a, b], fill=(200, 200, 200), width=1)
    return np.asarray(im)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    colour = "--colour" in sys.argv
    cap = cv2.VideoCapture(src)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIDE}x{SIDE}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dst],
        stdin=subprocess.PIPE)
    tracks = Tracks(reach=8, ease=0.6)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ink = engrave(gray)
        base = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if colour else np.repeat(ink[..., None], 3, -1)
        gray_small = cv2.resize(gray, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
        boxes = tracks.update(find_boxes(gray_small))
        enc.stdin.write(hud(base, boxes).tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", dst)


if __name__ == "__main__":
    main()
