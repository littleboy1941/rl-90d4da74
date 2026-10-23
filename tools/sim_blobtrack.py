"""PC preview of `blob tracking` for the back camera (author 2026-10-23,
references in fff PROJECT_NOTES, "breakcore для задней камеры"): a white
HUD over the camera - corner brackets around blobs with `x:### y:###`,
small squares on points, thin lines between neighbours, a dashed line
through the blobs. Only the HUD for now (no glitch fill).

    python tools/sim_blobtrack.py <clip.mp4> <out.mp4> [--sheet out.png]

Everything is found on a 100 px copy of the frame, so the phone can do
the same on the CPU:
  blobs  - connected spots of (motion | the brightest 6 %), 4..40 % of
           the side, not touching the frame edge; up to 4, the biggest;
  points - Shi-Tomasi corners, up to 8, at least 12 px apart.
Lines longer than 30 % of the side (dashed: 50 %) are not drawn.
Both are tracked: matched to the nearest of the last frame, the position
eased, shown after 2 frames seen, kept 4 frames after being lost.
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
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 11)
WHITE = (255, 255, 255)
MAX_LINE = SIDE * 0.3      # longer thin lines are dropped
MAX_DASH = SIDE * 0.5


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


def find_blobs(gray, prev):
    motion = cv2.GaussianBlur(cv2.absdiff(gray, prev), (0, 0), 1.5) > 18
    bright = gray >= np.percentile(gray, 94)
    spots = (motion | bright).astype(np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(spots)
    found = []
    for x, y, w, h, area in stats[1:]:
        side = max(w, h)
        touches = x <= 0 or y <= 0 or x + w >= SMALL or y + h >= SMALL
        if 4 <= side <= 40 and area >= 6 and not touches:
            found.append((area, (x, y, x + w, y + h)))
    found.sort(reverse=True)
    return [box for _, box in found[:4]]


def find_points(gray):
    corners = cv2.goodFeaturesToTrack(gray, 8, 0.08, 12)
    if corners is None:
        return []
    return [(x - 1, y - 1, x + 1, y + 1) for x, y in corners.reshape(-1, 2)]


def dashed(draw, a, b, width=2, dash=7, gap=5):
    a, b = np.array(a, float), np.array(b, float)
    length = np.hypot(*(b - a))
    if length < 1:
        return
    u = (b - a) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        draw.line([tuple(a + u * s), tuple(a + u * e)], fill=WHITE, width=width)
        s = e + gap


def brackets(draw, box, arm=0.3):
    x0, y0, x1, y1 = box
    ax, ay = max(4, (x1 - x0) * arm), max(4, (y1 - y0) * arm)
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(cx, cy), (cx + sx * ax, cy)], fill=WHITE, width=2)
        draw.line([(cx, cy), (cx, cy + sy * ay)], fill=WHITE, width=2)


def hud(img, blobs, points):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    centres = []
    for it in blobs:
        x0, y0, x1, y1 = it["box"] * SCALE
        pad = 6 + max(0, 24 - max(x1 - x0, y1 - y0)) / 2   # brackets at least ~36 px
        box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        brackets(draw, box)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        centres.append((cx, cy))
        draw.text((box[2] + 3, cy - 6), f"x:{int(cx)} y:{int(cy)}", fill=WHITE, font=FONT)
    pts = []
    for it in points:
        x0, y0, x1, y1 = it["box"] * SCALE
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts.append((cx, cy))
        r = 5
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=1)
        if it["id"] % 3 == 0:
            draw.text((cx + r + 3, cy - r - 2), f"x:{int(cx)} y:{int(cy)}", fill=WHITE, font=FONT)
    # Thin lines: every point to its nearest other point and to the
    # nearest blob; a dashed line through the blobs from left to right.
    for i, p in enumerate(pts):
        others = [q for j, q in enumerate(pts) if j != i]
        if others:
            q = min(others, key=lambda q: np.hypot(q[0] - p[0], q[1] - p[1]))
            if np.hypot(q[0] - p[0], q[1] - p[1]) < MAX_LINE:
                draw.line([p, q], fill=WHITE, width=1)
        if centres:
            c = min(centres, key=lambda c: np.hypot(c[0] - p[0], c[1] - p[1]))
            if np.hypot(c[0] - p[0], c[1] - p[1]) < MAX_LINE:
                draw.line([p, c], fill=WHITE, width=1)
    for a, b in zip(sorted(centres), sorted(centres)[1:]):
        if np.hypot(b[0] - a[0], b[1] - a[1]) < MAX_DASH:
            dashed(draw, a, b)
    return np.asarray(im)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cap = cv2.VideoCapture(src)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIDE}x{SIDE}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dst],
        stdin=subprocess.PIPE)
    blob_tracks, point_tracks = Tracks(reach=12, ease=0.6), Tracks(reach=6, ease=0.7)
    prev, kept = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(cv2.resize(frame, (SMALL, SMALL), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        blobs = blob_tracks.update(find_blobs(gray, prev if prev is not None else gray))
        points = point_tracks.update(find_points(gray))
        prev = gray
        out = hud(rgb, blobs, points)
        enc.stdin.write(out.tobytes())
        if len(kept) < 48:
            kept.append(out)
    enc.stdin.close()
    enc.wait()
    if "--sheet" in sys.argv:
        picks = kept[::4][:12]
        rows = [np.concatenate(picks[r * 4:(r + 1) * 4], 1) for r in range(3)]
        sheet = np.concatenate(rows, 0)
        cv2.imwrite(sys.argv[sys.argv.index("--sheet") + 1], cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", dst)


if __name__ == "__main__":
    main()
