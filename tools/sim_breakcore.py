"""PC preview of `breakcore` for the back camera (author 2026-10-23: the TV
static stays on the front camera; the back one films the room and the
author's own hands and legs). No person is looked for: the whole frame
is cut in time to a fixed breakbeat grid, the way breakcore chops a drum
break.

    python tools/sim_breakcore.py <clip.mp4> <out.mp4> [--sheet out.png]

Grid: 175 BPM, a step is a 16th (~2.6 frames at 30 fps), a loop is two
bars (32 steps). Every step has one move, from a fixed pattern (a
repeating break, not random noise):
  play     - the live frame;
  kick     - zoom punch 1.22x decaying over the step + short flash;
  snare    - a frame of TV snow over the picture (the same snow as the
             front camera's `tv_static`) + colour split;
  stutter  - replays the start of an earlier step ("ta-ta-ta");
  reverse  - runs back in time from the step start;
  freeze   - holds the step's first frame;
  roll     - end of the loop: retriggers faster and faster with a
             growing zoom and snow, snaps back on the next downbeat.
Every move only looks back in time, so on the phone it is a ring buffer
of the last ~1 s of frames plus one shader.
"""
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from sim_tvstatic import snow  # noqa: E402

SIDE = 400
FPS = 30
BPM = 175
STEP = 60 / BPM / 4                     # seconds per 16th

# Two bars of 16 steps. (move, argument): stutter -> anchor step.
BAR1 = ["kick", "play", "play", "play", "snare", "play", "play", "play",
        "play", ("stutter", 8), "kick", "play", "snare", "reverse", "reverse", "freeze"]
BAR2 = ["kick", "play", "kick", "play", "snare", ("stutter", 4), ("stutter", 4), "play",
        "roll", "roll", "roll", "roll", "roll", "roll", "roll", "roll"]
PATTERN = BAR1 + BAR2


def read(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    return frames


def zoom(img, k):
    if k <= 1.001:
        return img
    m = cv2.getRotationMatrix2D((SIDE / 2, SIDE / 2), 0, k)
    return cv2.warpAffine(img, m, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def split(img, px):
    if px < 0.5:
        return img
    d = int(round(px))
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], d, axis=1)
    out[..., 2] = np.roll(img[..., 2], -d, axis=1)
    return out


def static(img, rng, amount):
    s = snow(rng, SIDE, SIDE)
    v = np.stack([np.roll(s, 2, 1), s * 0.97, np.roll(s, -2, 1) * 1.05], -1) * np.array([1.0, 0.96, 0.92], np.float32)
    glow = cv2.GaussianBlur(v, (0, 0), 9) * 0.4
    return img * (1 - amount) + np.clip(v * 1.3 + glow, 0, 1.4) * amount


def grade(img):
    # A bit harder and colder than the camera, so the cuts read.
    grey = img.mean(-1, keepdims=True)
    img = grey + (img - grey) * 0.75
    return np.clip((img - 0.5) * 1.18 + 0.5, 0, 1)


def render(frames):
    rng = np.random.default_rng(3)
    out = []
    n = len(frames)
    for i in range(n):
        t = i / FPS
        step = int(t / STEP)
        start = step * STEP                     # time this step began
        phase = (t - start) / STEP              # 0..1 inside the step
        move = PATTERN[step % len(PATTERN)]
        name, arg = (move, None) if isinstance(move, str) else move
        loop_start = (step - step % len(PATTERN)) * STEP
        src_t, k, flash, px, snowy = t, 1.0, 0.0, 0.0, 0.0
        if name == "kick":
            k = 1 + 0.22 * np.exp(-phase * 3)
            flash = 0.35 * np.exp(-phase * 5)
        elif name == "snare":
            snowy = 0.75 if phase < 0.45 else 0.0
            px = 7 * np.exp(-phase * 3)
        elif name == "stutter":
            anchor = loop_start + arg * STEP + (16 * STEP if step % len(PATTERN) >= 16 else 0)
            src_t = anchor + (t - start)
        elif name == "reverse":
            first = start if PATTERN[(step - 1) % len(PATTERN)] != "reverse" else start - STEP
            src_t = first - (t - first)
        elif name == "freeze":
            src_t = start
        elif name == "roll":
            roll_start = loop_start + 24 * STEP
            into = (t - roll_start) / (8 * STEP)     # 0..1 over the roll
            length = STEP * (1.0 if into < 0.5 else 0.5 if into < 0.75 else 0.25)
            src_t = roll_start + (t - roll_start) % length
            k = 1 + 0.35 * into
            snowy = 0.15 + 0.35 * into if (t - roll_start) % length < length * 0.4 else 0.0
            px = 3 + 5 * into
        j = int(round(np.clip(src_t * FPS, 0, i)))   # only the past
        img = grade(frames[j])
        img = zoom(img, k)
        img = split(img, px)
        if snowy > 0:
            img = static(img, rng, snowy)
        img = np.clip(img + flash, 0, 1)
        out.append(img)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    frames = read(src)
    out = render(frames)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIDE}x{SIDE}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dst],
        stdin=subprocess.PIPE)
    for f in out:
        enc.stdin.write((f * 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    if "--sheet" in sys.argv:
        picks = out[:48:2]
        rows = [np.concatenate(picks[r * 6:(r + 1) * 6], 1) for r in range(4)]
        sheet = (np.concatenate(rows, 0) * 255).astype(np.uint8)
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(sys.argv[sys.argv.index("--sheet") + 1], cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", dst, len(out), "frames")


if __name__ == "__main__":
    main()
