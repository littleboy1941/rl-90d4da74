"""PC preview of the ASCII face idea: the face is drawn with real everyday
Linux command lines. Flat areas show whole readable commands; on the face
contours only the command syntax that matches the edge direction stays
(`/` from paths, `|` from pipes, `-` and `_` from flags and names, `\\`),
so edges are made of fragments of the same commands. Commands that change
or delete things (rm, sudo) are tinted.

    python tools/sim_ascii.py <image> <out.png> [--face x,y,w,h] [--palette robot|lain] [--time t]

The face box is in the 400 px square, screen pixels (top down).
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIDE = 400
CELL_W, CELL_H = 5, 8

# Everyday commands: files, git, builds, packages, media, system status.
COMMANDS = [
    "ls -la ~/projects",
    "cd ~/projects/roundlab && git status",
    "git log --oneline --graph | head -20",
    "git add -A && git commit -m 'fix frame timing'",
    "make -j8 && ./build/roundlab --verbose",
    "find . -name '*.swift' | xargs wc -l | sort -n",
    "grep -rn TODO src/ | less",
    "du -sh /var/log/* | sort -h",
    "df -h /home",
    "tar -czvf backup_2026.tar.gz ~/documents",
    "rsync -av ~/photos/ /mnt/backup/photos/",
    "ffmpeg -i round.mov -vf scale=400:400 round.mp4",
    "sudo apt update && sudo apt upgrade -y",
    "systemctl status nginx --no-pager",
    "journalctl -u nginx --since today | tail -50",
    "ps aux | grep python | grep -v grep",
    "uname -a; uptime; free -h",
    "cat /proc/cpuinfo | grep 'model name' | uniq",
    "chmod +x scripts/deploy.sh",
    "python3 -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements.txt",
    "docker ps --format '{{.Names}}\\t{{.Status}}'",
    "rm -rf build/ && mkdir build",
    "echo $PATH | tr ':' '\\n'",
    "sed -i 's/old_name/new_name/g' config.ini",
    "awk -F, '{print $2}' data.csv | sort | uniq -c",
    "watch -n 1 nvidia-smi",
    "man bash",
]
TINTED = ("rm ", "sudo ")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def edge_glyph(angle):
    """Glyph along an edge; `angle` is the gradient direction (radians)."""
    a = (np.degrees(angle) + 180) % 180          # edge runs perpendicular
    if a < 22.5 or a >= 157.5:
        return "|"
    if a < 67.5:
        return "/"
    if a < 112.5:
        return "-"
    return "\\"


def main():
    src, out = sys.argv[1], sys.argv[2]
    face = (140, 100, 120, 150)
    if "--face" in sys.argv:
        face = tuple(float(v) for v in sys.argv[sys.argv.index("--face") + 1].split(","))
    palette = sys.argv[sys.argv.index("--palette") + 1] if "--palette" in sys.argv else "robot"
    t = float(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 0.0

    img = cv2.imread(src)
    h, w = img.shape[:2]
    s = min(h, w)
    img = cv2.resize(img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    # Local contrast first (CLAHE), so eyes, nose and mouth separate from
    # the evenly lit skin at the coarse character grid.
    gray8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(gray8)
    lum = gray8.astype(np.float32) / 255
    soft = cv2.GaussianBlur(lum, (0, 0), 1.5)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)

    cols, rows = SIDE // CELL_W, SIDE // CELL_H
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2 - fh * 0.1
    rx, ry = fw * 0.8, fh * 0.95                      # head incl. hair

    stream = "   ".join(COMMANDS) + "   "
    tinted = np.zeros(len(stream), bool)
    pos = 0
    for c in COMMANDS:
        if c.startswith(TINTED) or " rm " in c:
            tinted[pos:pos + len(c)] = True
        pos += len(c) + 3

    font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 8)
    canvas = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    scroll = int(t * 6)                               # text drifts left slowly
    for r in range(rows):
        offset = (r * 37 + scroll) % len(stream)
        for c in range(cols):
            x0, y0 = c * CELL_W, r * CELL_H
            px, py = x0 + CELL_W / 2, y0 + CELL_H / 2
            inside = ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2
            if inside > 1.0:
                continue
            fade = 1 - smoothstep(0.75, 1.0, inside)
            cell = soft[y0:y0 + CELL_H, x0:x0 + CELL_W]
            L = float(cell.mean())
            ex = float(gx[y0:y0 + CELL_H, x0:x0 + CELL_W].mean())
            ey = float(gy[y0:y0 + CELL_H, x0:x0 + CELL_W].mean())
            mag = (ex * ex + ey * ey) ** 0.5
            k = (offset + c) % len(stream)
            ch = stream[k]
            edge = mag > 0.45
            if edge:
                ch = edge_glyph(np.arctan2(ey, ex))
            elif ch == " ":
                continue
            # Tone: dark cells drop out, light cells go bright; edges stay visible.
            v = smoothstep(0.3, 0.8, L) ** 1.8
            if edge:
                v = max(v, 0.35)
            v *= fade
            if palette == "lain":
                if L < 0.3 and not edge:                # shadows: dark red characters
                    if (r + c) % 2:
                        continue
                    color = (int(140 * fade), 8, int(20 * fade))
                else:
                    g = int(60 + 195 * v)
                    color = (g, int(g * 0.94), int(g * 0.96))
            else:
                if v < 0.06:
                    continue
                if tinted[k] and not edge:
                    color = (int(230 * max(v, 0.5)), 28, 38)
                else:
                    g = int(255 * v)
                    color = (int(g * 0.86), int(g * 0.95), g)
            draw.text((x0, y0 - 1), ch, font=font, fill=color)
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
