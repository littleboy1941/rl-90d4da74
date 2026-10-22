"""PC preview of the `half_skull` effect (reference `docs/ref_halfskull.jpg`
in fff): one half of the face turns into skull - eye socket, nasal cavity,
teeth through the lips - the other half stays; optional photocopy look
(dark ink on cream paper, heavy grain, scanner lines).

    python tools/sim_halfskull.py <photo> <skull.obj> <ao.png> <out.png> --ref ref.jpg
        [--face x,y,w,h] [--eyes lx,ly,rx,ry]

The skull (the 3D model, frontal) is placed so its eye sockets sit on the
eyes; it covers the person's left half (right in the picture) with a soft
seam through the nose.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

from sim_skull import load_obj, render

SIDE = 400


def sockets(shade, cover):
    """Centres of the two eye sockets in the frontal render: darkest blobs
    in the upper half, one on each side."""
    size = shade.shape[0]
    s = np.asarray(Image.fromarray((np.where(cover, shade, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(size * 0.03)), np.float32)
    band = s[int(size * 0.3):int(size * 0.55)]
    half = size // 2
    out = []
    for x0, x1 in ((0, half), (half, size)):
        part = band[:, x0:x1]
        y, x = np.unravel_index(np.argmin(part), part.shape)
        out.append((x + x0, y + int(size * 0.3)))
    return out


def lum(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def half_skull(photo, mesh, ao, face, eyes):
    v, vt, tri, tuv = mesh
    fx, fy, fw, fh = face
    size = 600
    shade, cover = render(v, vt, tri, tuv, ao, size)
    (ax, ay), (bx, by) = sockets(shade, cover)
    lx, ly, rx, ry = eyes
    k = (rx - lx) / (bx - ax)                       # render px -> photo px
    small = int(size * k)
    sh = np.asarray(Image.fromarray((shade * 255).astype(np.uint8)).resize((small, small), Image.LANCZOS), np.float32) / 255
    cv = np.asarray(Image.fromarray((cover * 255).astype(np.uint8)).resize((small, small), Image.LANCZOS), np.float32) / 255
    ox, oy = int(lx - ax * k), int((ly + ry) / 2 - ay * k)
    full_s = np.ones((SIDE, SIDE), np.float32)
    full_c = np.zeros((SIDE, SIDE), np.float32)
    y0, x0 = max(oy, 0), max(ox, 0)
    y1, x1 = min(oy + small, SIDE), min(ox + small, SIDE)
    full_s[y0:y1, x0:x1] = sh[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    full_c[y0:y1, x0:x1] = cv[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    # Half: right of the nose line (the person's left), soft seam ~5% of
    # the face width; only on the face skin (here an ellipse around the
    # face box; on the phone the network's skin classes) - hair, ears and
    # the outline stay.
    mid = (lx + rx) / 2 - fw * 0.07          # seam just left of the nose: the whole nasal hole
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    seam = np.clip((xx - mid) / (fw * 0.05) + 0.5, 0, 1)
    e = ((xx - (fx + fw / 2)) / (fw * 0.5)) ** 2 + ((yy - (fy + fh * 0.55)) / (fh * 0.55)) ** 2
    skin = np.clip((1 - e) / 0.25, 0, 1)
    m = seam * skin * full_c
    img = np.asarray(photo, np.float32) / 255
    # Bone shading laid over the skin: lit bone keeps the skin (a bit
    # paler), cavities go dark - socket, nasal hole, gaps between teeth.
    shade = np.clip(full_s, 0, 1)
    pale = img * 0.75 + 0.25 * lum(img)[..., None]
    bone = pale * (0.03 + 1.15 * shade[..., None] ** 2.2) * 1.1
    return np.clip(img * (1 - m[..., None]) + bone * m[..., None], 0, 1)


def photocopy(img, seed=4):
    """Dark ink on cream paper: grey -> soft -> levels -> grain -> duotone,
    plus faint scanner lines and paper speckle."""
    r = np.random.default_rng(seed)
    g = lum(img)
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    lo, hi = np.percentile(g, [8, 75])
    t = np.clip((g + r.normal(0, 0.09, g.shape) - lo) / (hi - lo), 0, 1)   # 1 = paper, 0 = ink
    h, w = g.shape
    lines = np.ones((h, w), np.float32)
    for y in r.choice(h, 14, replace=False):
        lines[y, :] *= 1 - r.uniform(0.15, 0.4) * (r.random(w) < 0.7)
    for x in r.choice(w, 10, replace=False):
        lines[:, x] *= 1 - r.uniform(0.1, 0.3) * (r.random(h) < 0.6)
    t = t * lines
    ink = np.array([0.24, 0.23, 0.21], np.float32)
    paper = np.array([0.94, 0.9, 0.8], np.float32)
    out = ink + (paper - ink) * t[..., None]
    out *= 1 - 0.06 * r.random((h, w, 1))
    return np.clip(out, 0, 1)


def main():
    src, obj, ao_path, out = sys.argv[1:5]
    face = (130, 80, 155, 150)
    eyes = (173, 114, 243, 114)
    if "--face" in sys.argv:
        face = tuple(float(x) for x in sys.argv[sys.argv.index("--face") + 1].split(","))
    if "--eyes" in sys.argv:
        eyes = tuple(float(x) for x in sys.argv[sys.argv.index("--eyes") + 1].split(","))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    photo = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    ao = np.asarray(Image.open(ao_path).convert("L"), np.float32) / 255
    mixed = half_skull(photo, load_obj(obj), ao, face, eyes)
    ref = Image.open(sys.argv[sys.argv.index("--ref") + 1]).convert("RGB")
    rw, rh = ref.size
    ref = np.asarray(ref.crop((0, (rh - rw) // 2, rw, (rh + rw) // 2)).resize((SIDE, SIDE), Image.LANCZOS), np.float32) / 255
    sheet = np.concatenate([ref, mixed, photocopy(mixed)], axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
