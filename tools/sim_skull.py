"""PC preview of the `skull` effect: the face replaced by a 3D skull drawn
in thin vertical lines (reference `docs/ref_lines.jpg` in fff), the room
around kept as is.

    python tools/sim_skull.py <photo> <skull.obj> <ao.png> <out.png> --ref ref_lines.jpg [--face x,y,w,h]

Software render: orthographic, z-buffer, Lambert from the front-top times
the model's ambient occlusion map; then the shade becomes line width.
Inside the skull outline the paper is white (the face is hidden); thin
drips run down below it over the room. Three head turns side by side.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

SIDE = 400
PERIOD = 2.6
SUPER = 4


def load_obj(path):
    vs, vts, faces = [], [], []
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vs.append([float(p) for p in parts[1:4]])
        elif parts[0] == "vt":
            vts.append([float(p) for p in parts[1:3]])
        elif parts[0] == "f":
            idx = [(int(p.split("/")[0]) - 1, int(p.split("/")[1]) - 1) for p in parts[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    f = np.array(faces)
    return np.array(vs, np.float32), np.array(vts, np.float32), f[:, :, 0], f[:, :, 1]


def vertex_normals(v, tri):
    n = np.zeros_like(v)
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    return rx @ ry


def render(v, vt, tri, tuv, ao, size, yaw=0.0, pitch=0.0):
    """Shade (0 dark .. 1 light) and coverage of the skull in a size x size
    square; the skull's height fills ~0.92 of it."""
    n = vertex_normals(v, tri)
    rot = rotation(yaw, pitch)
    p = v @ rot.T
    nr = n @ rot.T
    centre = (p.max(0) + p.min(0)) / 2
    scale = size * 0.92 / (v[:, 1].max() - v[:, 1].min())
    sx = (p[:, 0] - centre[0]) * scale + size / 2
    sy = size / 2 - (p[:, 1] - centre[1]) * scale
    z = p[:, 2]
    depth = np.full((size, size), -1e9, np.float32)
    shade = np.zeros((size, size), np.float32)
    light = np.array([0.0, 0.45, 1.0], np.float32)
    light /= np.linalg.norm(light)
    ah, aw = ao.shape
    for t in range(len(tri)):
        a, b, c = tri[t]
        xs, ys = sx[[a, b, c]], sy[[a, b, c]]
        x0, x1 = int(max(xs.min(), 0)), int(min(xs.max() + 1, size))
        y0, y1 = int(max(ys.min(), 0)), int(min(ys.max() + 1, size))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-9:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
        region = depth[y0:y1, x0:x1]
        win = inside & (zz > region)
        if not win.any():
            continue
        nn = w0[..., None] * nr[a] + w1[..., None] * nr[b] + w2[..., None] * nr[c]
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-9
        uv = w0[..., None] * vt[tuv[t, 0]] + w1[..., None] * vt[tuv[t, 1]] + w2[..., None] * vt[tuv[t, 2]]
        u = np.clip((uv[..., 0] % 1) * (aw - 1), 0, aw - 1).astype(int)
        vv = np.clip((1 - uv[..., 1] % 1) * (ah - 1), 0, ah - 1).astype(int)
        lam = np.clip(nn @ light, 0, 1)
        s = (0.15 + 0.85 * lam) * ao[vv, u]
        region[win] = zz[win]
        shade[y0:y1, x0:x1][win] = s[win]
    return shade, depth > -1e8


def line_screen(dark, rng):
    h, w = dark.shape
    xs = (np.arange(w * SUPER) + 0.5) / SUPER
    cell = np.floor(xs / PERIOD).astype(int)
    jitter = rng.uniform(-0.3, 0.3, cell.max() + 1) * PERIOD
    centre = (cell + 0.5) * PERIOD + jitter[cell]
    src_x = np.clip(centre.astype(int), 0, w - 1)
    width = dark[:, src_x] * PERIOD * 1.1
    ink = (np.abs(xs - centre)[None, :] < width / 2).astype(np.float32)
    return ink.reshape(h, w, SUPER).mean(-1)


def ref_darkness(ref_path):
    """Darkness values of the reference skull (line screen blurred away),
    sorted: the target distribution for the render's tones."""
    g = np.asarray(Image.open(ref_path).convert("L").crop((90, 150, 380, 560)).filter(ImageFilter.BoxBlur(4)), np.float32) / 255
    d = 1 - g
    return np.sort(d[d > 0.03])


def match(values, target):
    """Histogram matching: rank of each value -> the same rank in `target`."""
    order = np.argsort(values)
    out = np.empty_like(values)
    out[order] = target[(np.arange(len(values)) * (len(target) - 1) / max(len(values) - 1, 1)).astype(int)]
    return out


def skull_frame(photo, mesh, ao, face, yaw, target, seed=3, paper_only=False):
    v, vt, tri, tuv = mesh
    fx, fy, fw, fh = face
    rng = np.random.default_rng(seed)
    size = int(fh * 1.75)
    shade, cover = render(v, vt, tri, tuv, ao, size, yaw=yaw)
    # Tones taken from the reference: the render's darkness ranks are
    # mapped onto the reference skull's darkness distribution.
    dark = np.zeros_like(shade)
    dark[cover] = match(1 - shade[cover], target)
    h, w = SIDE, SIDE
    full = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.float32)
    ox, oy = int(fx + fw / 2 - size / 2), int(fy + fh * 0.5 - size * 0.48)
    ys0, xs0 = max(oy, 0), max(ox, 0)
    ys1, xs1 = min(oy + size, h), min(ox + size, w)
    full[ys0:ys1, xs0:xs1] = dark[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    mask[ys0:ys1, xs0:xs1] = cover[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    # Soft, out of focus, as in the reference.
    full = np.asarray(Image.fromarray((full * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    # Drips: darkness runs down each column with its own length.
    decay = 1 - rng.uniform(0.005, 0.035, w)
    run = np.zeros(w, np.float32)
    out = np.zeros_like(full)
    for y in range(h):
        run = np.maximum(full[y], run * decay)
        out[y] = run
    # Hairline fringe above the skull and long thin tails below.
    has = mask.any(0)
    top = np.where(has, (mask > 0).argmax(0), h)
    bottom = np.where(has, h - 1 - (mask[::-1] > 0).argmax(0), -1)
    up = rng.uniform(0.05, 1, w) ** 1.4 * fh * 0.35
    down = rng.uniform(0.1, 1, w) ** 0.8 * fh * 1.4
    yy = np.arange(h)[:, None]
    t_up = np.clip((yy - (top - up)[None, :]) / (up[None, :] + 1e-3), 0, 1) * (yy < top[None, :])
    t_dn = np.clip(((bottom + down)[None, :] - yy) / (down[None, :] + 1e-3), 0, 1) * (yy > bottom[None, :])
    out = np.maximum(out, rng.uniform(0.1, 0.3, w)[None, :] * np.maximum(t_up, t_dn) * has[None, :])
    ink = line_screen(out, rng)
    # Paper under the skull with a wide soft edge, the room further out.
    paper = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
    room = np.asarray(photo, np.float32) / 255
    if paper_only:
        room = np.ones_like(room)
    base = room * (1 - paper[..., None]) + 0.98 * paper[..., None]
    return np.clip(base * (1 - ink[..., None]) + 0.03 * ink[..., None], 0, 1)


def main():
    src, obj, ao_path, out = sys.argv[1:5]
    face = (130, 80, 155, 150)
    if "--face" in sys.argv:
        face = tuple(float(x) for x in sys.argv[sys.argv.index("--face") + 1].split(","))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = min(w, h)
    photo = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIDE, SIDE), Image.LANCZOS)
    ao = np.asarray(Image.open(ao_path).convert("L"), np.float32) / 255
    mesh = load_obj(obj)
    ref_path = sys.argv[sys.argv.index("--ref") + 1]
    target = ref_darkness(ref_path)
    ref = Image.open(ref_path).convert("RGB")
    canvas = Image.new("RGB", (ref.height, ref.height), (255, 255, 255))
    canvas.paste(ref, ((ref.height - ref.width) // 2, 0))
    tiles = [np.asarray(canvas.resize((SIDE, SIDE), Image.LANCZOS), np.float32) / 255,
             skull_frame(photo, mesh, ao, face, 0.0, target, paper_only=True),
             skull_frame(photo, mesh, ao, face, 0.0, target),
             skull_frame(photo, mesh, ao, face, 0.5, target)]
    sheet = np.concatenate(tiles, axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
