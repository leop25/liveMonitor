import json, math, sys, os, subprocess, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
TL = json.load(open("timeline.json", encoding="utf-8"))
SHOTS, ITEMS, EVENTS, TOTAL = TL["shots"], TL["items"], TL["events"], TL["total"]
FD = "../fonts/"
FONT = {
    "J": (FD + "CG-600.ttf", 66, (246, 232, 204)),
    "P": (FD + "CG-600.ttf", 66, (186, 198, 214)),
    "N": (FD + "CG-Italic500.ttf", 62, (222, 220, 214)),
    "Q": (FD + "CG-Italic500.ttf", 64, (236, 214, 176)),
}


def font(path, size):
    return _font(path, size)


@functools.lru_cache(None)
def _font(path, size):
    return ImageFont.truetype(path, size)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ss(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def lerp(a, b, u):
    return a + (b - a) * u


def rgba(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), int(a))


# ----------------------------------------------------------------- helpers
def vgrad(w, h, stops):
    """vertical gradient; stops = [(pos, (r,g,b)), ...]"""
    ys = np.linspace(0, 1, h)[:, None]
    out = np.zeros((h, 1, 3), np.float32)
    ps = [s[0] for s in stops]
    for c in range(3):
        out[:, 0, c] = np.interp(ys[:, 0], ps, [s[1][c] for s in stops])
    return np.repeat(out, w, axis=1)


def radial(w, h, cx, cy, rx, ry, power=2.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    return np.clip(1 - d, 0, 1) ** power


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(float(r)))


class Shape:
    """collection of coloured primitives in local units, rendered supersampled"""

    def __init__(self):
        self.ops = []

    def poly(self, pts, col, a=255):
        self.ops.append(("p", pts, rgba(col, a)))
        return self

    def curve(self, pts, col, a=255, n=10):
        out = []
        m = len(pts)
        for i in range(m):
            p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
            for k in range(n):
                t = k / n
                t2, t3 = t * t, t * t * t
                out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
        self.ops.append(("p", out, rgba(col, a)))
        return self

    def ell(self, cx, cy, rx, ry, col, a=255):
        self.ops.append(("e", (cx - rx, cy - ry, cx + rx, cy + ry), rgba(col, a)))
        return self

    def line(self, pts, col, width, a=255):
        self.ops.append(("l", pts, rgba(col, a), width))
        return self

    def render(self, size, tf, ssf=2):
        """tf maps local (x,y) -> pixel (x,y) in output size"""
        w, h = size
        im = Image.new("RGBA", (w * ssf, h * ssf), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for op in self.ops:
            if op[0] == "p":
                d.polygon([tuple(v * ssf for v in tf(*p)) for p in op[1]], fill=op[2])
            elif op[0] == "e":
                x0, y0 = tf(op[1][0], op[1][1])
                x1, y1 = tf(op[1][2], op[1][3])
                d.ellipse([min(x0, x1) * ssf, min(y0, y1) * ssf, max(x0, x1) * ssf, max(y0, y1) * ssf], fill=op[2])
            elif op[0] == "l":
                d.line([tuple(v * ssf for v in tf(*p)) for p in op[1]], fill=op[2], width=int(op[3] * ssf), joint="curve")
        return im.resize((w, h), Image.LANCZOS)


def light_sprite(im, top=1.25, bottom=0.55, rim_col=(255, 225, 170), rim_dirs=((0, 1),), rim_w=3, rim_gain=1.0, rim_blur=2.0):
    """apply vertical key light + rim light to an RGBA sprite"""
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    g = np.linspace(top, bottom, h)[:, None, None]
    rgb = a[..., :3] * g
    m = a[..., 3] / 255.0
    rim = np.zeros((h, w), np.float32)
    for dx, dy in rim_dirs:
        sh = np.zeros_like(m)
        ox, oy = int(dx * rim_w), int(dy * rim_w)
        ys0, ys1 = max(0, oy), h + min(0, oy)
        xs0, xs1 = max(0, ox), w + min(0, ox)
        sh[ys0:ys1, xs0:xs1] = m[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        rim = np.maximum(rim, np.clip(m - sh, 0, 1))
    rimi = Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rim_blur))
    rim = np.asarray(rimi).astype(np.float32) / 255.0 * m * rim_gain
    rgb = rgb + rim[..., None] * np.array(rim_col, np.float32)[None, None, :]
    out = np.dstack([np.clip(rgb, 0, 255), a[..., 3]]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def add_glow(base, glow_rgb_arr):
    """additive light (float array HxWx3) onto RGB image"""
    a = np.asarray(base).astype(np.float32) + glow_rgb_arr
    return to_img(a)


def paste(dst, src, x, y):
    """alpha composite src onto dst at integer position (clipped)"""
    x, y = int(round(x)), int(round(y))
    sw, sh = src.size
    dw, dh = dst.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + sw), min(dh, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    crop = src.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    dst.alpha_composite(crop, (x0, y0))


def with_alpha(im, a):
    if a >= 0.999:
        return im
    im = im.copy()
    al = im.getchannel("A").point(lambda v: int(v * a))
    im.putalpha(al)
    return im


def camera(canvas, cx, cy, s):
    """crop canvas around (cx,cy); s = output px per canvas px"""
    s = max(s, W / canvas.size[0], H / canvas.size[1])
    cw, ch = W / s, H / s
    x0, y0 = cx - cw / 2, cy - ch / 2
    x0 = clamp(x0, 0, canvas.size[0] - cw)
    y0 = clamp(y0, 0, canvas.size[1] - ch)
    return canvas.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))


# ------------------------------------------------------------ dust
RNG = np.random.default_rng(3)
DUST = RNG.random((140, 6))


@functools.lru_cache(None)
def dot(r, col):
    s = int(r * 6 + 4)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], fill=rgba(col, 255))
    return blur(im, r * 0.8 + 0.6)


def draw_dust(img, t, col=(255, 230, 190), n=90, region=(0, 0, W, H), amount=1.0, drift=(4, -9)):
    x0, y0, x1, y1 = region
    for i in range(n):
        p = DUST[i]
        sp = 0.4 + p[2]
        x = x0 + ((p[0] * (x1 - x0) + t * drift[0] * sp + 30 * math.sin(t * 0.3 * sp + p[3] * 6)) % (x1 - x0))
        y = y0 + ((p[1] * (y1 - y0) + t * drift[1] * sp) % (y1 - y0))
        tw = 0.5 + 0.5 * math.sin(t * (0.6 + p[4]) + p[5] * 6.28)
        a = amount * (0.15 + 0.6 * tw) * (0.4 + p[4] * 0.6)
        r = 1.0 + p[2] * 2.2
        paste(img, with_alpha(dot(round(float(r), 1), col), a), x - r * 3 - 2, y - r * 3 - 2)


# ------------------------------------------------------------ grain & vignette
GRAIN = [np.random.default_rng(i).normal(0, 1, (H // 2, W // 2)).astype(np.float32) for i in range(8)]
_vy, _vx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip(1.15 - 0.55 * (((_vx - W / 2) / (W * 0.62)) ** 2 + ((_vy - H * 0.47) / (H * 0.6)) ** 2), 0.25, 1.0)[..., None]
BOTTOM = np.clip((np.linspace(0, 1, H) - 0.6) / 0.35, 0, 1) ** 1.3
BOTTOM = (1 - 0.55 * BOTTOM)[:, None, None].astype(np.float32)


def finish(img, t):
    a = np.asarray(img).astype(np.float32)
    g = GRAIN[int(t * FPS) % 8]
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    a = a * VIG * BOTTOM + g * 5.0
    return to_img(a)


# =================================================================== FIGURES
SUIT = (27, 29, 35)
HAIR_G = (58, 57, 56)
SKIN_D = (88, 70, 58)
ROBE = (206, 196, 176)
MANTLE = (104, 44, 38)
HAIR_B = (52, 38, 28)
BEARD = (46, 33, 25)
SKIN_J = (150, 116, 90)


def man_back(pose):
    s = Shape()
    dy = 0.0
    if pose == "kneel":
        dy = -0.42
        s.poly([(-0.21, 0.47), (0.21, 0.47), (0.23, 0.2), (0.2, 0.03), (-0.2, 0.03), (-0.23, 0.2)], SUIT)
        s.ell(-0.11, 0.07, 0.075, 0.07, (20, 18, 17)).ell(0.11, 0.07, 0.075, 0.07, (20, 18, 17))
    else:
        for sg in (-1, 1):
            s.poly([(sg * 0.19, 0.95), (sg * 0.015, 0.95), (sg * 0.03, 0.04), (sg * 0.15, 0.04)], SUIT)
            s.ell(sg * 0.09, 0.035, 0.075, 0.035, (14, 13, 12))
    Y = lambda y: y + dy
    s.poly([(-0.07, Y(1.53)), (0.07, Y(1.53)), (0.21, Y(1.485)), (0.25, Y(1.43)), (0.262, Y(1.25)), (0.245, Y(1.0)), (0.225, Y(0.86)),
            (-0.225, Y(0.86)), (-0.245, Y(1.0)), (-0.262, Y(1.25)), (-0.25, Y(1.43)), (-0.21, Y(1.485))], SUIT)
    for sg in (-1, 1):
        s.poly([(sg * 0.25, Y(1.43)), (sg * 0.3, Y(1.2)), (sg * 0.3, Y(0.93)), (sg * 0.245, Y(0.9)), (sg * 0.215, Y(1.15)), (sg * 0.2, Y(1.38))], (24, 26, 31))
        s.ell(sg * 0.272, Y(0.87), 0.04, 0.058, (60, 48, 42))
    hy = 1.69 if pose == "stand" else 1.6
    s.poly([(-0.06, Y(1.5)), (0.06, Y(1.5)), (0.062, Y(hy - 0.05)), (-0.062, Y(hy - 0.05))], (45, 38, 34))
    s.ell(-0.094, Y(hy - 0.01), 0.02, 0.035, SKIN_D).ell(0.094, Y(hy - 0.01), 0.02, 0.035, SKIN_D)
    s.ell(0, Y(hy), 0.096, 0.115 if pose == "stand" else 0.1, HAIR_G)
    return s


def jesus_front(pose, point=0.0):
    """pose: seated | stand"""
    s = Shape()
    if pose == "seated":
        b = -0.44
    else:
        b = 0.0
        s.poly([(-0.27, 1.1), (0.27, 1.1), (0.31, 0.5), (0.35, 0.02), (-0.35, 0.02), (-0.31, 0.5)], ROBE)
        s.ell(-0.12, 0.03, 0.07, 0.03, (70, 55, 40)).ell(0.12, 0.03, 0.07, 0.03, (70, 55, 40))
        s.poly([(-0.26, 1.2), (-0.1, 1.05), (0.34, 0.05), (0.12, 0.02), (-0.3, 0.9)], MANTLE)
    Y = lambda y: y + b
    s.poly([(-0.085, Y(1.62)), (0.085, Y(1.62)), (0.22, Y(1.55)), (0.265, Y(1.45)), (0.275, Y(1.1)), (0.29, Y(0.98)),
            (-0.29, Y(0.98)), (-0.275, Y(1.1)), (-0.265, Y(1.45)), (-0.22, Y(1.55))], ROBE)
    s.poly([(-0.25, Y(1.5)), (-0.1, Y(1.55)), (0.12, Y(1.2)), (0.2, Y(0.98)), (-0.05, Y(0.98)), (-0.27, Y(1.2))], MANTLE)
    if point > 0:
        ang = math.radians(lerp(-70, 18, point))
        L = 0.62
        ex, ey = -0.22 - L * math.cos(ang), Y(1.45) + L * math.sin(ang)
        nx, ny = -math.sin(ang) * 0.055, math.cos(ang) * 0.055
        s.poly([(-0.22 + nx, Y(1.45) + ny), (ex + nx * 0.7, ey + ny * 0.7), (ex - nx * 0.7, ey - ny * 0.7), (-0.22 - nx, Y(1.45) - ny)], ROBE)
        s.ell(ex - 0.03 * math.cos(ang), ey + 0.03 * math.sin(ang), 0.045, 0.04, SKIN_J)
    # hair behind, face, beard
    s.poly([(-0.125, Y(1.8)), (0.125, Y(1.8)), (0.15, Y(1.55)), (0.1, Y(1.5)), (-0.1, Y(1.5)), (-0.15, Y(1.55))], HAIR_B)
    s.ell(0, Y(1.745), 0.078, 0.1, SKIN_J)
    s.poly([(-0.075, Y(1.73)), (0.075, Y(1.73)), (0.06, Y(1.63)), (0, Y(1.6)), (-0.06, Y(1.63))], BEARD)
    s.ell(0, Y(1.81), 0.105, 0.065, HAIR_B)
    return s


@functools.lru_cache(None)
def sprite_m(kind, pose, pxm, point=0.0):
    """figure sprite at pxm pixels per metre; anchor = feet centre at (w/2, h-pad)"""
    shape = man_back(pose) if kind == "man" else jesus_front(pose, point)
    w = int(1.7 * pxm) + 8
    hgt = int(2.0 * pxm) + 8
    tf = lambda x, y: (w / 2 + x * pxm, hgt - 4 - y * pxm)
    im = shape.render((w, hgt), tf)
    if kind == "man":
        im = light_sprite(im, 0.95, 0.5, (255, 214, 160), ((0, 1), (1, 0), (-1, 0)), max(2, pxm / 160), 0.85, max(1, pxm / 250))
    else:
        im = light_sprite(im, 1.35, 0.7, (255, 238, 200), ((0, 1),), max(2, pxm / 120), 1.0, max(1, pxm / 250))
    return im


# ===================================================================== HALL
CW, CH = 1350, 2400
F = 2000.0
VPY = 960.0
CX = CW / 2
EYE = 1.6


def P(x, y, z):
    return CX + F * x / z, VPY - F * (y - EYE) / z


@functools.lru_cache(None)
def hall_bg():
    a = vgrad(CW, CH, [(0, (2, 2, 3)), (0.3, (8, 9, 12)), (0.4, (18, 20, 25)), (0.46, (14, 15, 19)), (0.7, (9, 9, 11)), (1, (5, 5, 6))])
    img = to_img(a).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    zw = 24.0
    # back wall
    x0, yt = P(-2.6, 11, zw)
    x1, yb = P(2.6, 0, zw)
    wall = vgrad(int(x1 - x0), int(yb - yt), [(0, (6, 6, 8)), (0.6, (20, 21, 26)), (1, (26, 26, 30))])
    img.paste(to_img(wall), (int(x0), int(yt)))
    # floor perspective lines
    for xi in np.arange(-6, 6.01, 0.8):
        a0 = P(xi, 0, zw)
        a1 = P(xi, 0, 1.2)
        d.line([a0, a1], fill=(40, 42, 50, 38), width=2)
    for zi in [4, 5, 6, 7.2, 8.6, 10.2, 12, 14, 16.5, 19, 21.5]:
        l0 = P(-8, 0, zi)
        l1 = P(8, 0, zi)
        d.line([l0, l1], fill=(40, 42, 50, 30), width=2)
    # columns far -> near
    for zc in [21, 18, 15, 12, 9, 6, 3.6]:
        for sg in (-1, 1):
            xa, xb = sg * 2.6, sg * 3.5
            pa = P(xa, 14, zc)
            pb = P(xb, 0, zc)
            pa2 = P(xa, 14, zc + 0.9)
            pb2 = P(xa, 0, zc + 0.9)
            shade = int(lerp(22, 10, clamp((21 - zc) / 17)))
            # inner side face (toward centre)
            d.polygon([(pa[0], pa[1]), (pa2[0], pa2[1]), (pb2[0], pb2[1]), (pa[0], P(xa, 0, zc)[1])], fill=(shade + 6, shade + 6, shade + 9, 255))
            lx, rx = sorted([pa[0], pb[0]])
            d.rectangle([lx, pa[1], rx, pb[1]], fill=(shade, shade, shade + 3, 255))
            # plinth
            pl = P(sg * 2.5, 0.35, zc - 0.05)
            d.rectangle([min(pl[0], pb[0]) - 3, pl[1], max(pl[0], pb[0]) + 3, pb[1]], fill=(shade + 4, shade + 4, shade + 7, 255))
            # edge highlight on inner edge
            ex = pa[0]
            d.line([(ex, pa[1]), (ex, P(xa, 0, zc)[1])], fill=(120, 100, 75, int(lerp(70, 25, clamp((21 - zc) / 17)))), width=max(2, int(F / zc / 250)))
    img = img.convert("RGB")
    # light shaft (additive)
    glow = Image.new("L", (CW, CH), 0)
    gd = ImageDraw.Draw(glow)
    top = P(0, 16, 9.4)
    fl = P(0, 0, 9.4)
    gd.polygon([(top[0] - 60, -200), (top[0] + 60, -200), (fl[0] + 420, fl[1] + 20), (fl[0] - 420, fl[1] + 20)], fill=150)
    glow = blur(glow, 70)
    g = np.asarray(glow).astype(np.float32)[..., None] / 255 * np.array([95, 80, 58], np.float32)
    pool = radial(CW, CH, fl[0], fl[1] + 10, 700, 170, 1.6)[..., None] * np.array([110, 88, 60], np.float32)
    halo = radial(CW, CH, CX, P(0, 1.2, 9.6)[1], 420, 380, 2.2)[..., None] * np.array([60, 48, 32], np.float32)
    horizon = radial(CW, CH, CX, VPY + 120, 900, 260, 1.5)[..., None] * np.array([10, 12, 18], np.float32)
    img = add_glow(img, g + pool + halo + horizon)
    return img.convert("RGBA")


@functools.lru_cache(None)
def table_sprite():
    im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fl, fr = P(-1.25, 0.78, 9.0), P(1.25, 0.78, 9.0)
    bl, br = P(-1.25, 0.78, 9.8), P(1.25, 0.78, 9.8)
    d.polygon([bl, br, fr, fl], fill=(98, 76, 52, 255))
    d.line([fl, fr], fill=(170, 135, 92, 255), width=3)
    f0 = P(-1.25, 0, 9.0)
    f1 = P(1.25, 0, 9.0)
    body = vgrad(int(fr[0] - fl[0]), int(f0[1] - fl[1]), [(0, (34, 25, 18)), (1, (10, 8, 7))])
    im.paste(to_img(body).convert("RGBA"), (int(fl[0]), int(fl[1]) + 2))
    # panel lines
    for k in range(1, 4):
        x = lerp(fl[0], fr[0], k / 4)
        d.line([(x, fl[1] + 14), (x, f0[1] - 6)], fill=(0, 0, 0, 90), width=2)
    # book on the table
    b0, b1 = P(-0.25, 0.8, 9.25), P(0.25, 0.8, 9.55)
    d.polygon([P(-0.25, 0.8, 9.55), P(0.25, 0.8, 9.55), P(0.26, 0.8, 9.25), P(-0.26, 0.8, 9.25)], fill=(215, 200, 170, 255))
    d.line([P(0, 0.8, 9.55), P(0, 0.8, 9.25)], fill=(120, 100, 80, 255), width=2)
    return im


DOOR_X0, DOOR_X1, DOOR_H = -1.95, -0.75, 3.9


def door_layer(appear, open_amt, light):
    """door on back wall; appear 0..1 (outline drawing), open 0..1"""
    im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    if appear <= 0:
        return im
    d = ImageDraw.Draw(im)
    z = 23.95
    a0, a1 = P(DOOR_X0, DOOR_H, z), P(DOOR_X1, 0, z)
    x0, y0, x1, y1 = a0[0], a0[1], a1[0], a1[1]
    # interior
    inner = int(255 * ss(appear * 1.5 - 0.5))
    d.rectangle([x0, y0, x1, y1], fill=(10, 10, 12, inner))
    if open_amt > 0:
        wv = (x1 - x0) / 2 * (1 - open_amt)
        d.rectangle([x0 + wv, y0, x1 - wv, y1], fill=(0, 0, 0, inner))
        d.line([(x0 + wv, y0), (x0 + wv, y1)], fill=(60, 64, 72, inner), width=2)
        d.line([(x1 - wv, y0), (x1 - wv, y1)], fill=(60, 64, 72, inner), width=2)
    else:
        d.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=(4, 4, 5, inner), width=2)
    # outline drawn progressively
    per = [(x0, y1), (x0, y0), (x1, y0), (x1, y1)]
    lens = [y1 - y0, x1 - x0, y1 - y0]
    tot = sum(lens)
    rem = appear * tot * 1.2
    pts = [per[0]]
    for i in range(3):
        if rem <= 0:
            break
        f = clamp(rem / lens[i])
        pts.append((lerp(per[i][0], per[i + 1][0], f), lerp(per[i][1], per[i + 1][1], f)))
        rem -= lens[i]
    line = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(line)
    ld.line(pts, fill=(170, 185, 210, int(200 * light)), width=3)
    im.alpha_composite(blur(line, 6))
    im.alpha_composite(line)
    return im


JPOS = {"seat": (0.0, 9.6), "stand": (1.0, 7.3), "near": (0.5, 5.9)}
MANPOS = (-0.6, 4.5)


def path_walk(u):
    pts = [(0.0, 9.6), (1.6, 9.6), (1.7, 8.3), JPOS["stand"]]
    u = ease(u) * (len(pts) - 1)
    i = min(int(u), len(pts) - 2)
    f = u - i
    return lerp(pts[i][0], pts[i + 1][0], f), lerp(pts[i][1], pts[i + 1][1], f)


def render_hall(sh, t):
    p = sh["params"]
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    canvas = hall_bg().copy()
    light = 1.0
    cam = p.get("cam", "")
    if cam in ("final", "end"):
        light = lerp(1.0, 0.72, ease(u)) if cam == "final" else 0.72
    # door
    door = p.get("door")
    if door:
        appear = ss(lt / 2.5) if door == "appear" else 1.0
        op = {"on": 0.0, "appear": 0.0, "open": 1.0, "closing": 1 - ease((lt - 1.0) / 5.0)}[door]
        canvas.alpha_composite(door_layer(appear, op, 1.0))
    # jesus
    jmode = p.get("jesus", "seat")
    jpose, jx, jz, point, bob = "seated", 0.0, 9.6, 0.0, 0.0
    if jmode == "walk":
        jx, jz = path_walk(clamp((lt - 0.3) / (dur * 0.85)))
        jpose = "stand"
        bob = abs(math.sin(lt * 3.2)) * 0.012 * (1 - ss((lt - dur * 0.85) / 0.5))
        if lt < 0.6:
            jpose = "seated"
    elif jmode == "stand":
        jx, jz = JPOS["stand"]
        jpose = "stand"
    elif jmode == "return":
        uu = clamp((lt - 0.2) / (dur * 0.8))
        jx, jz = path_walk(1 - uu)
        jpose = "stand" if uu < 0.97 else "seated"
        bob = abs(math.sin(lt * 3.2)) * 0.012 * (uu < 0.97)
    elif jmode == "approach":
        uu = ease(clamp((lt - 0.1) / (dur * 0.8)))
        jx = lerp(JPOS["stand"][0], JPOS["near"][0], uu)
        jz = lerp(JPOS["stand"][1], JPOS["near"][1], uu)
        jpose = "stand"
        bob = abs(math.sin(lt * 3.2)) * 0.012 * (uu < 0.98)
    elif jmode in ("near", "point"):
        jx, jz = JPOS["near"]
        jpose = "stand"
        if jmode == "point":
            point = ease(clamp((lt - 0.2) / 1.2))
            if lt > dur - 1.5:
                point *= 1.0
    pxm = F / jz
    pq = round(point * 8) / 8
    spr = sprite_m("jesus", jpose, 360 if jpose == "stand" else 220, pq)
    base_pxm = 360 if jpose == "stand" else 220
    k = pxm / base_pxm
    spr = spr.resize((max(1, int(spr.size[0] * k)), max(1, int(spr.size[1] * k))), Image.LANCZOS)
    fx, fy = P(jx, bob, jz)
    jimg = (spr, fx - spr.size[0] / 2, fy - spr.size[1] + 4 * k)
    behind_table = jz > 9.05
    if behind_table:
        paste(canvas, *jimg)
    canvas.alpha_composite(table_sprite())
    if not behind_table:
        # soft floor shadow
        paste(canvas, *jimg)
    # man
    mpose = p.get("man", "stand")
    if mpose != "gone":
        spr_m = sprite_m("man", "stand" if mpose == "stand" else mpose, 444)
        if mpose == "kneel" and lt < 0.9 and p.get("jesus") == "near" and sh.get("_kneel_anim"):
            pass
        mx, my = P(MANPOS[0], 0, MANPOS[1])
        breath = math.sin(t * 1.3) * 1.5
        if mpose == "kneel" and "caiu" in sh.get("_first_text", ""):
            ksp = sprite_m("man", "stand", 444)
            f = ss((lt - 0.4) / 0.9)
            paste(canvas, with_alpha(ksp, 1 - f), mx - ksp.size[0] / 2, my - ksp.size[1] + 4 - f * 40)
            paste(canvas, with_alpha(spr_m, f), mx - spr_m.size[0] / 2, my - spr_m.size[1] + 4 + (1 - f) * -30)
        elif mpose == "bow" and "abaixou" in sh.get("_first_text", ""):
            ksp = sprite_m("man", "stand", 444)
            f = ss((lt - 0.3) / 1.0)
            paste(canvas, with_alpha(ksp, 1 - f), mx - ksp.size[0] / 2, my - ksp.size[1] + 4)
            paste(canvas, with_alpha(spr_m, f), mx - spr_m.size[0] / 2, my - spr_m.size[1] + 4)
        else:
            paste(canvas, spr_m, mx - spr_m.size[0] / 2, my - spr_m.size[1] + 4 + breath)
    # dust in shaft
    fl = P(0, 0, 9.4)
    draw_dust(canvas, t, (255, 228, 185), 110, (fl[0] - 380, 0, fl[0] + 380, fl[1]), 0.9, (3, -7))
    # camera
    if cam == "reveal":
        z = lerp(1.62, 1.22, ease(u))
        cy = lerp(1080, 1180, ease(u))
        cx = lerp(CX + 20, CX, ease(u))
    elif cam == "high":
        z = lerp(1.05, 1.16, ease(u))
        cx, cy = CX, lerp(1060, 1160, ease(u))
    elif cam == "door":
        dz = P((DOOR_X0 + DOOR_X1) / 2, 2, 24)
        z = lerp(1.25, 1.75, ease(u))
        cx, cy = lerp(CX, dz[0], ease(u)), lerp(1200, dz[1] + 120, ease(u))
    elif cam == "verdict":
        z = lerp(1.3, 1.2, ease(u))
        cx, cy = CX + 30, 1160
    elif cam in ("wide", "final"):
        z = lerp(1.2, 1.08, ease(u))
        cx, cy = CX, 1130
    elif cam == "end":
        z = lerp(1.08, 1.0, ease(u))
        cx, cy = CX, 1150
    else:
        z = lerp(1.25, 1.3, ease(u))
        cx, cy = CX - 10, 1170
    out = camera(canvas, cx, cy, z)
    if light < 1:
        out = Image.eval(out, lambda v: int(v * light))
    return out.convert("RGB")


# ===================================================================== CLOSE-UPS
KC = 1.15
CCW, CCH = int(W * KC), int(H * KC)


def T(x, y):
    """close canvas transform from 1080-space"""
    return x * KC + (CCW - W * KC) / 2, y * KC


@functools.lru_cache(None)
def jclose_bg(mood):
    a = vgrad(CCW, CCH, [(0, (10, 8, 6)), (0.35, (22, 17, 12)), (0.7, (10, 8, 7)), (1, (4, 3, 3))])
    warm = np.array([150, 112, 70], np.float32) if mood != "cold" else np.array([95, 92, 90], np.float32)
    beam = np.zeros((CCH, CCW), np.float32)
    bm = Image.new("L", (CCW, CCH), 0)
    ImageDraw.Draw(bm).polygon([T(400, -100), T(680, -100), T(900, 1300), T(180, 1300)], fill=120)
    beam = np.asarray(blur(bm, 80)).astype(np.float32) / 255
    glow = radial(CCW, CCH, *T(540, 560), 620 * KC, 680 * KC, 1.7)
    a = a + (glow * 0.9 + beam * 0.5)[..., None] * warm
    # distant columns
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for x in (70, 170, 910, 1010):
        d.rectangle([T(x, 0), T(x + 55, 1920)], fill=(0, 0, 0, 70))
    return blur(img, 3).convert("RGBA")


def jesus_bust_shapes():
    body = Shape()
    body.poly([(540, 930), (700, 960), (905, 1080), (1020, 1250), (1100, 1920), (-20, 1920), (60, 1250), (175, 1080), (380, 960)], ROBE)
    body.poly([(470, 960), (540, 1110), (610, 960)], (150, 138, 118))
    body.poly([(175, 1080), (330, 1000), (420, 1060), (520, 1250), (820, 1920), (400, 1920), (130, 1450), (60, 1250)], MANTLE)
    body.line([(705, 1150), (740, 1450), (760, 1920)], (150, 140, 122), 10)
    body.line([(850, 1200), (905, 1500), (930, 1920)], (160, 150, 130), 8)
    body.line([(300, 1300), (380, 1600), (430, 1920)], (70, 30, 26), 10)
    head = Shape()
    head.curve([(540, 545), (650, 570), (705, 650), (715, 770), (730, 900), (765, 1060), (690, 1095), (625, 980), (455, 980), (390, 1095), (315, 1060), (350, 900), (365, 770), (375, 650), (430, 570)], HAIR_B)
    head.curve([(540, 640), (608, 662), (628, 740), (622, 822), (598, 892), (540, 930), (482, 892), (458, 822), (452, 740), (472, 662)], SKIN_J)
    head.curve([(458, 826), (490, 850), (516, 846), (540, 856), (564, 846), (590, 850), (622, 826), (632, 885), (612, 955), (576, 1005), (540, 1022), (504, 1005), (468, 955), (448, 885)], BEARD)
    for sg in (-1, 1):
        head.curve([(540 + sg * 4, 588), (540 + sg * 70, 622), (540 + sg * 98, 700), (540 + sg * 104, 800), (540 + sg * 116, 900), (540 + sg * 138, 985), (540 + sg * 96, 955), (540 + sg * 76, 860), (540 + sg * 66, 760), (540 + sg * 46, 680), (540 + sg * 10, 640)], (60, 44, 32))
    return body, head


def face_model(hd, face_pts, sockets, ridge, lit, lit_col, dark=0.75):
    """sculpt a featureless face: shadowed sockets, lit ridge / cheek"""
    fm = Shape().curve(face_pts, (255, 255, 255)).render((CCW, CCH), T)
    fmask = np.asarray(blur(fm.getchannel("A"), 6)).astype(np.float32) / 255
    a = np.asarray(hd).astype(np.float32)
    mul = np.ones((CCH, CCW), np.float32) * dark
    for (x, y, rx, ry) in sockets:
        mul -= 0.35 * radial(CCW, CCH, *T(x, y), rx * KC, ry * KC, 1.3)
    add = np.zeros((CCH, CCW), np.float32)
    x, y, rx, ry = ridge
    add += radial(CCW, CCH, *T(x, y), rx * KC, ry * KC, 1.6)
    x, y, rx, ry = lit
    add += 0.8 * radial(CCW, CCH, *T(x, y), rx * KC, ry * KC, 1.4)
    m = fmask
    a[..., :3] = a[..., :3] * (1 - m[..., None] * (1 - np.clip(mul, 0.2, 1.5)[..., None])) + (add * m)[..., None] * np.array(lit_col, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


@functools.lru_cache(None)
def jclose_layers():
    body, head = jesus_bust_shapes()
    b = body.render((CCW, CCH), T)
    hd = head.render((CCW, CCH), T)
    b = light_sprite(b, 1.25, 0.45, (255, 232, 190), ((0, 1), (1, 1), (-1, 1)), 7, 0.9, 5)
    hd = face_model(hd, [(540, 640), (608, 662), (628, 740), (622, 822), (598, 892), (540, 930), (482, 892), (458, 822), (452, 740), (472, 662)],
                    [(508, 772, 42, 26), (572, 772, 42, 26)], (540, 760, 16, 70), (540, 690, 70, 40), (70, 52, 36), 0.8)
    hd = light_sprite(hd, 1.15, 0.55, (255, 236, 200), ((0, 1), (1, 0), (-1, 0)), 7, 1.0, 5)
    return blur(b, 1.2), blur(hd, 1.0)


@functools.lru_cache(None)
def hand_sprite():
    s = Shape()
    s.ell(0, 0, 70, 85, SKIN_J)
    for i, (dx, ln) in enumerate(((-48, 120), (-16, 140), (16, 135), (46, 110))):
        s.poly([(dx - 14, -40), (dx + 14, -40), (dx + 12, -40 - ln), (dx - 12, -40 - ln)], SKIN_J)
        s.ell(dx, -40 - ln, 12, 12, SKIN_J)
    s.poly([(-60, 10), (-80, -30), (-110, -80), (-95, -95), (-55, -40)], SKIN_J)
    s.poly([(-75, 40), (75, 40), (95, 330), (-95, 330)], ROBE)
    im = s.render((300, 560), lambda x, y: (150 + x, 220 + y))
    return light_sprite(im, 1.1, 0.5, (255, 232, 190), ((0, 1), (1, 0)), 5, 0.9, 4)


def render_jclose(sh, t):
    p = sh["params"]
    mode = p.get("mode", "")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    mood = "cold" if mode in ("sad", "eyes") else "warm"
    canvas = jclose_bg(mood).copy()
    body, head = jclose_layers()
    breath = math.sin(t * 1.1) * 3
    hy = breath
    if mode == "look":
        hy += -14 * ease(lt / 1.0) + 6
    if mode == "above":
        pass
    canvas.alpha_composite(body, (0, int(breath)))
    canvas.alpha_composite(head, (0, int(hy)))
    if mode == "hand":
        hs = hand_sprite()
        f = ease((lt - 0.2) / 0.9)
        paste(canvas, hs, T(700, 0)[0], T(0, lerp(1950, 1130, f))[1])
    draw_dust(canvas, t, (255, 226, 180), 70, (0, 0, CCW, int(CCH * 0.75)), 0.8, (2, -6))
    # camera
    s0, s1 = 0.9, 0.96
    cy0, cy1 = 1000, 980
    if mode == "lean":
        s0, s1, cy0, cy1 = 0.93, 1.08, 980, 900
    elif mode == "slow":
        s0, s1, cy0, cy1 = 0.9, 1.04, 1000, 940
    elif mode == "above":
        s0, s1, cy0, cy1 = 0.88, 0.92, 1080, 1060
    elif mode in ("sad", "eyes"):
        s0, s1, cy0, cy1 = 0.95, 1.0, 960, 930
    elif mode == "look":
        s0, s1 = 0.92, 0.97
    e = ease(u)
    out = camera(canvas, CCW / 2 + math.sin(t * 0.21) * 6, lerp(cy0, cy1, e) * KC, lerp(s0, s1, e) / KC * KC)
    if mode == "above":
        out = out.transform(out.size, Image.AFFINE, (1, 0, 0, 0, 1, -40), Image.BICUBIC)
    if mood == "cold":
        a = np.asarray(out).astype(np.float32)
        gray = a[..., :3].mean(axis=2, keepdims=True)
        a = (a[..., :3] * 0.55 + gray * 0.45) * np.array([0.82, 0.84, 0.9])
        out = to_img(a)
    return out.convert("RGB")


@functools.lru_cache(None)
def pclose_bg(pale):
    a = vgrad(CCW, CCH, [(0, (6, 7, 10)), (0.35, (16, 19, 26)), (0.65, (9, 10, 13)), (1, (3, 3, 4))])
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for x, w_, al in ((80, 130, 60), (330, 60, 40), (760, 70, 45), (960, 150, 70)):
        d.rectangle([T(x, 0), T(x + w_, 1920)], fill=(40, 48, 62, al))
    img = blur(img, 18)
    cool = radial(CCW, CCH, *T(300, 480), 700 * KC, 900 * KC, 1.6)[..., None] * np.array([28, 38, 58], np.float32)
    warm = radial(CCW, CCH, *T(1000, 300), 600 * KC, 700 * KC, 1.8)[..., None] * np.array([55, 40, 24], np.float32)
    img = add_glow(img, cool + (warm * (0.4 if pale else 1.0)))
    return img.convert("RGBA")


def man_bust_shapes():
    body = Shape()
    body.poly([(540, 930), (690, 975), (920, 1060), (1040, 1230), (1110, 1920), (-30, 1920), (40, 1230), (160, 1060), (390, 975)], SUIT)
    body.poly([(455, 960), (540, 1170), (625, 960), (660, 1010), (540, 1260), (420, 1010)], (150, 155, 162))
    body.poly([(522, 1080), (558, 1080), (585, 1400), (540, 1470), (495, 1400)], (32, 40, 64))
    body.poly([(518, 1045), (562, 1045), (556, 1090), (524, 1090)], (38, 46, 72))
    body.poly([(420, 1010), (540, 1260), (470, 1500), (330, 1100)], (33, 35, 42))
    body.poly([(660, 1010), (540, 1260), (610, 1500), (750, 1100)], (33, 35, 42))
    body.poly([(470, 860), (610, 860), (625, 990), (455, 990)], (70, 56, 48))
    head = Shape()
    head.ell(430, 805, 17, 36, (70, 56, 48)).ell(650, 805, 17, 36, (70, 56, 48))
    head.curve([(540, 650), (615, 668), (650, 740), (648, 822), (630, 892), (592, 945), (540, 962), (488, 945), (450, 892), (432, 822), (430, 740), (465, 668)], (78, 63, 54))
    head.curve([(540, 626), (622, 640), (664, 700), (662, 772), (648, 764), (634, 708), (584, 684), (540, 690), (496, 684), (446, 708), (432, 764), (418, 772), (416, 700), (458, 640)], HAIR_G)
    return body, head


@functools.lru_cache(None)
def pclose_layers(pale):
    body, head = man_bust_shapes()
    b = body.render((CCW, CCH), T)
    hd = head.render((CCW, CCH), T)
    rimc = (200, 215, 240) if pale else (255, 214, 165)
    b = light_sprite(b, 1.1, 0.5, rimc, ((1, 1), (0, 1), (1, 0)), 7, 0.75, 5)
    b = light_sprite(b, 1.0, 1.0, (120, 150, 200), ((-1, 0),), 4, 0.5, 4)
    hd = face_model(hd, [(540, 650), (615, 668), (650, 740), (648, 822), (630, 892), (592, 945), (540, 962), (488, 945), (450, 892), (432, 822), (430, 740), (465, 668)],
                    [(505, 785, 40, 24), (577, 785, 40, 24)], (548, 800, 14, 60), (610, 820, 45, 110),
                    (50, 42, 36) if not pale else (46, 48, 54), 0.72 if not pale else 0.9)
    hd = light_sprite(hd, 1.0, 0.6, rimc, ((1, 0), (0, 1), (1, 1)), 8, 0.9, 6)
    hd = light_sprite(hd, 1.0, 1.0, (110, 140, 190), ((-1, 0),), 4, 0.6, 4)
    if pale:
        a = np.asarray(hd).astype(np.float32)
        g = a[..., :3].mean(axis=2, keepdims=True)
        a[..., :3] = a[..., :3] * 0.5 + g * 0.5 + 6
        hd = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    return blur(b, 1.2), blur(hd, 1.0)


def render_pclose(sh, t):
    p = sh["params"]
    mode = p.get("mode", "")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    pale = mode == "pale"
    canvas = pclose_bg(pale).copy()
    body, head = pclose_layers(pale)
    by = math.sin(t * 1.2) * 3
    hx, hy = 0.0, by
    if mode == "up":
        hy += 18 - 26 * ease(lt / 0.9)
    if mode == "bow":
        hy += 26 * ease(lt / 1.2)
    if mode in ("shake", "kneelshake"):
        hx += math.sin(t * 37) * 2.2 + math.sin(t * 23) * 1.4
        hy += math.sin(t * 31) * 1.6
        by += math.sin(t * 29) * 1.8
    if mode == "cry":
        by += max(0, math.sin(t * 5.2)) * 5
        hy = by + 18
    if mode in ("kneel", "kneelshake"):
        hy -= 18
    canvas.alpha_composite(body, (0, int(by)))
    canvas.alpha_composite(head, (int(hx), int(hy)))
    if mode in ("tears", "cry", "kneel", "kneelshake"):
        d = ImageDraw.Draw(canvas, "RGBA")
        for k, (x, spd, off) in enumerate(((600, 0.28, 0.0), (590, 0.22, 0.55))):
            ph = ((t * spd + off) % 1.0)
            if mode == "tears" and lt < 1.5:
                ph = ph * ss(lt / 1.5)
            yy = lerp(800, 960, ph)
            al = int(200 * math.sin(math.pi * ph))
            gx, gy = T(x + hx - ph * 6, yy + hy)
            d.ellipse([gx - 4, gy - 6, gx + 4, gy + 6], fill=(230, 235, 245, al))
    draw_dust(canvas, t, (180, 200, 230), 50, (0, 0, CCW, int(CCH * 0.7)), 0.6, (-3, -5))
    s0, s1, cy0, cy1 = 0.92, 0.98, 1000, 980
    if mode in ("kneel", "kneelshake"):
        s0, s1, cy0, cy1 = 0.96, 1.0, 900, 880
    elif mode == "cry":
        s0, s1 = 0.95, 1.03
    elif mode == "tears":
        s0, s1, cy0, cy1 = 0.95, 1.08, 960, 880
    e = ease(u)
    out = camera(canvas, CCW / 2 + math.sin(t * 0.17) * 5, lerp(cy0, cy1, e) * KC, lerp(s0, s1, e))
    return out.convert("RGB")


# ===================================================================== BOOK
BK_PW, BK_PH = 440, 640
BCX, BCY = 540, 900


@functools.lru_cache(None)
def page_img(seed, side):
    rng = np.random.default_rng(seed)
    a = np.zeros((BK_PH, BK_PW, 3), np.float32) + np.array([222, 208, 178], np.float32)
    xs = np.linspace(0, 1, BK_PW)[None, :]
    g = (xs if side == "R" else 1 - xs)
    a *= (0.72 + 0.28 * np.clip(g * 4, 0, 1) ** 0.6)[..., None]
    img = to_img(a).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    y = 70
    while y < BK_PH - 60:
        x = 50
        while x < BK_PW - 60:
            ln = int(rng.integers(18, 70))
            if x + ln > BK_PW - 50:
                break
            d.rounded_rectangle([x, y, x + ln, y + 5], radius=2, fill=(90, 72, 52, int(rng.integers(90, 150))))
            x += ln + int(rng.integers(8, 14))
        y += 24
        if rng.random() < 0.12:
            y += 20
    d.rectangle([0, 0, BK_PW - 1, BK_PH - 1], outline=(150, 130, 100, 120))
    return img


@functools.lru_cache(None)
def cover_img():
    a = vgrad(BK_PW, BK_PH, [(0, (72, 34, 26)), (1, (42, 18, 14))])
    img = to_img(a).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([22, 22, BK_PW - 23, BK_PH - 23], outline=(170, 130, 70, 140), width=3)
    d.rectangle([34, 34, BK_PW - 35, BK_PH - 35], outline=(170, 130, 70, 80), width=1)
    return img


@functools.lru_cache(None)
def book_bg(cold):
    a = vgrad(W, H, [(0, (14, 9, 6)), (1, (8, 5, 4))])
    rng = np.random.default_rng(5)
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(160):
        y = rng.uniform(0, H)
        d.line([(0, y), (W, y + rng.uniform(-30, 30))], fill=(40, 26, 16, int(rng.integers(10, 40))), width=int(rng.integers(1, 4)))
    img = blur(img, 1.5)
    col = np.array([130, 96, 60], np.float32) if not cold else np.array([70, 75, 85], np.float32)
    img = add_glow(img, radial(W, H, BCX, BCY, 820, 900, 1.4)[..., None] * col)
    return img.convert("RGBA")


def flip_page(canvas, front, back, prog, shade=1.0):
    """page flipping around spine from right (prog 0) to left (prog 1)"""
    c = math.cos(math.pi * prog)
    wv = int(abs(c) * BK_PW)
    if wv < 2:
        return
    x0 = BCX
    yb = BCY - BK_PH // 2
    lift = int(math.sin(math.pi * prog) * 28)
    if c > 0:
        im = front.resize((wv, BK_PH))
        dark = 1 - 0.35 * math.sin(math.pi * prog)
        im = Image.eval(im.convert("RGB"), lambda v: int(v * dark)).convert("RGBA")
        sh = Image.new("RGBA", (wv + 40, BK_PH), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([0, 0, wv + 20, BK_PH], fill=(0, 0, 0, int(90 * math.sin(math.pi * prog))))
        paste(canvas, blur(sh, 12), x0, yb + 10)
        paste(canvas, im, x0, yb - lift)
    else:
        im = back.resize((wv, BK_PH))
        dark = 1 - 0.35 * math.sin(math.pi * prog)
        im = Image.eval(im.convert("RGB"), lambda v: int(v * dark)).convert("RGBA")
        paste(canvas, im, x0 - wv, yb - lift)


def render_book(sh, t):
    p = sh["params"]
    mode = p.get("mode", "")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    bgimg = book_bg(mode == "still")
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    yb = BCY - BK_PH // 2
    L, R = page_img(1, "L"), page_img(2, "R")
    L2, R2 = page_img(3, "L"), page_img(4, "R")
    cov = cover_img()
    shadow = Image.new("RGBA", (BK_PW * 2 + 120, BK_PH + 120), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle([60, 60, BK_PW * 2 + 60, BK_PH + 60], fill=(0, 0, 0, 160))
    shadow = blur(shadow, 26)
    if mode in ("open", "close", "still"):
        pr = ease((lt - 0.5) / 1.3) if mode == "open" else (1 - ease((lt - 0.7) / 1.3) if mode == "close" else 0.0)
        # closed book base: pages on right with cover; as it opens left side shows inner cover
        sw = shadow.resize((int(BK_PW * (1 + pr) + 120), BK_PH + 120))
        paste(canvas, sw, BCX - BK_PW * pr - 60 + 12, yb - 60 + 16)
        paste(canvas, R, BCX, yb)
        if pr >= 0.5:
            wl = int(BK_PW * min(1, (pr - 0.5) * 2 + 0.0001))
            if pr > 0.99:
                paste(canvas, L, BCX - BK_PW, yb)
        if 0 < pr < 1:
            flip_page(canvas, cov, L, pr)
        elif pr <= 0:
            paste(canvas, cov, BCX, yb)
            ImageDraw.Draw(canvas).line([(BCX, yb), (BCX, yb + BK_PH)], fill=(20, 8, 6, 255), width=6)
        else:
            paste(canvas, L, BCX - BK_PW, yb)
        shift = -BK_PW / 2 * (1 - pr)
        cx, s = BCX, (lerp(1.12, 1.0, ease(u)) if mode != "still" else lerp(1.0, 1.12, ease(u)))
    else:
        paste(canvas, shadow, BCX - BK_PW - 60 + 12, yb - 60 + 16)
        paste(canvas, L, BCX - BK_PW, yb)
        pr = ease((lt - 0.4) / 1.4) if mode == "turn" else 1.0
        paste(canvas, R2 if mode == "turn" else R, BCX, yb)
        if mode == "turn":
            if pr >= 1:
                paste(canvas, L2, BCX - BK_PW, yb)
            elif pr > 0.5:
                pass
            flip_page(canvas, R, L2, pr)
        # spine gutter shadow
        g = Image.new("RGBA", (80, BK_PH), (0, 0, 0, 0))
        ImageDraw.Draw(g).rectangle([30, 0, 50, BK_PH], fill=(40, 25, 10, 120))
        paste(canvas, blur(g, 12), BCX - 40, yb)
        cx, s = BCX, lerp(1.0, 1.1, ease(u))
        shift = 0
        if mode == "point":
            gl = radial(W, H, BCX + 120, BCY - 60, 380, 300, 1.5)[..., None] * np.array([90, 70, 40], np.float32) * ease(lt / 2.5)
            bgimg = add_glow(bgimg.convert("RGB"), gl).convert("RGBA")
            s = lerp(1.0, 1.25, ease(u))
            cx = BCX + 60 * ease(u)
    full = bgimg.copy()
    full.alpha_composite(canvas, (int(shift), 0)) if shift <= 0 and False else paste(full, canvas, shift, 0)
    canvas = full
    draw_dust(canvas, t, (255, 226, 180), 60, (0, 0, W, H), 0.7, (2, -5))
    out = camera(canvas, cx, BCY + 60, s)
    return out.convert("RGB")


# ===================================================================== CANDLES
CF = 1150.0
CVPY = 820.0
CEYE = 1.1


@functools.lru_cache(None)
def candles_bg():
    a = vgrad(W, H, [(0, (1, 1, 2)), (0.4, (5, 5, 7)), (0.43, (10, 10, 13)), (1, (4, 4, 5))])
    img = add_glow(to_img(a), radial(W, H, 540, CVPY + 20, 900, 120, 1.4)[..., None] * np.array([14, 14, 18], np.float32))
    return img.convert("RGBA")


@functools.lru_cache(None)
def glow_sprite(r, warm=True):
    s = int(r * 4) + 2
    g = radial(s, s, s / 2, s / 2, s / 2, s / 2, 2.2)
    col = np.array([255, 170, 80] if warm else [220, 220, 255], np.float32)
    a = (g * 255).astype(np.uint8)
    rgb = np.ones((s, s, 3), np.float32) * col
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), a]), "RGBA")


@functools.lru_cache(None)
def candle_sprite(hpx):
    hpx = max(4, hpx)
    wpx = max(2, int(hpx * 0.32))
    im = Image.new("RGBA", (wpx * 3, int(hpx * 2.2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x0 = wpx
    top = int(hpx * 1.2)
    body = vgrad(wpx, hpx, [(0, (225, 205, 170)), (1, (120, 100, 80))])
    im.paste(to_img(body).convert("RGBA"), (x0, top))
    return im


def render_candles(sh, t):
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    canvas = candles_bg().copy()
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cands = sorted(sh.get("candles", []), key=lambda c: -c["z"])
    camz = -1.2 * ease(u)
    for i, c in enumerate(cands):
        a = ss((t - c["t"]) / 0.8)
        if a <= 0:
            continue
        z = c["z"] + camz
        if z < 0.8:
            continue
        sx = 540 + CF * c["x"] / z
        sy = CVPY + CF * CEYE / z
        if sx < -50 or sx > W + 50:
            continue
        hpx = int(CF * 0.22 / z)
        sp = candle_sprite(hpx)
        paste(canvas, with_alpha(sp, a), sx - sp.size[0] / 2, sy - sp.size[1])
        fl = 0.85 + 0.15 * math.sin(t * 13 + i * 1.7) * math.sin(t * 7.3 + i)
        fh = max(3, hpx * 0.45 * fl)
        fx, fy = sx, sy - hpx - fh * 0.6
        d = ImageDraw.Draw(canvas, "RGBA")
        d.ellipse([fx - fh * 0.22, fy - fh * 0.6, fx + fh * 0.22, fy + fh * 0.45], fill=(255, 236, 190, int(255 * a)))
        gr = max(4, int(fh * 3.2 * fl))
        gs = glow_sprite(min(gr, 400))
        paste(glow, with_alpha(gs, a * 0.55), fx - gs.size[0] / 2, fy - gs.size[1] / 2)
        # reflection
        rs = glow_sprite(max(3, int(gr * 0.5)))
        rs = rs.resize((rs.size[0], rs.size[1] * 2))
        paste(glow, with_alpha(rs, a * 0.18), sx - rs.size[0] / 2, sy + 2)
    out = Image.alpha_composite(canvas, glow)
    g = np.asarray(glow).astype(np.float32)
    base = np.asarray(canvas.convert("RGB")).astype(np.float32)
    base += g[..., :3] * (g[..., 3:] / 255) * 0.9
    out = to_img(base).convert("RGBA")
    draw_dust(out, t, (255, 210, 160), 40, (0, 0, W, H), 0.5, (2, -8))
    return out.convert("RGB")


# ===================================================================== SCALE
def render_scale(sh, t):
    p = sh["params"]
    mode = p.get("mode", "")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    bg = add_glow(Image.new("RGB", (W, H), (5, 5, 6)), radial(W, H, 540, 800, 700, 800, 1.6)[..., None] * np.array([45, 36, 24], np.float32))
    canvas = bg.convert("RGBA")
    its = [it for it in ITEMS if it["shot"] == SHOTS.index(sh)]
    it0 = its[0]
    ft = lt
    fall = 0.0
    if mode == "break":
        prev = [s for s in SHOTS if s["type"] == "scale" and s is not sh][0]
        ft = sh["start"] - prev["start"] + 99
        fall = ease((lt - 0.2) / 2.6)
    # timeline for blocks
    if mode != "break":
        ws = it0["words"]
        def wt(word):
            for w in ws:
                if word in w["text"]:
                    return it0["start"] + w["t"] - sh["start"]
            return None
        tb = wt("boas") or dur * 0.4
        tm = wt("más") or dur * 0.6
        tq = wt("pilha") or dur * 0.8
    else:
        tb, tm, tq = 0, 0, 0
    nl = int(clamp((ft - tb) / 1.6) * 5 + (ft > tb)) if ft > tb else 0
    nr = int(clamp((ft - tm) / 1.6) * 5 + (ft > tm)) if ft > tm else 0
    nl, nr = min(nl, 5), min(nr, 5)
    ang = (nr - nl) * 2.2 + math.sin(ft * 1.1) * 1.2
    if ft > tq:
        ang += math.sin((ft - tq) * 2.3) * 6 * math.exp(-(ft - tq) * 0.4)
    layer = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    S2 = 2
    gold = (205, 170, 110, 255)
    px, py = 540, 700
    d.line([(px * S2, py * S2), (px * S2, 1300 * S2)], fill=gold, width=8)
    d.polygon([((px - 150) * S2, 1320 * S2), ((px + 150) * S2, 1320 * S2), ((px + 40) * S2, 1285 * S2), ((px - 40) * S2, 1285 * S2)], fill=gold)
    d.ellipse([(px - 16) * S2, (py - 16) * S2, (px + 16) * S2, (py + 16) * S2], fill=gold)
    a = math.radians(ang)
    Lb = 330
    ends = []
    for sg in (-1, 1):
        ex, ey = px + sg * Lb * math.cos(a), py + sg * Lb * math.sin(a)
        ends.append((ex, ey))
    d.line([(ends[0][0] * S2, ends[0][1] * S2), (ends[1][0] * S2, ends[1][1] * S2)], fill=gold, width=9)
    for k, (ex, ey) in enumerate(ends):
        pyb = ey + 330
        d.line([(ex * S2, ey * S2), ((ex - 95) * S2, pyb * S2)], fill=(180, 150, 100, 200), width=3)
        d.line([(ex * S2, ey * S2), ((ex + 95) * S2, pyb * S2)], fill=(180, 150, 100, 200), width=3)
        d.chord([(ex - 115) * S2, (pyb - 45) * S2, (ex + 115) * S2, (pyb + 45) * S2], 0, 180, fill=gold)
        n = nl if k == 0 else nr
        col = (255, 214, 140, 255) if k == 0 else (110, 112, 118, 255)
        for b in range(n):
            bx = ex - 70 + (b % 3) * 50
            by = pyb - 38 - (b // 3) * 44
            d.rectangle([bx * S2, by * S2, (bx + 40) * S2, (by + 36) * S2], fill=col)
    layer = layer.resize((W, H), Image.LANCZOS)
    if fall > 0:
        layer = layer.rotate(-fall * 8, center=(540, 700), resample=Image.BICUBIC, translate=(0, fall * fall * 500))
        layer = with_alpha(layer, 1 - fall)
    gl = blur(layer, 14)
    canvas.alpha_composite(with_alpha(gl, 0.8))
    canvas.alpha_composite(layer)
    draw_dust(canvas, t, (255, 226, 180), 50, (0, 0, W, H), 0.5, (2, -5))
    s = lerp(0.95, 1.02, ease(lt / max(dur, 1)))
    return camera(canvas, 540, 960, s).convert("RGB")


# ===================================================================== CROWD
@functools.lru_cache(None)
def crowd_layer(seed, fade_rows=False):
    rng = np.random.default_rng(seed)
    im = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rows = 9
    for r in range(rows):
        k = r / (rows - 1)
        y = lerp(1000, 1900, k) * 2
        sc = lerp(0.35, 1.5, k)
        shade = int(lerp(40, 12, k))
        x = -40
        while x < W * 2 + 60:
            hr = 52 * sc * rng.uniform(0.85, 1.1)
            d.ellipse([x - hr * 0.8, y - hr * 1.0, x + hr * 0.8, y + hr * 1.0], fill=(shade, shade + 3, shade + 8, 255))
            d.polygon([(x - hr * 2.4, y + hr * 3.2), (x - hr * 1.8, y + hr * 1.2), (x, y + hr * 0.9), (x + hr * 1.8, y + hr * 1.2), (x + hr * 2.4, y + hr * 3.2)], fill=(shade, shade + 3, shade + 8, 255))
            if rng.random() < 0.28:
                sg = 1 if rng.random() < 0.5 else -1
                d.line([(x + sg * hr * 1.4, y + hr), (x + sg * hr * 2.0, y - hr * 2.2), (x + sg * hr * 1.7, y - hr * 3.4)], fill=(shade, shade + 3, shade + 8, 255), width=int(hr * 0.7))
            x += hr * rng.uniform(2.6, 3.6)
    im = im.resize((W, H), Image.LANCZOS)
    return light_sprite(im, 1.0, 1.0, (190, 205, 235), ((0, 1),), 3, 0.8, 2)


def render_crowd(sh, t):
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    bg = vgrad(W, H, [(0, (4, 5, 8)), (0.45, (26, 30, 40)), (1, (4, 4, 6))])
    stage = radial(W, H, 540, 700, 800, 380, 1.4)[..., None] * np.array([90, 95, 110], np.float32) * (1 - 0.6 * u)
    img = to_img(bg + stage).convert("RGBA")
    cl = crowd_layer(1)
    presence = 1 - ease((u - 0.15) / 0.8) * 0.92
    sway = int(math.sin(t * 1.5) * 6)
    paste(img, with_alpha(cl, presence), sway, int(math.sin(t * 3.1) * 4))
    rng = np.random.default_rng(int(t * 8))
    d = ImageDraw.Draw(img, "RGBA")
    if rng.random() < 0.7 * presence:
        for _ in range(rng.integers(1, 4)):
            x, y = rng.uniform(0, W), rng.uniform(900, 1600)
            gs = glow_sprite(40, False)
            paste(img, with_alpha(gs, presence * rng.uniform(0.4, 1.0)), x - gs.size[0] / 2, y - gs.size[1] / 2)
    draw_dust(img, t, (200, 210, 230), 60, (0, 0, W, H), 0.5, (2, -5))
    return camera(img, 540, 960, lerp(1.0, 1.08, ease(u))).convert("RGB")


# ===================================================================== VOID
def render_void(sh, t):
    kind = sh["params"].get("kind")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    img = Image.new("RGBA", (W, H), (3, 3, 4, 255))
    a = ss(lt / 1.2) * (1 - ss((u - 0.55) / 0.45))
    lay = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    col = (150, 152, 160, 255)
    if kind == "flags":
        for i, x in enumerate((300, 540, 780)):
            base_y = 1250 + (i == 1) * -40
            d.line([(x * 2, (base_y - 700) * 2), (x * 2, base_y * 2)], fill=col, width=6)
            pts_top, pts_bot = [], []
            for k in range(21):
                f = k / 20
                wx = x + f * 260
                wy = base_y - 690 + math.sin(t * 2.6 + f * 5 + i) * 22 * f
                pts_top.append((wx * 2, wy * 2))
                pts_bot.append((wx * 2, (wy + 170) * 2))
            d.polygon(pts_top + pts_bot[::-1], outline=col, fill=(60, 62, 68, 120), width=4)
    elif kind == "crowd":
        cl = crowd_layer(2)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        lay.alpha_composite(cl)
        lay = lay.resize((W * 2, H * 2))
    elif kind == "cameras":
        for i, (x, y, s) in enumerate(((270, 1050, 1.0), (560, 980, 0.8), (820, 1080, 1.1))):
            x2, y2 = x * 2, y * 2
            s2 = s * 2
            d.rectangle([x2 - 90 * s2, y2 - 60 * s2, x2 + 70 * s2, y2 + 50 * s2], outline=col, width=5, fill=(40, 42, 48, 200))
            d.rectangle([x2 + 70 * s2, y2 - 30 * s2, x2 + 130 * s2, y2 + 30 * s2], outline=col, width=5, fill=(40, 42, 48, 200))
            d.ellipse([x2 + 110 * s2, y2 - 38 * s2, x2 + 150 * s2, y2 + 38 * s2], outline=col, width=5)
            for dx in (-80, 0, 80):
                d.line([(x2, y2 + 50 * s2), (x2 + dx * s2, y2 + 380 * s2)], fill=col, width=5)
            if (int(t * 2 + i) % 3) == 0:
                gx, gy = x + 130 * s, y
                d.ellipse([(gx - 8) * 2, (gy - 8) * 2, (gx + 8) * 2, (gy + 8) * 2], fill=(200, 60, 50, 255))
    lay = lay.resize((W, H), Image.LANCZOS)
    br = (1 - a) * 10
    if br > 0.5:
        lay = blur(lay, br)
    lay = with_alpha(lay, a * 0.8)
    img.alpha_composite(lay, (0, int(-u * 40)))
    draw_dust(img, t, (200, 200, 210), 60, (0, 0, W, H), 0.5, (2, -6))
    return img.convert("RGB")


# ===================================================================== EYES
def render_eyes(sh, t):
    lt = t - sh["start"]
    fake = dict(type="hall", params={"cam": "reveal"}, start=sh["start"], end=sh["end"] + 20)
    base = render_hall(fake, t)
    br = max(0, 14 * (1 - lt / 3.2))
    if br > 0.4:
        base = blur(base, br)
    if lt < 0.9:
        o = 0.35 * ss(lt / 0.9)
    elif lt < 1.3:
        o = lerp(0.35, 0.08, ss((lt - 0.9) / 0.4))
    else:
        o = lerp(0.08, 1.4, ease((lt - 1.3) / 1.8))
    m = Image.new("L", (270, 480), 0)
    ImageDraw.Draw(m).ellipse([-80, 240 - o * 300, 350, 240 + o * 300], fill=255)
    m = blur(m, 18).resize((W, H), Image.BILINEAR)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    return Image.composite(base, black, m)


# ===================================================================== DOOR
@functools.lru_cache(None)
def door_bg():
    a = vgrad(W, H, [(0, (5, 5, 7)), (0.5, (20, 21, 26)), (0.78, (16, 16, 19)), (1, (6, 6, 7))])
    img = to_img(a)
    d = ImageDraw.Draw(img, "RGBA")
    rng = np.random.default_rng(9)
    for row in range(0, 1500, 90):
        off = 0 if (row // 90) % 2 == 0 else 110
        d.line([(0, row), (W, row)], fill=(0, 0, 0, 60), width=2)
        for x in range(-off, W, 220):
            d.line([(x, row), (x, row + 90)], fill=(0, 0, 0, 45), width=2)
    d.rectangle([0, 1480, W, H], fill=(10, 10, 12, 255))
    for zi in range(12):
        y = 1480 + (zi ** 1.6) * 14
        d.line([(0, y), (W, y)], fill=(34, 36, 44, 40), width=2)
    img = add_glow(img, radial(W, H, 380, 400, 900, 1100, 1.5)[..., None] * np.array([40, 32, 22], np.float32))
    return img.convert("RGBA")


def render_door(sh, t):
    p = sh["params"]
    mode = p.get("mode")
    lt = t - sh["start"]
    dur = sh["end"] - sh["start"]
    u = clamp(lt / max(dur, 0.01))
    img = door_bg().copy()
    x0, y0, x1, y1 = 330, 420, 750, 1480
    appear = ss(lt / 3.0) if mode == "closed" else 1.0
    opn = 0.0
    if mode == "open":
        opn = ease((lt - 0.3) / 3.2)
    d = ImageDraw.Draw(img, "RGBA")
    if appear > 0:
        inner = int(255 * ss(appear * 1.6 - 0.4))
        d.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, inner))
        if opn < 1:
            half = (x1 - x0) / 2
            wv = half * (1 - opn)
            panel = (24, 22, 22, inner)
            d.polygon([(x0, y0), (x0 + wv, y0 + 30 * opn), (x0 + wv, y1 - 20 * opn), (x0, y1)], fill=panel)
            d.polygon([(x1, y0), (x1 - wv, y0 + 30 * opn), (x1 - wv, y1 - 20 * opn), (x1, y1)], fill=panel)
            if wv > 20:
                for xx, sg in ((x0, 1), (x1, -1)):
                    d.rectangle([min(xx + sg * wv * 0.15, xx + sg * wv * 0.85), y0 + 60, max(xx + sg * wv * 0.15, xx + sg * wv * 0.85), y0 + 460], outline=(40, 38, 38, inner), width=3)
                    d.rectangle([min(xx + sg * wv * 0.15, xx + sg * wv * 0.85), y0 + 520, max(xx + sg * wv * 0.15, xx + sg * wv * 0.85), y1 - 70], outline=(40, 38, 38, inner), width=3)
        if opn > 0:
            # darkness pulls particles in
            for i in range(60):
                pp = DUST[i]
                ph = (t * (0.08 + pp[2] * 0.1) + pp[0]) % 1
                sx = lerp(pp[1] * W, 540, ph)
                sy = lerp(pp[3] * 1800, 950, ph)
                al = int(160 * math.sin(math.pi * ph) * opn)
                d.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=(170, 175, 190, al))
        # frame outline
        per = [(x0, y1), (x0, y0), (x1, y0), (x1, y1)]
        lens = [y1 - y0, x1 - x0, y1 - y0]
        rem = appear * sum(lens) * 1.15
        pts = [per[0]]
        for i in range(3):
            if rem <= 0:
                break
            f = clamp(rem / lens[i])
            pts.append((lerp(per[i][0], per[i + 1][0], f), lerp(per[i][1], per[i + 1][1], f)))
            rem -= lens[i]
        line = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(line).line(pts, fill=(175, 190, 215, 230), width=4)
        img.alpha_composite(blur(line, 10))
        img.alpha_composite(blur(line, 3))
    # the room dims as the door opens
    draw_dust(img, t, (200, 200, 215), 50, (0, 0, W, H), 0.5 * (1 - opn * 0.5), (2, -5))
    s = lerp(1.0, 1.12, ease(u))
    out = camera(img, 540, 960, s)
    if opn > 0:
        out = Image.eval(out, lambda v, k=1 - 0.35 * opn: int(v * k))
    return out.convert("RGB")


# ===================================================================== WORDS / END
def render_words_bg(sh, t):
    img = Image.new("RGBA", (W, H), (4, 4, 5, 255))
    img = add_glow(img.convert("RGB"), radial(W, H, 540, 900, 700, 600, 1.8)[..., None] * np.array([26, 22, 16], np.float32)).convert("RGBA")
    draw_dust(img, t, (255, 226, 180), 60, (0, 0, W, H), 0.5, (2, -6))
    return img


_words_bg_cache = {}


def render_words(sh, t):
    if "bg" not in _words_bg_cache:
        img = Image.new("RGB", (W, H), (4, 4, 5))
        _words_bg_cache["bg"] = add_glow(img, radial(W, H, 540, 900, 700, 600, 1.8)[..., None] * np.array([26, 22, 16], np.float32)).convert("RGBA")
    img = _words_bg_cache["bg"].copy()
    draw_dust(img, t, (255, 226, 180), 60, (0, 0, W, H), 0.5, (2, -6))
    si = SHOTS.index(sh)
    its = [it for it in ITEMS if it["shot"] == si and it["spk"] != "S"]
    lst = sh["params"].get("mode") == "list"
    for k, it in enumerate(its):
        st = it["start"] - 0.05
        if t < st:
            continue
        nxt = its[k + 1]["start"] if k + 1 < len(its) else sh["end"] + 5
        a = ss((t - st) / 0.5)
        if lst:
            age = sum(1 for j in its[k + 1:] if j["start"] <= t)
            a *= [1, 0.35, 0.15, 0.06, 0][min(age, 4)]
            yoff = -age * 120 - ease((t - nxt + 0.3) / 0.5) * 0 if age else 0
        else:
            a *= 1 - ss((t - nxt + 0.2) / 0.4)
            yoff = 0
        if a <= 0.01:
            continue
        col = FONT[it["spk"]][2]
        txt = it["text"].upper()
        size = 96 if len(txt) < 16 else 70
        f = font(FD + "Cinzel.ttf", size)
        lines = wrap(txt, f, 900)
        lh = size * 1.25
        y = 900 - lh * len(lines) / 2 + (yoff if lst else 0) - 8 * (1 - ss((t - st) / 1.2))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for ln in lines:
            wdt = d.textlength(ln, font=f)
            d.text(((W - wdt) / 2, y), ln, font=f, fill=rgba(col, 255))
            y += lh
        img.alpha_composite(with_alpha(blur(layer, 12), a * 0.6))
        img.alpha_composite(with_alpha(layer, a))
    return img.convert("RGB")


def render_end(sh, t):
    lt = t - sh["start"]
    img = Image.new("RGB", (W, H), (0, 0, 0))
    k = ss(lt / 3) * (1 - ss((t - (TOTAL - 5)) / 3))
    img = add_glow(img, radial(W, H, 540, 900, 420, 420, 2.4)[..., None] * np.array([40, 30, 18], np.float32) * k)
    return img


RENDER = dict(hall=render_hall, jclose=render_jclose, pclose=render_pclose, book=render_book, candles=render_candles,
              scale=render_scale, crowd=render_crowd, void=render_void, eyes=render_eyes, door=render_door,
              words=render_words, end=render_end)

for i, s in enumerate(SHOTS):
    its = [it for it in ITEMS if it["shot"] == i]
    s["_first_text"] = its[0]["text"] if its else ""


# ===================================================================== SUBTITLES
def wrap(text, f, maxw):
    words = text.split()
    lines, cur = [], ""
    dd = ImageDraw.Draw(Image.new("L", (1, 1)))
    for w_ in words:
        tr = (cur + " " + w_).strip()
        if dd.textlength(tr, font=f) <= maxw:
            cur = tr
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    # balance two-line splits
    return lines


@functools.lru_cache(None)
def sub_layout(idx, center_y):
    it = ITEMS[idx]
    if it["spk"] == "S":
        f = font(FD + "Cinzel.ttf", 34)
        txt = " ".join("SILÊNCIO")
        img = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        wdt = d.textlength(txt, font=f)
        d.text(((W - wdt) / 2, 40), txt, font=f, fill=(165, 160, 150, 200))
        return img, [(0, 0, W, 120)], center_y - 60
    path, size, col = FONT[it["spk"]]
    text = it["text"]
    if len(text) > 170:
        size = int(size * 0.86)
    elif len(text) > 110:
        size = int(size * 0.93)
    f = font(path, size)
    maxw = 900
    lines = wrap(text, f, maxw)
    lh = int(size * 1.22)
    hh = lh * len(lines) + 40
    img = Image.new("RGBA", (W, hh), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ds = ImageDraw.Draw(shadow)
    boxes = []
    y = 20
    for ln in lines:
        wdt = d.textlength(ln, font=f)
        x = (W - wdt) / 2
        for w_ in ln.split(" "):
            ww = d.textlength(w_, font=f)
            boxes.append((int(x) - 2, y - 4, int(x + ww) + 6, y + lh))
            x += ww + d.textlength(" ", font=f)
        d.text(((W - wdt) / 2, y), ln, font=f, fill=rgba(col, 255))
        ds.text(((W - wdt) / 2, y + 3), ln, font=f, fill=(0, 0, 0, 255))
        y += lh
    shadow = blur(shadow, 7)
    comb = Image.alpha_composite(shadow, Image.alpha_composite(with_alpha(shadow, 0.8), img))
    return comb, boxes, center_y - hh / 2


def draw_subs(frame, t, shot):
    if shot["type"] == "words":
        return frame
    frame = frame.convert("RGBA")
    for idx, it in enumerate(ITEMS):
        if it["start"] - 0.15 > t:
            break
        nxt = ITEMS[idx + 1]["start"] if idx + 1 < len(ITEMS) else TOTAL
        end = min(it["start"] + it["dur"] + max(it.get("pause", 0), 0.3), nxt - 0.42)
        if t > end + 0.35:
            continue
        sh = SHOTS[it["shot"]]
        if sh["type"] == "words":
            continue
        cy = 1500
        if sh["type"] in ("void", "end"):
            cy = 1000 if sh["type"] == "end" else 1380
        img, boxes, y0 = sub_layout(idx, cy)
        fade_out = 1 - ss((t - end) / 0.3)
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        ws = it["words"]
        if it["spk"] == "S":
            md.rectangle(boxes[0], fill=int(255 * ss((t - it["start"]) / 0.6) * fade_out))
        else:
            for bi, bx in enumerate(boxes):
                wt = it["start"] + (ws[bi]["t"] if bi < len(ws) else ws[-1]["t"] if ws else 0) - 0.06
                al = ss((t - wt) / 0.22) * fade_out
                if al > 0:
                    md.rectangle(bx, fill=int(255 * al))
            mask = blur(mask, 3)
        al = ImageChops.multiply(img.getchannel("A"), mask)
        im2 = img.copy()
        im2.putalpha(al)
        paste(frame, im2, 0, y0)
    return frame.convert("RGB")


# ===================================================================== FRAME
def shot_at(t):
    for i, s in enumerate(SHOTS):
        if s["start"] <= t < s["end"]:
            return i
    return len(SHOTS) - 1


def xfade_len(a, b):
    ta, tb = SHOTS[a]["type"], SHOTS[b]["type"]
    if tb == "end":
        return 2.0
    if {ta, tb} <= {"hall", "jclose", "pclose"}:
        return 0.35
    return 0.8


def render_frame(t):
    i = shot_at(t)
    sh = SHOTS[i]
    img = RENDER[sh["type"]](sh, t)
    # crossfade into this shot from previous
    if i > 0:
        d = xfade_len(i - 1, i)
        if t - sh["start"] < d:
            w = ss((t - sh["start"]) / d)
            prev = RENDER[SHOTS[i - 1]["type"]](SHOTS[i - 1], t)
            img = Image.blend(prev, img, w)
    img = finish(img, t)
    img = draw_subs(img, t, sh)
    fi = ss(t / 1.5)
    fo = 1 - ss((t - (TOTAL - 3.5)) / 3.0)
    k = fi * fo
    if k < 1:
        img = Image.eval(img, lambda v: int(v * k))
    return img


if __name__ == "__main__":
    if sys.argv[1] == "still":
        for ts in sys.argv[2:]:
            t = float(ts)
            render_frame(t).save(f"still_{t:07.2f}.jpg", quality=88)
    elif sys.argv[1] == "chunk":
        a, b, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        proc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for fr in range(a, b):
            proc.stdin.write(render_frame(fr / FPS).tobytes())
            if fr % 300 == 0:
                print(out, fr, flush=True)
        proc.stdin.close()
        proc.wait()
