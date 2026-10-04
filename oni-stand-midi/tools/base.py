"""Stepped pedestal with a wave frieze, the 鬼 panel and the neck peg.

Frame: z = 0 is the table, the front looks towards +y.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import map_coordinates

import params as C
from sdf import cut, round_box, smin

FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
KANJI = "鬼"          # "oni"

PX_MM = 10            # relief raster resolution, px per mm
RELIEF = 1.6          # relief height, mm

# pedestal tiers: (half width, z0, z1, edge radius)
TIERS = [(75.0, 0.0, 9.0, 2.0), (71.0, 9.0, 13.0, 1.0), (66.0, 13.0, 41.0, 1.0),
         (71.0, 41.0, 46.0, 1.0), (68.0, 46.0, 50.0, 1.0)]
FRIEZE = TIERS[2]
TOP = 50.0
NECK = 22.0           # half width of the visible neck
NECK_Z = (54.0, 86.0)
PEG_Z = (86.0, 86.0 + C.PEG_LEN)
CABLE_Y = -46.0       # cable slot behind the neck


def _spiral(draw, cx, cy, R, turns, w, sense=1, phase=0.0):
    th = np.linspace(0.6 * np.pi, 2 * np.pi * turns, 200)
    r = R * th / th[-1]
    pts = [(cx + sense * rr * np.cos(t + phase), cy - rr * np.sin(t + phase)) for rr, t in zip(r, th)]
    draw.line(pts, fill=255, width=w, joint="curve")
    return pts[-1]


def _wave_panel(width_mm, height_mm, gap=None):
    """Raised waves: rolling swells plus curling crests. gap=(u0, u1) keeps room for the panel."""
    W, H = int(width_mm * PX_MM), int(height_mm * PX_MM)
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    lw = int(1.6 * PX_MM)
    u = np.linspace(0, width_mm, 400)
    for k, base in enumerate((0.70, 0.48)):
        v = height_mm * base + 1.8 * np.sin(2 * np.pi * u / 26 + k * 1.6)
        dr.line([(a * PX_MM, (height_mm - b) * PX_MM) for a, b in zip(u, v)], fill=255, width=lw, joint="curve")
    n = max(2, int(round(width_mm / 42)))
    for i in range(n):
        cu = (i + 0.5) * width_mm / n
        s = 1 if i % 2 == 0 else -1
        cx, cy = cu * PX_MM, height_mm * 0.36 * PX_MM
        end = _spiral(dr, cx, cy, 8.0 * PX_MM, 2.4, lw, sense=s)
        # the crest sweeping back down into the swell
        tail = [end, (cx - s * 10 * PX_MM, cy + 3 * PX_MM), (cx - s * 17 * PX_MM, height_mm * 0.50 * PX_MM)]
        dr.line(tail, fill=255, width=lw, joint="curve")
        for j in range(3):   # spray
            ex, ey = cx + s * (9 + 3.2 * j) * PX_MM, cy - (6.5 + 1.5 * j) * PX_MM
            dr.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=255)
    dr.rectangle([0, 0, W, int(1.2 * PX_MM)], fill=255)
    dr.rectangle([0, H - int(1.2 * PX_MM), W, H], fill=255)
    if gap:
        dr.rectangle([gap[0] * PX_MM, 0, gap[1] * PX_MM, H], fill=0)
    return im


def _kanji_panel(w_mm, h_mm):
    W, H = int(w_mm * PX_MM), int(h_mm * PX_MM)
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    b = int(1.4 * PX_MM)
    dr.rectangle([0, 0, W - 1, H - 1], outline=255, width=b)
    font = ImageFont.truetype(FONT, int((h_mm - 6) * PX_MM))
    bbox = dr.textbbox((0, 0), KANJI, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    dr.text(((W - tw) / 2 - bbox[0], (H - th) / 2 - bbox[1]), KANJI, fill=255, font=font)
    return im


class Relief:
    """A height map laid on one face. u runs left->right seen from outside, v bottom->top."""

    def __init__(self, img, u0, v0, blur_mm=0.35):
        img = img.filter(ImageFilter.GaussianBlur(blur_mm * PX_MM))
        self.h = np.asarray(img, np.float32)[::-1] / 255.0   # row 0 = bottom
        self.u0, self.v0 = u0, v0

    def __call__(self, u, v):
        r = (v - self.v0) * PX_MM
        c = (u - self.u0) * PX_MM
        return map_coordinates(self.h, [r, c], order=1, mode="constant", cval=0.0)


def build_reliefs():
    hw, z0, z1, _ = FRIEZE
    w = 2 * hw - 6
    h = z1 - z0 - 3
    pw, ph = 30.0, h
    front = _wave_panel(w, h, gap=(w / 2 - pw / 2 - 2, w / 2 + pw / 2 + 2))
    front.paste(_kanji_panel(pw, ph), (int((w / 2 - pw / 2) * PX_MM), 0))
    sides = _wave_panel(w, h)
    return {
        "front": Relief(front, -w / 2, z0 + 1.5),
        "side": Relief(sides, -w / 2, z0 + 1.5),
    }


R = None


def base_sdf(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    d = None
    for hw, z0, z1, r in TIERS:
        t = round_box(P, (0, 0, (z0 + z1) / 2), (hw, hw, (z1 - z0) / 2), r)
        if hw == FRIEZE[0]:
            # relief on the face each point is closest to
            ax, ay = np.abs(x), np.abs(y)
            band = (z > z0) & (z < z1) & (np.maximum(ax, ay) > hw - 3)
            h = np.zeros(len(P), np.float32)
            if band.any():
                xb, yb, zb = x[band], y[band], z[band]
                hb = np.zeros(band.sum(), np.float32)
                fy = np.abs(yb) >= np.abs(xb)
                # u = the viewer's right when standing in front of that face
                m = fy & (yb > 0); hb[m] = R["front"](-xb[m], zb[m])
                m = fy & (yb <= 0); hb[m] = R["side"](xb[m], zb[m])
                m = ~fy & (xb > 0); hb[m] = R["side"](yb[m], zb[m])
                m = ~fy & (xb <= 0); hb[m] = R["side"](-yb[m], zb[m])
                h[band] = hb
            t = t - RELIEF * h
        d = t if d is None else np.minimum(d, t)

    # neck: small plinth, shaft, peg
    d = smin(d, round_box(P, (0, 0, (TOP + 54) / 2), (NECK + 4, NECK + 4, 2.0 + 1), 1.0), 0.8)
    d = smin(d, round_box(P, (0, 0, np.mean(NECK_Z)), (NECK, NECK, (NECK_Z[1] - NECK_Z[0]) / 2), 2.5), 1.5)
    # carved vertical grooves on the shaft
    for gx in (-11.0, 0.0, 11.0):
        for sgn in (1, -1):
            g = round_box(P, (gx, sgn * NECK, 70), (1.2, 1.2, 10), 1.1)
            d = cut(d, g)
            g = round_box(P, (sgn * NECK, gx, 70), (1.2, 1.2, 10), 1.1)
            d = cut(d, g)
    s = C.PEG / 2
    peg = round_box(P, (0, 0, np.mean(PEG_Z)), (s, s, C.PEG_LEN / 2), 1.0)
    peg = np.maximum(peg, np.maximum(np.abs(x), np.abs(y)) - (s - 1.5) + (z - PEG_Z[1]) - 0.0)
    d = np.minimum(d, peg)

    # underside: shallow recess (sits on its rim), weight pocket, cable path
    d = cut(d, round_box(P, (0, 0, 0), (64, 64, 1.2), 3.0))
    d = cut(d, np.maximum(np.sqrt(x ** 2 + y ** 2) - 30.5, np.abs(z - 3) - 6))
    d = cut(d, round_box(P, (0, CABLE_Y, 30), (10, 5.5, 40), 5.0))
    d = cut(d, round_box(P, (0, (CABLE_Y - 80) / 2, 0), (6.5, (80 + CABLE_Y) / 2 + 2, 8), 2.0))
    return d


def setup():
    global R
    R = build_reliefs()


BASE_LO = (-78, -78, -1)
BASE_HI = (78, 78, PEG_Z[1] + 1)
HEAD_Z = NECK_Z[1] + 32.0     # head frame z = 0 sits this high in the assembly
