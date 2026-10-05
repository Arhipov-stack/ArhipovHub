"""Draw the concept sheet of the magnetic-vacuum sealing plate as an SVG.

The general view is a real 3D scene (pipe, plate, magnet blocks, straps)
projected orthographically, so the parts stay in proportion to each other.
The section, the contact-side view and the magnet ON/OFF sketch are flat
schematics.

    python3 tools/draw_concept.py  ->  concept.svg next to this folder
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "concept.svg"

W, H = 1600, 1080

# ---------------------------------------------------------------- 3D helpers

AZ, EL = math.radians(22), math.radians(28)
D = (math.sin(AZ) * math.cos(EL), math.cos(AZ) * math.cos(EL), math.sin(EL))
RIGHT = (-math.cos(AZ), math.sin(AZ), 0.0)
UP = (-math.sin(EL) * math.sin(AZ), -math.sin(EL) * math.cos(AZ), math.cos(EL))
LIGHT = (0.30, 0.55, 0.78)
_l = math.sqrt(sum(c * c for c in LIGHT))
LIGHT = tuple(c / _l for c in LIGHT)

CX3, CY3, S3 = 470, 610, 132


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def mul(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def scr(p):
    return (CX3 + S3 * dot(p, RIGHT), CY3 - S3 * dot(p, UP))


def depth(p):
    return dot(p, D)


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(base, n, amb=0.42, spec=0.0):
    k = amb + (1 - amb) * max(0.0, dot(n, LIGHT))
    r, g, b = hexrgb(base)
    hl = spec * max(0.0, dot(n, LIGHT)) ** 12 * 255
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(c * k + hl))) for c in (r, g, b))


def poly(pts3, fill, stroke=None, sw=0.7, extra=""):
    pts = " ".join("%.1f,%.1f" % scr(p) for p in pts3)
    st = stroke or fill
    return '<polygon points="%s" fill="%s" stroke="%s" stroke-width="%.1f" stroke-linejoin="round" %s/>' % (
        pts, fill, st, sw, extra)


def surf(x, phi, rad):
    """Point on a cylinder around the pipe axis; phi=0 is the top, +phi faces the viewer."""
    return (x, rad * math.sin(phi), rad * math.cos(phi))


def radial(phi):
    return (0.0, math.sin(phi), math.cos(phi))


def tangent(phi):
    return (0.0, math.cos(phi), -math.sin(phi))


class Scene:
    def __init__(self):
        self.items = []  # (depth, svg)

    def face(self, pts, n, base, outline=None, sw=0.7, amb=0.42, spec=0.0, force=False):
        if not force and dot(n, D) <= 0:
            return
        z = sum(depth(p) for p in pts) / len(pts)
        fill = shade(base, n, amb, spec)
        self.items.append((z, poly(pts, fill, outline or fill, sw)))

    def flush(self):
        self.items.sort(key=lambda t: t[0])
        out = [s for _, s in self.items]
        self.items = []
        return out


def box(sc, origin, ex, et, en, hx, ht, h0, h1, base, outline):
    """Box sitting on the plate, local frame ex/et/en (en = outward normal)."""
    def v(sx, st, sh):
        return add(origin, mul(ex, sx * hx), mul(et, st * ht), mul(en, sh))
    faces = [
        ([v(-1, -1, h1), v(1, -1, h1), v(1, 1, h1), v(-1, 1, h1)], en),
        ([v(-1, 1, h0), v(1, 1, h0), v(1, 1, h1), v(-1, 1, h1)], et),
        ([v(-1, -1, h0), v(1, -1, h0), v(1, -1, h1), v(-1, -1, h1)], mul(et, -1)),
        ([v(1, -1, h0), v(1, 1, h0), v(1, 1, h1), v(1, -1, h1)], ex),
        ([v(-1, -1, h0), v(-1, 1, h0), v(-1, 1, h1), v(-1, -1, h1)], mul(ex, -1)),
    ]
    for pts, n in faces:
        sc.face(pts, n, base, outline, 1.0)


def cylinder(sc, center, axis, ra, rb, h0, h1, base, outline, n=28, cap=True):
    """Upright cylinder (axis is a unit vector, ra/rb are the in-plane axes)."""
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        p0 = add(mul(ra, math.cos(a0)), mul(rb, math.sin(a0)))
        p1 = add(mul(ra, math.cos(a1)), mul(rb, math.sin(a1)))
        am = (a0 + a1) / 2
        nm = add(mul(ra, math.cos(am)), mul(rb, math.sin(am)))
        ln = math.sqrt(dot(nm, nm))
        nm = mul(nm, 1 / ln)
        pts = [add(center, p0, mul(axis, h0)), add(center, p1, mul(axis, h0)),
               add(center, p1, mul(axis, h1)), add(center, p0, mul(axis, h1))]
        sc.face(pts, nm, base, None, 0.8, spec=0.35)
    if cap:
        pts = [add(center, mul(ra, math.cos(2 * math.pi * i / n)),
                   mul(rb, math.sin(2 * math.pi * i / n)), mul(axis, h1)) for i in range(n)]
        sc.face(pts, axis, base, outline, 1.0, spec=0.2)


def unit(v):
    ln = math.sqrt(dot(v, v))
    return mul(v, 1 / ln)


# ---------------------------------------------------------------- palette

PIPE = "#7f8c84"
PLATE = "#c3ccd6"
PLATE_EDGE = "#56616d"
MAG = "#c8372d"
MAG_BASE = "#4a4f55"
BRASS = "#c9a24a"
STRAP = "#efb52c"
STEEL = "#7a828b"
OIL = "#2a2118"
INK = "#1f2933"
MUTED = "#5b6773"
SEAL = "#141414"
CAVITY = "#8fd0f2"

R = 1.0            # pipe outer radius
PLATE_IN, PLATE_OUT = 1.02, 1.11
PLATE_X = 1.15
PLATE_PHI = math.radians(55)
PIPE_X0, PIPE_X1 = -3.35, 3.15   # +X is drawn to the left


def general_view():
    sc = Scene()
    out = []

    # ground shadow
    a, b = scr((PIPE_X1, 0, -1.25)), scr((PIPE_X0, 0, -1.25))
    out.append('<ellipse cx="%.0f" cy="%.0f" rx="%.0f" ry="26" fill="#000" opacity="0.08"/>' % (
        (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 40, abs(b[0] - a[0]) / 2 + 30))

    # pipe body
    n = 120
    for i in range(n):
        p0, p1 = -math.pi + 2 * math.pi * i / n, -math.pi + 2 * math.pi * (i + 1) / n
        pm = (p0 + p1) / 2
        sc.face([surf(PIPE_X0, p0, R), surf(PIPE_X1, p0, R), surf(PIPE_X1, p1, R), surf(PIPE_X0, p1, R)],
                radial(pm), PIPE, None, 0.9, amb=0.38, spec=0.25)
    out += sc.flush()

    # cut end of the pipe (shows the wall and the oil inside)
    ring = [surf(PIPE_X1, 2 * math.pi * i / 90, R) for i in range(90)]
    inner = [surf(PIPE_X1, 2 * math.pi * i / 90, 0.9) for i in range(90)]
    out.append(poly(ring, "#b9c0c6", "#5d666e", 1.2))
    out.append(poly(inner, OIL, "#000", 1.0))
    c = scr((PIPE_X1, 0.25, 0.35))
    out.append('<ellipse cx="%.0f" cy="%.0f" rx="20" ry="34" fill="#fff" opacity="0.07" transform="rotate(-12 %.0f %.0f)"/>' % (
        c[0], c[1], c[0], c[1]))
    lab = scr((PIPE_X1, 0.0, -0.15))
    out.append('<text x="%.0f" y="%.0f" class="t-oil" text-anchor="middle">нефть</text>' % lab)

    # plate: seal strip, then the aluminium plate
    m = 44
    for i in range(m):
        p0 = -PLATE_PHI + 2 * PLATE_PHI * i / m
        p1 = -PLATE_PHI + 2 * PLATE_PHI * (i + 1) / m
        pm = (p0 + p1) / 2
        sc.face([surf(PLATE_X, p0, PLATE_OUT), surf(-PLATE_X, p0, PLATE_OUT),
                 surf(-PLATE_X, p1, PLATE_OUT), surf(PLATE_X, p1, PLATE_OUT)],
                radial(pm), PLATE, None, 0.9, amb=0.5, spec=0.45)
    # end face at +X
    arc_o = [surf(PLATE_X, -PLATE_PHI + 2 * PLATE_PHI * i / m, PLATE_OUT) for i in range(m + 1)]
    arc_i = [surf(PLATE_X, -PLATE_PHI + 2 * PLATE_PHI * i / m, PLATE_IN) for i in range(m + 1)]
    sc.face(arc_o + arc_i[::-1], (1, 0, 0), "#9aa5b1", PLATE_EDGE, 1.0, force=True)
    arc_s = [surf(PLATE_X, -PLATE_PHI + 2 * PLATE_PHI * i / m, R) for i in range(m + 1)]
    sc.face(arc_i + arc_s[::-1], (1, 0, 0), SEAL, SEAL, 0.6, force=True)
    # front edge face (phi = +55 deg)
    t = tangent(PLATE_PHI)
    sc.face([surf(PLATE_X, PLATE_PHI, PLATE_IN), surf(-PLATE_X, PLATE_PHI, PLATE_IN),
             surf(-PLATE_X, PLATE_PHI, PLATE_OUT), surf(PLATE_X, PLATE_PHI, PLATE_OUT)],
            t, "#9aa5b1", PLATE_EDGE, 1.0, force=True)
    sc.face([surf(PLATE_X, PLATE_PHI, R), surf(-PLATE_X, PLATE_PHI, R),
             surf(-PLATE_X, PLATE_PHI, PLATE_IN), surf(PLATE_X, PLATE_PHI, PLATE_IN)],
            t, SEAL, SEAL, 0.6, force=True)
    out += sc.flush()

    # outline of the plate top
    top = ([surf(PLATE_X, -PLATE_PHI + 2 * PLATE_PHI * i / m, PLATE_OUT) for i in range(m + 1)]
           + [surf(-PLATE_X, PLATE_PHI - 2 * PLATE_PHI * i / m, PLATE_OUT) for i in range(m + 1)])
    out.append(poly(top, "none", PLATE_EDGE, 1.4))

    # straps (mode B) wrapped over the plate
    def strap_r(phi):
        a = abs(phi)
        if a <= PLATE_PHI:
            return PLATE_OUT + 0.012
        if a >= math.radians(72):
            return R + 0.012
        k = (a - PLATE_PHI) / (math.radians(72) - PLATE_PHI)
        return PLATE_OUT + 0.012 + k * (R - PLATE_OUT)
    for xs in (0.52, -0.52):
        k = 140
        for i in range(k):
            p0, p1 = -math.pi + 2 * math.pi * i / k, -math.pi + 2 * math.pi * (i + 1) / k
            pm = (p0 + p1) / 2
            sc.face([surf(xs + 0.07, p0, strap_r(p0)), surf(xs - 0.07, p0, strap_r(p0)),
                     surf(xs - 0.07, p1, strap_r(p1)), surf(xs + 0.07, p1, strap_r(p1))],
                    radial(pm), STRAP, None, 0.8, amb=0.45, spec=0.3)
        out += sc.flush()
        # ratchet buckle on the front
        ph = math.radians(80)
        o = surf(xs, ph, R + 0.015)
        box(sc, o, (1, 0, 0), tangent(ph), radial(ph), 0.11, 0.17, 0.0, 0.09, STEEL, "#3d434a")
        lever_a = scr(add(o, mul(radial(ph), 0.09), mul(tangent(ph), -0.12)))
        lever_b = scr(add(o, mul(radial(ph), 0.16), mul(tangent(ph), 0.16)))
        out += sc.flush()
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#2b2f34" stroke-width="5" stroke-linecap="round"/>' % (
            *lever_a, *lever_b))

    # objects on the plate, painted far to near
    objects = []

    def magnet(x0, ph0):
        o = surf(x0, ph0, PLATE_OUT - 0.01)
        ex, et, en = (1, 0, 0), tangent(ph0), radial(ph0)
        s = Scene()
        box(s, o, ex, et, en, 0.19, 0.17, 0.0, 0.06, MAG_BASE, "#26292d")
        box(s, o, ex, et, en, 0.19, 0.17, 0.06, 0.25, MAG, "#7d1d16")
        svg = s.flush()
        # white plate with the "on/off" marks
        topc = add(o, mul(en, 0.25))
        label = [add(topc, mul(ex, sx * 0.12), mul(et, sy * 0.10)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        svg.append(poly(label, "#f3f3f3", "#7d1d16", 0.8))
        # rotary switch lever, turned to ON (points along the pipe, away from centre)
        piv = add(topc, mul(en, 0.03))
        tip = add(piv, mul(ex, 0.30 * (1 if x0 > 0 else -1)), mul(en, 0.02))
        a, b = scr(piv), scr(tip)
        svg.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#2b2f34"/>' % a)
        svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#2b2f34" stroke-width="7" stroke-linecap="round"/>' % (*a, *b))
        svg.append('<circle cx="%.1f" cy="%.1f" r="8.5" fill="%s" stroke="#2b2f34" stroke-width="2"/>' % (b[0], b[1], MAG))
        objects.append((depth(o), svg))

    for x0 in (0.83, -0.83):
        for ph in (math.radians(-28), math.radians(28)):
            magnet(x0, ph)

    # central boss + pressing screw + valve
    s = Scene()
    up = (0, 0, 1)
    cylinder(s, (0, 0, 0), up, (0.19, 0, 0), (0, 0.19, 0), PLATE_OUT - 0.01, 1.30, "#a9b4bf", "#56616d")
    cyl_hex = [(0.13 * math.cos(math.pi / 3 * i + 0.3), 0.13 * math.sin(math.pi / 3 * i + 0.3)) for i in range(6)]
    for i in range(6):
        (x0, y0), (x1, y1) = cyl_hex[i], cyl_hex[(i + 1) % 6]
        nm = unit(((x0 + x1) / 2, (y0 + y1) / 2, 0))
        s.face([(x0, y0, 1.30), (x1, y1, 1.30), (x1, y1, 1.39), (x0, y0, 1.39)], nm, "#5f666e", "#2e3338", 0.9)
    s.face([(x, y, 1.39) for x, y in cyl_hex], up, "#6d747c", "#2e3338", 0.9)
    box(s, (0, 0, 1.39), (1, 0, 0), (0, 1, 0), up, 0.10, 0.085, 0.0, 0.15, BRASS, "#6d5420")
    # outlet spout towards -X (drawn to the right)
    box(s, (-0.21, 0, 1.43), (1, 0, 0), (0, 1, 0), up, 0.11, 0.04, 0.0, 0.07, BRASS, "#6d5420")
    svg = s.flush()
    # ball-valve handle across the flow = closed
    a = scr((0, 0, 1.56))
    b = scr((0, 0.34, 1.58))
    svg.append('<circle cx="%.1f" cy="%.1f" r="6" fill="#2b2f34"/>' % a)
    svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="9" stroke-linecap="round"/>' % (*a, *b))
    objects.append((depth((0, 0, 1.1)), svg))

    # vacuum nipple (rear) and pressure gauge (front)
    def stem(ph, h, col):
        o = surf(0, ph, PLATE_OUT - 0.01)
        en = radial(ph)
        s = Scene()
        cylinder(s, o, en, (0.05, 0, 0), mul(tangent(ph), 0.05), 0, h, col, "#6d5420", n=16)
        return o, en, s.flush()

    o, en, svg = stem(math.radians(-30), 0.13, BRASS)
    s = Scene()
    cylinder(s, add(o, mul(en, 0.13)), en, (0.075, 0, 0), mul(tangent(math.radians(-30)), 0.075), 0, 0.05,
             "#2f6fb5", "#173a63", n=6)
    svg += s.flush()
    objects.append((depth(o), svg))

    o, en, svg = stem(math.radians(30), 0.15, BRASS)
    g = scr(add(o, mul(en, 0.19)))
    svg.append('<circle cx="%.1f" cy="%.1f" r="27" fill="#2b2f34"/>' % g)
    svg.append('<circle cx="%.1f" cy="%.1f" r="22" fill="#fbfbf7"/>' % g)
    for i in range(9):
        ang = math.radians(-135 + 270 * i / 8)
        p0 = (g[0] + 16 * math.sin(ang), g[1] - 16 * math.cos(ang))
        p1 = (g[0] + 20 * math.sin(ang), g[1] - 20 * math.cos(ang))
        svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1.4"/>' % (*p0, *p1))
    ang = math.radians(-125)
    svg.append('<path d="M %.1f %.1f A 18 18 0 0 1 %.1f %.1f" fill="none" stroke="#2e9e4f" stroke-width="3"/>' % (
        g[0] + 18 * math.sin(math.radians(-135)), g[1] - 18 * math.cos(math.radians(-135)),
        g[0] + 18 * math.sin(math.radians(-70)), g[1] - 18 * math.cos(math.radians(-70))))
    svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="2.2" stroke-linecap="round"/>' % (
        g[0], g[1], g[0] + 15 * math.sin(ang), g[1] - 15 * math.cos(ang)))
    svg.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="#333"/>' % g)
    objects.append((depth(o) + 0.3, svg))

    objects.sort(key=lambda t: t[0])
    for _, svg in objects:
        out += svg

    # drain hose from the spout
    sp = scr((-0.32, 0, 1.465))
    hose = "M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (
        sp[0], sp[1], sp[0] + 120, sp[1] - 40, sp[0] + 210, sp[1] + 10, sp[0] + 250, sp[1] + 95)
    out.append('<path d="%s" fill="none" stroke="#24282c" stroke-width="11" stroke-linecap="round"/>' % hose)
    out.append('<path d="%s" fill="none" stroke="#4b525a" stroke-width="4" stroke-linecap="round" opacity="0.8"/>' % hose)

    anchors = {
        "plate": scr(surf(0.95, math.radians(-50), PLATE_OUT)),
        "magnet": scr(add(surf(-0.83, math.radians(-28), PLATE_OUT), (0, 0, 0.22))),
        "lever": scr(add(surf(0.83, math.radians(-28), PLATE_OUT), (0.30, 0, 0.27))),
        "valve": scr((0, 0.30, 1.58)),
        "screw": scr((0.09, 0.09, 1.36)),
        "hose": (sp[0] + 175, sp[1] - 8),
        "gauge": (g[0] - 22, g[1] + 10),
        "vacuum": scr(add(surf(0, math.radians(-30), PLATE_OUT), mul(radial(math.radians(-30)), 0.17))),
        "strap": scr(surf(0.59, math.radians(95), R + 0.01)),
        "seal": scr(surf(PLATE_X, math.radians(40), R + 0.01)),
        "pipe": scr(surf(-2.6, math.radians(70), R)),
        "buckle": scr(surf(-0.52, math.radians(85), R + 0.1)),
    }
    return out, anchors, sp


def text_w(txt, size):
    return len(txt) * size * (0.70 if size >= 15 else 0.60)


def callout(anchor, tx, ty, lines):
    """Label block at (tx, ty) with a leader from the anchor to its nearer side."""
    ax, ay = anchor
    w = max(text_w(ln, 15 if i == 0 else 13) for i, ln in enumerate(lines))
    if ax >= tx + w:
        jx = tx + w + 6
    else:
        jx = tx - 6
    s = ['<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="lead"/>' % (ax, ay, jx, ty - 5)]
    s.append('<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s"/>' % (ax, ay, INK))
    s.append('<circle cx="%.1f" cy="%.1f" r="1.8" fill="%s"/>' % (jx, ty - 5, INK))
    for i, ln in enumerate(lines):
        cls = "t-lab" if i == 0 else "t-sub"
        s.append('<text x="%.1f" y="%.1f" class="%s">%s</text>' % (tx, ty + i * 17, cls, ln))
    return s


# ---------------------------------------------------------------- section A-A

def section():
    cx, cy = 1275, 395
    rp, wall = 118, 15
    gap, pt = 7, 18          # seal gap and plate thickness, px
    span = 55

    def pt_(rad, ang):
        a = math.radians(ang)
        return (cx + rad * math.sin(a), cy - rad * math.cos(a))

    def band(r0, r1, a0, a1, fill, stroke="none", sw=1, extra=""):
        n = max(4, int(abs(a1 - a0) / 2))
        o = [pt_(r1, a0 + (a1 - a0) * i / n) for i in range(n + 1)]
        i_ = [pt_(r0, a1 - (a1 - a0) * i / n) for i in range(n + 1)]
        pts = " ".join("%.1f,%.1f" % p for p in o + i_)
        return '<polygon points="%s" fill="%s" stroke="%s" stroke-width="%s" %s/>' % (pts, fill, stroke, sw, extra)

    s = []
    s.append('<circle cx="%d" cy="%d" r="%d" fill="url(#hatchPipe)" stroke="%s" stroke-width="1.5"/>' % (cx, cy, rp, "#3f4a44"))
    s.append('<circle cx="%d" cy="%d" r="%d" fill="%s" stroke="#3f4a44" stroke-width="1.5"/>' % (cx, cy, rp - wall, OIL))
    s.append('<text x="%d" y="%d" class="t-oil" text-anchor="middle">нефть, p</text>' % (cx, cy + 40))
    # through-hole in the wall
    s.append(band(rp - wall - 1, rp + 0.5, -2.6, 2.6, OIL))
    # oil-filled space inside the inner seal contour, control cavity outside it
    s.append(band(rp, rp + gap, -20, 20, "#6b4a2b"))
    s.append(band(rp, rp + gap, 20, 45, CAVITY))
    s.append(band(rp, rp + gap, -45, -20, CAVITY))
    # plate
    s.append(band(rp + gap, rp + gap + pt, -span, span, "url(#hatchPlate)", PLATE_EDGE, 1.4))
    # seals: inner contour at +-20, outer at +-45 (and the squeezed edge seal)
    for a in (-20, 20):
        s.append(band(rp, rp + gap, a - 2.6, a + 2.6, SEAL))
    for a in (-45, 45):
        s.append(band(rp, rp + gap, a - 2.6, a + 2.6, SEAL))
    # hollow cone pressed into the hole
    top = cy - (rp + gap)
    s.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s" stroke="#6d5420" stroke-width="1.2"/>' % (
        cx - 15, top, cx + 15, top, cx + 4.5, cy - rp + wall * 0.55, cx - 4.5, cy - rp + wall * 0.55, BRASS))
    # boss, screw, valve
    b0 = cy - (rp + gap + pt)
    s.append('<rect x="%.1f" y="%.1f" width="44" height="34" fill="url(#hatchPlate)" stroke="%s" stroke-width="1.4"/>' % (
        cx - 22, b0 - 34, PLATE_EDGE))
    s.append('<rect x="%.1f" y="%.1f" width="20" height="%.1f" fill="#8b929a" stroke="#3d434a" stroke-width="1"/>' % (
        cx - 10, b0 - 46, 46 + pt))
    for k in range(7):
        yy = b0 - 40 + k * 6
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#3d434a" stroke-width="0.8"/>' % (cx - 10, yy, cx + 10, yy + 3))
    s.append('<rect x="%.1f" y="%.1f" width="34" height="11" fill="#5f666e" stroke="#2e3338" stroke-width="1"/>' % (cx - 17, b0 - 57))
    s.append('<rect x="%.1f" y="%.1f" width="40" height="26" rx="3" fill="%s" stroke="#6d5420" stroke-width="1.2"/>' % (cx - 20, b0 - 83, BRASS))
    s.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#e7cf8d" stroke="#6d5420" stroke-width="1"/>' % (cx, b0 - 70))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#6d5420" stroke-width="2.4"/>' % (cx, b0 - 76, cx, b0 - 64))
    s.append('<rect x="%.1f" y="%.1f" width="40" height="10" fill="%s" stroke="#6d5420" stroke-width="1"/>' % (cx + 20, b0 - 75, BRASS))
    s.append('<path d="M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" fill="none" stroke="#24282c" stroke-width="8" stroke-linecap="round"/>' % (
        cx + 60, b0 - 70, cx + 95, b0 - 72, cx + 110, b0 - 50, cx + 118, b0 - 20))
    s.append('<rect x="%.1f" y="%.1f" width="9" height="9" fill="#c0281f"/>' % (cx - 4.5, b0 - 92))
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="6" stroke-linecap="round"/>' % (
        cx - 30, b0 - 88, cx + 30, b0 - 88))
    # drain channel through cone, screw and valve
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#f4f4f4" stroke-width="3"/>' % (
        cx, cy - rp + wall * 0.55, cx, b0 - 62))
    # gauge (+32 deg) and vacuum nipple (-32 deg) connected to the control cavity
    for ang, kind in ((32, "gauge"), (-32, "vac")):
        p_in = pt_(rp + gap, ang)
        p_out = pt_(rp + gap + pt + 22, ang)
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="6"/>' % (*pt_(rp + gap + pt, ang), *p_out, BRASS))
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#f4f4f4" stroke-width="2"/>' % (*p_in, *p_out))
        if kind == "gauge":
            gc = pt_(rp + gap + pt + 38, ang)
            s.append('<circle cx="%.1f" cy="%.1f" r="16" fill="#fbfbf7" stroke="#2b2f34" stroke-width="3.5"/>' % gc)
            s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="2"/>' % (gc[0], gc[1], gc[0] - 9, gc[1] + 5))
        else:
            gc = pt_(rp + gap + pt + 27, ang)
            s.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#2f6fb5" stroke="#173a63" stroke-width="1.5"/>' % gc)
    # strap of mode B (dashed: lies outside the section plane)
    n = 120
    pts = []
    for i in range(n + 1):
        a = -180 + 360 * i / n
        aa = abs(a)
        if aa <= span:
            rr = rp + gap + pt + 3
        elif aa >= 70:
            rr = rp + 3
        else:
            rr = rp + gap + pt + 3 - (aa - span) / (70 - span) * (gap + pt)
        pts.append(pt_(rr, a))
    s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="3" stroke-dasharray="9 6"/>' % (
        " ".join("%.1f,%.1f" % p for p in pts), "#d99a12"))

    labels = {
        "plate": pt_(rp + gap + pt * 0.5, -50),
        "seal_in": pt_(rp + gap * 0.5, 20),
        "seal_out": pt_(rp + gap * 0.5, -45),
        "cavity": pt_(rp + gap * 0.5, 38),
        "cone": (cx + 6, cy - rp - 2),
        "hole": (cx - 3, cy - rp + wall * 0.5),
        "valve": (cx - 20, b0 - 72),
        "screw": (cx - 17, b0 - 52),
        "gauge": pt_(rp + gap + pt + 52, 32),
        "vac": pt_(rp + gap + pt + 33, -32),
        "strap": pt_(rp + 3, -100),
        "wall": pt_(rp - 7, 120),
        "hose": (cx + 118, b0 - 20),
    }
    return s, labels


# ---------------------------------------------------------------- contact-side view

def bottom_view():
    x0, y0, w, h = 1010, 650, 210, 260    # plate outline: w across the pipe, h along it
    cx, cy = x0 + w / 2, y0 + h / 2
    s = []
    s.append('<rect x="%d" y="%d" width="%d" height="%d" rx="14" fill="url(#hatchPlate)" stroke="%s" stroke-width="1.6"/>' % (
        x0, y0, w, h, PLATE_EDGE))
    # outer and inner seal contours with the control cavity between them
    ow, oh = 0.82 * w, 0.52 * h
    iw, ih = 0.36 * w, 0.24 * h
    s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="16" fill="%s" stroke="%s" stroke-width="7"/>' % (
        cx - ow / 2, cy - oh / 2, ow, oh, CAVITY, SEAL))
    s.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="12" fill="#e9eef3" stroke="%s" stroke-width="7"/>' % (
        cx - iw / 2, cy - ih / 2, iw, ih, SEAL))
    # cone with the drain channel
    s.append('<circle cx="%.1f" cy="%.1f" r="17" fill="%s" stroke="#6d5420" stroke-width="1.4"/>' % (cx, cy, BRASS))
    s.append('<circle cx="%.1f" cy="%.1f" r="6" fill="#e7cf8d" stroke="#6d5420"/>' % (cx, cy))
    s.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (cx, cy, OIL))
    # ports into the cavity (gauge / vacuum)
    for dx in (-1, 1):
        px = cx + dx * (iw / 2 + (ow - iw) / 4)
        s.append('<circle cx="%.1f" cy="%.1f" r="5" fill="#fff" stroke="%s" stroke-width="2"/>' % (px, cy, INK))
    # magnet windows at the corners: two pole shoes split by a brass strip
    for sx in (-1, 1):
        for sy in (-1, 1):
            mx = cx + sx * 0.255 * w
            my = cy + sy * 0.375 * h
            s.append('<rect x="%.1f" y="%.1f" width="46" height="38" rx="4" fill="%s" stroke="#26292d" stroke-width="1.2"/>' % (
                mx - 23, my - 19, MAG_BASE))
            s.append('<rect x="%.1f" y="%.1f" width="5" height="38" fill="%s"/>' % (mx - 2.5, my - 19, BRASS))
            s.append('<text x="%.1f" y="%.1f" class="t-pole" text-anchor="middle">N</text>' % (mx - 12, my + 5))
            s.append('<text x="%.1f" y="%.1f" class="t-pole" text-anchor="middle">S</text>' % (mx + 12, my + 5))
    # pipe axis
    s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" stroke-dasharray="14 4 3 4"/>' % (
        cx, y0 - 14, cx, y0 + h + 14, MUTED))
    s.append('<text x="%.1f" y="%.1f" class="t-sub">ось трубы</text>' % (cx + 6, y0 + h + 12))
    # legend
    rows = [
        ('<rect x="%d" y="%d" width="22" height="12" rx="2" fill="#fff" stroke="%s" stroke-width="4"/>', "уплотнение NBR, два контура"),
        ('<rect x="%d" y="%d" width="22" height="12" rx="2" fill="%s"/>' , "контрольная полость (вакуум)"),
        ('<circle cx="%d" cy="%d" r="7" fill="%s" stroke="#6d5420"/>', "полый конус с каналом"),
        ('<rect x="%d" y="%d" width="22" height="12" rx="2" fill="%s"/>', "окна полюсов магнитов"),
    ]
    ly = y0 + h + 40
    for i, (shape, txt) in enumerate(rows):
        yy = ly + i * 23
        if i == 0:
            s.append(shape % (978, yy - 10, SEAL))
        elif i == 1:
            s.append(shape % (978, yy - 10, CAVITY))
        elif i == 2:
            s.append(shape % (989, yy - 4, BRASS))
        else:
            s.append(shape % (978, yy - 10, MAG_BASE))
        s.append('<text x="1010" y="%d" class="t-small">%s</text>' % (yy, txt))
    labels = {
        "outer": (cx - ow / 2, cy + oh / 2 - 30),
        "inner": (cx + iw / 2, cy - ih / 2 + 14),
        "cavity": (cx + ow / 2 - 14, cy + oh / 2 - 14),
        "magnet": (cx + 0.255 * w + 29, cy - 0.375 * h),
        "cone": (cx + 12, cy + 12),
        "port": (cx + (iw / 2 + (ow - iw) / 4), cy + 5),
    }
    return s, labels


# ---------------------------------------------------------------- magnet ON / OFF

def magnet_switch(x, y, on):
    s = []
    w, h = 170, 92
    # pipe wall under the block
    s.append('<rect x="%d" y="%d" width="%d" height="22" fill="url(#hatchPipe)" stroke="#3f4a44" stroke-width="1.2"/>' % (
        x - 35, y + h + 4, w + 70))
    s.append('<rect x="%d" y="%d" width="%d" height="4" fill="#d9dee3"/>' % (x - 35, y + h, w + 70))
    # pole pieces and brass separators
    s.append('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="#8b929a" stroke="#2e3338" stroke-width="1.4"/>' % (x, y, w, h))
    s.append('<rect x="%d" y="%d" width="10" height="%d" fill="%s"/>' % (x + w / 2 - 5, y, h, BRASS))
    cx, cy, r = x + w / 2, y + h / 2, 30
    s.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="#fff" stroke="#2e3338" stroke-width="1.4"/>' % (cx, cy, r + 3))
    if on:
        s.append('<path d="M %.1f %.1f A %d %d 0 0 0 %.1f %.1f Z" fill="#d6463b"/>' % (cx, cy - r, r, r, cx, cy + r))
        s.append('<path d="M %.1f %.1f A %d %d 0 0 1 %.1f %.1f Z" fill="#3f73c4"/>' % (cx, cy - r, r, r, cx, cy + r))
        s.append('<text x="%.1f" y="%.1f" class="t-pole-w" text-anchor="middle">N</text>' % (cx - 14, cy + 6))
        s.append('<text x="%.1f" y="%.1f" class="t-pole-w" text-anchor="middle">S</text>' % (cx + 14, cy + 6))
        # flux through the pipe wall
        for k, dd in enumerate((14, 28, 42)):
            s.append('<path d="M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" fill="none" stroke="#1f6fd1" stroke-width="2" marker-end="url(#arrowBlue)"/>' % (
                x + 18 + k * 10, y + h - 6, x + 18 + k * 10, y + h + dd, x + w - 18 - k * 10, y + h + dd, x + w - 18 - k * 10, y + h - 6))
    else:
        s.append('<path d="M %.1f %.1f A %d %d 0 0 1 %.1f %.1f Z" fill="#d6463b"/>' % (cx - r, cy, r, r, cx + r, cy))
        s.append('<path d="M %.1f %.1f A %d %d 0 0 0 %.1f %.1f Z" fill="#3f73c4"/>' % (cx - r, cy, r, r, cx + r, cy))
        s.append('<text x="%.1f" y="%.1f" class="t-pole-w" text-anchor="middle">N</text>' % (cx, cy - 9))
        s.append('<text x="%.1f" y="%.1f" class="t-pole-w" text-anchor="middle">S</text>' % (cx, cy + 21))
        # flux closes inside the housing
        for side in (-1, 1):
            s.append('<path d="M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" fill="none" stroke="#1f6fd1" stroke-width="2" marker-end="url(#arrowBlue)"/>' % (
                cx + side * 12, cy - r - 2, cx + side * 62, cy - r - 4, cx + side * 66, cy + r + 4, cx + side * 14, cy + r + 2))
    return s


# ---------------------------------------------------------------- sheet

def main():
    o = []
    o.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif">' % (W, H, W, H))
    o.append("""<defs>
<pattern id="hatchPipe" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <rect width="7" height="7" fill="#8f9b93"/><line x1="0" y1="0" x2="0" y2="7" stroke="#55615a" stroke-width="1.4"/></pattern>
<pattern id="hatchPlate" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
  <rect width="9" height="9" fill="#d3dbe3"/><line x1="0" y1="0" x2="0" y2="9" stroke="#8e9aa6" stroke-width="1"/></pattern>
<marker id="arrowBlue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#1f6fd1"/></marker>
</defs>
<style>
.title{font-size:30px;font-weight:700;fill:%(ink)s}
.subtitle{font-size:16px;fill:%(muted)s}
.ptitle{font-size:18px;font-weight:700;fill:%(ink)s}
.pnote{font-size:13px;fill:%(muted)s}
.t-lab{font-size:15px;font-weight:700;fill:%(ink)s}
.t-sub{font-size:13px;fill:%(muted)s}
.t-small{font-size:13px;fill:%(ink)s}
.t-oil{font-size:15px;font-weight:700;fill:#d9c7a8}
.t-pole{font-size:13px;font-weight:700;fill:#e6e6e6}
.t-pole-w{font-size:15px;font-weight:700;fill:#fff}
.t-num{font-size:13px;font-weight:700;fill:#fff}
.lead{fill:none;stroke:%(ink)s;stroke-width:1.1}
.panel{fill:#f7f8fa;stroke:#d5dbe1;stroke-width:1.2}
</style>""" % {"ink": INK, "muted": MUTED})
    o.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    o.append('<text x="30" y="48" class="title">МВГП — магнитно-вакуумная герметизирующая пластина</text>')
    o.append('<text x="30" y="74" class="subtitle">Концепт-схема устройства для временной герметизации течи на нефтепроводах и резервуарах · без масштаба</text>')

    # panels
    o.append('<rect x="20" y="92" width="920" height="968" rx="14" class="panel"/>')
    o.append('<rect x="955" y="92" width="625" height="490" rx="14" class="panel"/>')
    o.append('<rect x="955" y="596" width="320" height="464" rx="14" class="panel"/>')
    o.append('<rect x="1289" y="596" width="291" height="464" rx="14" class="panel"/>')
    o.append('<text x="42" y="124" class="ptitle">Общий вид на трубе (установлено, режим Б — с ремнями)</text>')
    o.append('<text x="977" y="124" class="ptitle">Разрез А–А поперёк трубы</text>')
    o.append('<text x="977" y="566" class="pnote">Магнитные блоки стоят по углам пластины — вне плоскости разреза</text>')
    o.append('<text x="977" y="628" class="ptitle">Вид со стороны прилегания</text>')
    o.append('<text x="1311" y="628" class="ptitle">Магнитный блок</text>')

    # general view
    gv, A, _ = general_view()
    o += gv
    o += callout(A["lever"], 60, 200, ["Рычаг магнита", "ВЫКЛ → ВКЛ, крест-накрест"])
    o += callout(A["plate"], 60, 290, ["Корпус-пластина", "немагнитный сплав,", "изогнута по радиусу трубы"])
    o += callout(A["vacuum"], 280, 235, ["Штуцер вакуума", "ручной насос"])
    o += callout(A["screw"], 280, 165, ["Нажимной винт", "дожимает конус"])
    o += callout(A["valve"], 610, 165, ["Кран отвода", "закрыт после установки"])
    o += callout(A["magnet"], 640, 235, ["Магнитный блок ×4", "поворотный, переключаемый"])
    o += callout(A["hose"], 700, 320, ["Шланг отвода", "струя уходит в ёмкость,", "пока ставим пластину"])
    o += callout(A["seal"], 60, 420, ["Уплотнитель NBR", "два контура"])
    o += callout(A["gauge"], 60, 900, ["Манометр контрольной полости", "0 — держит; растёт — подтекает"])
    o += callout(A["strap"], 470, 900, ["Стяжной ремень с трещоткой", "режим Б: высокое давление,", "толстая изоляция"])
    o += callout(A["pipe"], 700, 790, ["Трубопровод", "или стенка резервуара"])

    # section
    sec, L = section()
    o += sec
    o += callout(L["valve"], 975, 175, ["Кран отвода"])
    o += callout(L["screw"], 975, 225, ["Нажимной винт"])
    o += callout(L["vac"], 975, 290, ["Вакуум"])
    o += callout(L["plate"], 975, 350, ["Пластина"])
    o += callout(L["seal_out"], 975, 410, ["Внешний контур"])
    o += callout(L["strap"], 975, 520, ["Ремень", "(за плоскостью)"])
    o += callout(L["gauge"], 1440, 175, ["Манометр"])
    o += callout(L["cavity"], 1440, 255, ["Контрольная", "полость"])
    o += callout(L["seal_in"], 1440, 330, ["Внутренний", "контур"])
    o += callout(L["cone"], 1440, 395, ["Полый конус"])
    o += callout(L["hole"], 1440, 450, ["Сквозное", "отверстие"])
    o += callout(L["wall"], 1440, 530, ["Стенка трубы"])

    # contact side
    bv, B = bottom_view()
    o += bv

    # magnet switch
    o.append('<text x="1311" y="660" class="t-lab">ВЫКЛ — примерка</text>')
    o += magnet_switch(1350, 672, on=False)
    o.append('<text x="1311" y="808" class="pnote">поток замкнут в корпусе,</text>')
    o.append('<text x="1311" y="823" class="pnote">пластина двигается свободно</text>')
    o.append('<text x="1311" y="862" class="t-lab">ВКЛ — поворот на 90°</text>')
    o += magnet_switch(1350, 874, on=True)
    o.append('<text x="1311" y="1022" class="pnote">поток уходит в стенку трубы —</text>')
    o.append('<text x="1311" y="1037" class="pnote">прижим</text>')

    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
