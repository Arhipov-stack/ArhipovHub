"""Extra 3D views of the sealing plate for the presentation.

Reuses the scene helpers of draw_concept.py. Everything is modelled in the
pipe frame (X along the axis, Z = outward normal at the plate centre); the
tank view only turns the camera so that this frame stands upright.

    python3 tools/draw_views.py  ->  views/*.svg
"""

import math
from pathlib import Path

import draw_concept as d
from draw_concept import Scene, box, cylinder, surf, radial, tangent, add, mul, unit, poly

OUT = Path(__file__).resolve().parent.parent / "views"

LX = 1.15          # plate half-length along the axis
SW = 1.066         # plate half-width, arc length on the outer face
MAG_X, MAG_S = 0.83, 0.542
PORT_S = 0.581


def set_camera(az, el, cx, cy, s, upright=False):
    """Orthographic camera; upright=True stands the pipe frame on its end (tank wall)."""
    az, el = math.radians(az), math.radians(el)
    D = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    RIGHT = (-math.cos(az), math.sin(az), 0.0)
    UP = (-math.sin(el) * math.sin(az), -math.sin(el) * math.cos(az), math.cos(el))
    L = unit((0.30, 0.55, 0.78))
    if upright:
        # world = (y_l, z_l, x_l)  ->  local = (w_z, w_x, w_y)
        to_local = lambda v: (v[2], v[0], v[1])
        D, RIGHT, UP, L = map(to_local, (D, RIGHT, UP, L))
    d.D, d.RIGHT, d.UP, d.LIGHT = D, RIGHT, UP, L
    d.CX3, d.CY3, d.S3 = cx, cy, s


def center_on(p, x, y):
    """Shift the camera so that 3D point p lands on screen point (x, y)."""
    sx, sy = d.scr(p)
    d.CX3 += x - sx
    d.CY3 += y - sy


class Device:
    def __init__(self, R, magnets_on=True, valve_open=False, straps=False, explode=False):
        self.R = R
        self.on = magnets_on
        self.open = valve_open
        self.straps = straps
        self.explode = explode
        self.ro = R + 0.11                    # plate outer radius
        self.phi = SW / self.ro               # plate half angle
        # lifts of the layers in the exploded view
        self.lift_seal = 0.6 if explode else 0.0
        self.lift_cone = 1.2 if explode else 0.0
        self.lift_plate = 2.35 if explode else 0.0
        self.lift_mag = 3.45 if explode else 0.0

    def up(self, p, h):
        return add(p, (0, 0, h))

    # ------------------------------------------------------------- surfaces
    def pipe(self, x0, x1, color=d.PIPE, cut=True):
        sc, out = Scene(), []
        n = 120
        for i in range(n):
            p0, p1 = -math.pi + 2 * math.pi * i / n, -math.pi + 2 * math.pi * (i + 1) / n
            pm = (p0 + p1) / 2
            sc.face([surf(x0, p0, self.R), surf(x1, p0, self.R), surf(x1, p1, self.R), surf(x0, p1, self.R)],
                    radial(pm), color, None, 0.9, amb=0.38, spec=0.25)
        out += sc.flush()
        if cut:
            ring = [surf(x1, 2 * math.pi * i / 90, self.R) for i in range(90)]
            inner = [surf(x1, 2 * math.pi * i / 90, self.R * 0.9) for i in range(90)]
            out.append(poly(ring, "#b9c0c6", "#5d666e", 1.2))
            out.append(poly(inner, d.OIL, "#000", 1.0))
        return out

    def wall(self, half_h, half_w):
        """Patch of a tank wall: x is vertical, the arc runs sideways."""
        sc, out = Scene(), []
        pw = half_w / self.R
        n = 60
        for i in range(n):
            p0, p1 = -pw + 2 * pw * i / n, -pw + 2 * pw * (i + 1) / n
            pm = (p0 + p1) / 2
            sc.face([surf(-half_h, p0, self.R), surf(half_h, p0, self.R), surf(half_h, p1, self.R), surf(-half_h, p1, self.R)],
                    radial(pm), "#c9d0d8", None, 0.9, amb=0.55, spec=0.1)
        out += sc.flush()
        # weld seams: a horizontal girth seam and two vertical seams
        for xs in (-1.75,):
            pts = [d.scr(surf(xs, -pw + 2 * pw * i / 40, self.R + 0.003)) for i in range(41)]
            out.append('<polyline points="%s" fill="none" stroke="#8d97a2" stroke-width="5"/>' % " ".join("%.1f,%.1f" % p for p in pts))
        for ss in (-2.4, 2.6):
            a, b = d.scr(surf(-half_h, ss / self.R, self.R + 0.003)), d.scr(surf(half_h, ss / self.R, self.R + 0.003))
            out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#8d97a2" stroke-width="5"/>' % (*a, *b))
        return out

    def hole(self):
        pts = [surf(0.075 * math.cos(2 * math.pi * i / 30), 0.075 * math.sin(2 * math.pi * i / 30) / self.R, self.R + 0.002)
               for i in range(30)]
        return [poly(pts, d.OIL, "#000", 1.0)]

    # ------------------------------------------------------------- seals (exploded view)
    def seal_ring(self, hx, hs, rad, lift, width=0.06):
        """Rounded-rectangle sealing contour lying on the surface, hx/hs half sizes."""
        sc = Scene()
        rr = min(hx, hs) * 0.35
        path = []
        for cx_, cs_, a0 in ((hx - rr, hs - rr, 0), (-(hx - rr), hs - rr, 90), (-(hx - rr), -(hs - rr), 180), (hx - rr, -(hs - rr), 270)):
            for k in range(7):
                a = math.radians(a0 + 90 * k / 6)
                path.append((cx_ + rr * math.cos(a), cs_ + rr * math.sin(a)))
        r = self.R + 0.03
        for i in range(len(path)):
            (x0, s0), (x1, s1) = path[i], path[(i + 1) % len(path)]
            # inward normal of the segment in (x, s)
            nx, ns = -(s1 - s0), (x1 - x0)
            ln = math.hypot(nx, ns) or 1
            nx, ns = nx / ln * width, ns / ln * width
            q = [(x0, s0), (x1, s1), (x1 + nx, s1 + ns), (x0 + nx, s0 + ns)]
            pts = [self.up(surf(x, s / r, r), lift) for x, s in q]
            sc.face(pts, (0, 0, 1), d.SEAL, "#000", 0.6, amb=0.6, force=True)
        return sc.flush()

    # ------------------------------------------------------------- plate
    def plate(self):
        sc, out = Scene(), []
        L = self.lift_plate
        R_in, R_out, ph = self.R + 0.02, self.ro, self.phi
        m = 44
        for i in range(m):
            p0, p1 = -ph + 2 * ph * i / m, -ph + 2 * ph * (i + 1) / m
            pm = (p0 + p1) / 2
            sc.face([self.up(surf(LX, p0, R_out), L), self.up(surf(-LX, p0, R_out), L),
                     self.up(surf(-LX, p1, R_out), L), self.up(surf(LX, p1, R_out), L)],
                    radial(pm), d.PLATE, None, 0.9, amb=0.5, spec=0.45)
        # four edge faces, drawn when they face the camera
        arc = lambda x, r: [self.up(surf(x, -ph + 2 * ph * i / m, r), L) for i in range(m + 1)]
        for x, n in ((LX, (1, 0, 0)), (-LX, (-1, 0, 0))):
            sc.face(arc(x, R_out) + arc(x, R_in)[::-1], n, "#9aa5b1", d.PLATE_EDGE, 1.0)
        for p, sgn in ((ph, 1), (-ph, -1)):
            t = mul(tangent(p), sgn)
            sc.face([self.up(surf(LX, p, R_in), L), self.up(surf(-LX, p, R_in), L),
                     self.up(surf(-LX, p, R_out), L), self.up(surf(LX, p, R_out), L)], t, "#9aa5b1", d.PLATE_EDGE, 1.0)
        if not self.explode:
            # the squeezed edge seal between plate and wall
            for x, n in ((LX, (1, 0, 0)), (-LX, (-1, 0, 0))):
                sc.face(arc(x, R_in) + arc(x, self.R)[::-1], n, d.SEAL, d.SEAL, 0.6)
            for p, sgn in ((ph, 1), (-ph, -1)):
                t = mul(tangent(p), sgn)
                sc.face([surf(LX, p, self.R), surf(-LX, p, self.R), surf(-LX, p, R_in), surf(LX, p, R_in)], t, d.SEAL, d.SEAL, 0.6)
        out += sc.flush()
        top = arc(LX, R_out) + arc(-LX, R_out)[::-1]
        out.append(poly(top, "none", d.PLATE_EDGE, 1.4))
        if self.explode:
            # windows for the magnet poles
            for xs in (MAG_X, -MAG_X):
                for ss in (MAG_S, -MAG_S):
                    q = [(xs - 0.19, ss - 0.17), (xs + 0.19, ss - 0.17), (xs + 0.19, ss + 0.17), (xs - 0.19, ss + 0.17)]
                    out.append(poly([self.up(surf(x, s / R_out, R_out + 0.002), L) for x, s in q], "#3a3f45", "#22262a", 1.0))
        return out

    def straps_svg(self):
        sc, out = Scene(), []
        ph = self.phi
        lim = ph + math.radians(17)

        def sr(p):
            a = abs(p)
            if a <= ph:
                return self.ro + 0.012
            if a >= lim:
                return self.R + 0.012
            return self.ro + 0.012 + (a - ph) / (lim - ph) * (self.R - self.ro)
        for xs in (0.52, -0.52):
            k = 140
            for i in range(k):
                p0, p1 = -math.pi + 2 * math.pi * i / k, -math.pi + 2 * math.pi * (i + 1) / k
                pm = (p0 + p1) / 2
                sc.face([surf(xs + 0.07, p0, sr(p0)), surf(xs - 0.07, p0, sr(p0)),
                         surf(xs - 0.07, p1, sr(p1)), surf(xs + 0.07, p1, sr(p1))],
                        radial(pm), d.STRAP, None, 0.8, amb=0.45, spec=0.3)
            out += sc.flush()
            p = math.radians(80)
            o = surf(xs, p, self.R + 0.015)
            box(sc, o, (1, 0, 0), tangent(p), radial(p), 0.11, 0.17, 0.0, 0.09, d.STEEL, "#3d434a")
            out += sc.flush()
        return out

    # ------------------------------------------------------------- parts on the plate
    def parts(self):
        objects = []
        Lp, Lm = self.lift_plate, self.lift_mag
        ro = self.ro

        for xs in (MAG_X, -MAG_X):
            for ss in (MAG_S, -MAG_S):
                ph0 = ss / ro
                o = self.up(surf(xs, ph0, ro - 0.01), Lm)
                ex, et, en = (1, 0, 0), tangent(ph0), radial(ph0)
                s = Scene()
                box(s, o, ex, et, en, 0.19, 0.17, -0.06 if self.explode else 0.0, 0.06, d.MAG_BASE, "#26292d")
                box(s, o, ex, et, en, 0.19, 0.17, 0.06, 0.25, d.MAG, "#7d1d16")
                svg = s.flush()
                topc = add(o, mul(en, 0.25))
                label = [add(topc, mul(ex, sx * 0.12), mul(et, sy * 0.10)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                svg.append(poly(label, "#f3f3f3", "#7d1d16", 0.8))
                piv = add(topc, mul(en, 0.03))
                if self.on:
                    tip = add(piv, mul(ex, 0.30 * (1 if xs > 0 else -1)), mul(en, 0.02))
                else:
                    tip = add(piv, mul(et, 0.27 * (1 if ss > 0 else -1)), mul(en, 0.02))
                a, b = d.scr(piv), d.scr(tip)
                svg.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#2b2f34"/>' % a)
                svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#2b2f34" stroke-width="7" stroke-linecap="round"/>' % (*a, *b))
                svg.append('<circle cx="%.1f" cy="%.1f" r="8.5" fill="%s" stroke="#2b2f34" stroke-width="2"/>' % (b[0], b[1], d.MAG))
                objects.append((d.depth(o), svg))

        # boss, pressing screw, valve (in the local frame at the plate centre)
        z = lambda h: self.R + h + Lp
        s = Scene()
        up = (0, 0, 1)
        cylinder(s, (0, 0, 0), up, (0.19, 0, 0), (0, 0.19, 0), z(0.10), z(0.30), "#a9b4bf", "#56616d")
        hx = [(0.13 * math.cos(math.pi / 3 * i + 0.3), 0.13 * math.sin(math.pi / 3 * i + 0.3)) for i in range(6)]
        for i in range(6):
            (x0, y0), (x1, y1) = hx[i], hx[(i + 1) % 6]
            s.face([(x0, y0, z(0.30)), (x1, y1, z(0.30)), (x1, y1, z(0.39)), (x0, y0, z(0.39))],
                   unit(((x0 + x1) / 2, (y0 + y1) / 2, 0)), "#5f666e", "#2e3338", 0.9)
        s.face([(x, y, z(0.39)) for x, y in hx], up, "#6d747c", "#2e3338", 0.9)
        box(s, (0, 0, z(0.39)), (1, 0, 0), (0, 1, 0), up, 0.10, 0.085, 0.0, 0.15, d.BRASS, "#6d5420")
        box(s, (-0.21, 0, z(0.43)), (1, 0, 0), (0, 1, 0), up, 0.11, 0.04, 0.0, 0.07, d.BRASS, "#6d5420")
        svg = s.flush()
        a = d.scr((0, 0, z(0.56)))
        b = d.scr((-0.34, 0, z(0.58)) if self.open else (0, 0.34, z(0.58)))
        svg.append('<circle cx="%.1f" cy="%.1f" r="6" fill="#2b2f34"/>' % a)
        svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="9" stroke-linecap="round"/>' % (*a, *b))
        objects.append((d.depth((0, 0, z(0.1))), svg))

        # vacuum nipple and gauge
        def stem(ss, h):
            ph0 = ss / ro
            o = self.up(surf(0, ph0, ro - 0.01), Lp)
            en = radial(ph0)
            s = Scene()
            cylinder(s, o, en, (0.05, 0, 0), mul(tangent(ph0), 0.05), 0, h, d.BRASS, "#6d5420", n=16)
            return o, en, ph0, s.flush()

        o, en, ph0, svg = stem(-PORT_S, 0.13)
        s = Scene()
        cylinder(s, add(o, mul(en, 0.13)), en, (0.075, 0, 0), mul(tangent(ph0), 0.075), 0, 0.05, "#2f6fb5", "#173a63", n=6)
        svg += s.flush()
        self.vac_tip = d.scr(add(o, mul(en, 0.18)))
        objects.append((d.depth(o), svg))

        o, en, ph0, svg = stem(PORT_S, 0.15)
        g = d.scr(add(o, mul(en, 0.19)))
        svg.append('<circle cx="%.1f" cy="%.1f" r="27" fill="#2b2f34"/>' % g)
        svg.append('<circle cx="%.1f" cy="%.1f" r="22" fill="#fbfbf7"/>' % g)
        for i in range(9):
            ang = math.radians(-135 + 270 * i / 8)
            svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333" stroke-width="1.4"/>' % (
                g[0] + 16 * math.sin(ang), g[1] - 16 * math.cos(ang), g[0] + 20 * math.sin(ang), g[1] - 20 * math.cos(ang)))
        ang = math.radians(-125)
        svg.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c0281f" stroke-width="2.2" stroke-linecap="round"/>' % (
            g[0], g[1], g[0] + 15 * math.sin(ang), g[1] - 15 * math.cos(ang)))
        svg.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="#333"/>' % g)
        self.gauge = g
        objects.append((d.depth(o) + 0.3, svg))

        objects.sort(key=lambda t: t[0])
        out = []
        for _, svg in objects:
            out += svg
        self.spout = d.scr((-0.32, 0, z(0.465)))
        return out

    def cone(self):
        """Hollow brass cone, drawn on its own in the exploded view."""
        sc = Scene()
        L = self.lift_cone
        n = 24
        h0, h1, r0, r1 = -0.02, 0.16, 0.035, 0.11
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            am = (a0 + a1) / 2
            nm = unit((math.cos(am), math.sin(am), -(r1 - r0) / (h1 - h0)))
            p = lambda a, r, h: (r * math.cos(a), r * math.sin(a), self.R + h + L)
            sc.face([p(a0, r0, h0), p(a1, r0, h0), p(a1, r1, h1), p(a0, r1, h1)], nm, d.BRASS, None, 0.8, spec=0.4)
        sc.face([(r1 * math.cos(2 * math.pi * i / n), r1 * math.sin(2 * math.pi * i / n), self.R + h1 + L) for i in range(n)],
                (0, 0, 1), "#d8b867", "#6d5420", 1.0)
        out = sc.flush()
        c = d.scr((0, 0, self.R + h1 + L))
        out.append('<ellipse cx="%.1f" cy="%.1f" rx="5" ry="3" fill="%s"/>' % (c[0], c[1], d.OIL))
        return out

    def hose(self, dx=1, flowing=False):
        sp = self.spout
        if not flowing:
            path = "M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (
                sp[0], sp[1], sp[0] + dx * 120, sp[1] - 40, sp[0] + dx * 210, sp[1] + 10, sp[0] + dx * 250, sp[1] + 95)
            return ['<path d="%s" fill="none" stroke="#24282c" stroke-width="11" stroke-linecap="round"/>' % path,
                    '<path d="%s" fill="none" stroke="#4b525a" stroke-width="4" stroke-linecap="round" opacity="0.8"/>' % path]
        # long hose running down in front of the pipe into a canister
        end = (sp[0] + dx * 330, sp[1] + 300)
        path = "M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (
            sp[0], sp[1], sp[0] + dx * 200, sp[1] - 60, sp[0] + dx * 340, sp[1] + 40, end[0], end[1])
        cx, cy = end[0], end[1] + 18
        return ['<path d="%s" fill="none" stroke="#24282c" stroke-width="11" stroke-linecap="round"/>' % path,
                '<path d="%s" fill="none" stroke="#4b525a" stroke-width="4" stroke-linecap="round" opacity="0.8"/>' % path,
                '<rect x="%.1f" y="%.1f" width="96" height="66" rx="8" fill="#5d7a3a" stroke="#2f3f1d" stroke-width="2.5"/>' % (cx - 48, cy),
                '<rect x="%.1f" y="%.1f" width="96" height="14" fill="#3b2a17"/>' % (cx - 48, cy + 6),
                '<text x="%.1f" y="%.1f" font-size="15" font-weight="700" fill="#eef3e6" text-anchor="middle">ёмкость</text>' % (cx, cy + 46)]


def svg_doc(body, vb, w, h, extra_style=""):
    return "\n".join(['<svg xmlns="http://www.w3.org/2000/svg" viewBox="%s" width="%d" height="%d" font-family="DejaVu Sans, Arial, sans-serif">' % (vb, w, h),
                      '<style>.lab{font-size:30px;font-weight:700;fill:#1f2933}.sub{font-size:24px;fill:#5b6773}%s</style>' % extra_style,
                      '<rect x="-5000" y="-5000" width="20000" height="20000" fill="#ffffff"/>'] + body + ["</svg>"])


def label(anchor, tx, ty, text, sub=None, anchor_side="start"):
    ax, ay = anchor
    jx = tx - 10 if anchor_side == "start" else tx + 10
    s = ['<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f2933" stroke-width="2"/>' % (ax, ay, jx, ty - 9),
         '<circle cx="%.1f" cy="%.1f" r="5" fill="#1f2933"/>' % (ax, ay),
         '<text x="%.1f" y="%.1f" class="lab" text-anchor="%s">%s</text>' % (tx, ty, anchor_side, text)]
    if sub:
        s.append('<text x="%.1f" y="%.1f" class="sub" text-anchor="%s">%s</text>' % (tx, ty + 30, anchor_side, sub))
    return s


def shadow(x0, x1, z, dy=40):
    a, b = d.scr((x1, 0, z)), d.scr((x0, 0, z))
    return ['<ellipse cx="%.0f" cy="%.0f" rx="%.0f" ry="26" fill="#000" opacity="0.08"/>' % (
        (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, abs(b[0] - a[0]) / 2 + 30)]


def view_pipe(on=True, open_=False, flowing=False, straps=False, name="pipe"):
    set_camera(22, 28, 470, 610, 132)
    dv = Device(1.0, magnets_on=on, valve_open=open_, straps=straps)
    body = shadow(-3.35, 3.15, -1.25)
    body += dv.pipe(-3.35, 3.15)
    body += dv.plate()
    if straps:
        body += dv.straps_svg()
    body += dv.parts()
    body += dv.hose(1, flowing)
    return dv, body


def main():
    OUT.mkdir(exist_ok=True)

    # 1. mode A on the pipe, no straps
    dv, body = view_pipe(name="mode_a")
    (OUT / "pipe_mode_a.svg").write_text(svg_doc(body, "30 330 900 520", 1800, 1040), encoding="utf-8")

    # 2. installation frames: fitting (magnets off, valve open, oil into the hose) and done
    dv, body = view_pipe(on=False, open_=True, flowing=True)
    pump = dv.vac_tip
    body.append('<path d="M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" fill="none" stroke="#2f6fb5" stroke-width="5"/>' % (
        pump[0], pump[1], pump[0] - 60, pump[1] - 90, pump[0] - 160, pump[1] - 80, pump[0] - 210, pump[1] - 40))
    px, py = pump[0] - 250, pump[1] - 40
    body.append('<rect x="%.1f" y="%.1f" width="80" height="34" rx="10" fill="#2f6fb5" stroke="#173a63" stroke-width="2"/>' % (px - 40, py - 17))
    body.append('<rect x="%.1f" y="%.1f" width="70" height="12" rx="6" fill="#2b2f34" transform="rotate(-25 %.1f %.1f)"/>' % (px - 110, py - 20, px - 40, py))
    (OUT / "step_fit.svg").write_text(svg_doc(body, "30 250 900 610", 1800, 1220), encoding="utf-8")
    dv, body = view_pipe(on=True, open_=False, flowing=False)
    (OUT / "step_done.svg").write_text(svg_doc(body, "30 250 900 610", 1800, 1220), encoding="utf-8")

    # 3. tank wall: the same plate on a vertical, almost flat wall
    set_camera(28, 8, 600, 560, 150, upright=True)
    center_on((0, 0, 14.0), 600, 560)
    dv = Device(14.0, magnets_on=True, valve_open=False)
    body = dv.wall(2.6, 3.4)
    body += dv.plate()
    body += dv.parts()
    sp = dv.spout
    body += ['<path d="M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" fill="none" stroke="#24282c" stroke-width="11" stroke-linecap="round"/>' % (
        sp[0], sp[1], sp[0] + 60, sp[1] + 40, sp[0] + 40, sp[1] + 200, sp[0] + 90, sp[1] + 330)]
    (OUT / "tank.svg").write_text(svg_doc(body, "120 120 960 860", 1440, 1290), encoding="utf-8")

    # 4. exploded view on the pipe
    set_camera(22, 22, 470, 760, 185)
    center_on((0, 0, 2.6), 680, 470)
    dv = Device(1.0, magnets_on=True, explode=True)
    body = []
    body += dv.pipe(-2.0, 2.0)
    body += dv.hole()
    body += dv.seal_ring(0.62, 0.78, 1.0, dv.lift_seal)
    body += dv.seal_ring(0.30, 0.36, 1.0, dv.lift_seal)
    body += dv.cone()
    body += dv.plate()
    body += dv.parts()
    # assembly axes
    for p0, p1 in (((0, 0, 1.0), (0, 0, 1.0 + dv.lift_plate)),):
        a, b = d.scr(p0), d.scr(p1)
        body.insert(len(body) - 1, '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f6fd1" stroke-width="2.5" stroke-dasharray="10 7"/>' % (*a, *b))
    for xs in (MAG_X, -MAG_X):
        for ss in (MAG_S, -MAG_S):
            ph0 = ss / dv.ro
            a = d.scr(dv.up(surf(xs, ph0, dv.ro), dv.lift_plate))
            b = d.scr(dv.up(surf(xs, ph0, dv.ro), dv.lift_mag - 0.06))
            body.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f6fd1" stroke-width="2.5" stroke-dasharray="10 7"/>' % (*a, *b))
    L = []
    L += label(d.scr(dv.up(surf(-MAG_X - 0.19, -MAG_S / dv.ro, dv.ro), dv.lift_mag + 0.2)), 1250, 120, "Магнитные блоки ×4", "поворотные, ВЫКЛ / ВКЛ")
    L += label(d.scr(dv.up(surf(-LX, -0.4 / dv.ro, dv.ro), dv.lift_plate)), 1250, 330, "Корпус-пластина", "немагнитная, окна под полюса")
    L += label(d.scr((0, 0.11, 1.0 + dv.lift_cone + 0.16)), 1250, 560, "Полый конус", "латунь, с каналом отвода")
    L += label(d.scr(dv.up(surf(-0.62, 0.55, 1.03), dv.lift_seal)), 1250, 700, "Два контура уплотнения", "NBR: внутренний и внешний")
    L += label(d.scr(surf(-1.4, math.radians(60), 1.0)), 1250, 880, "Труба со свищом")
    L += label(dv.gauge, 330, 420, "Манометр", None, "end")
    L += label(d.scr((0.05, 0.05, 1.0 + dv.lift_plate + 0.42)), 330, 250, "Кран и винт", None, "end")
    body += L
    (OUT / "exploded.svg").write_text(svg_doc(body, "0 20 1900 1060", 1900, 1060), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
