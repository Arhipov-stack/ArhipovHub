#!/usr/bin/env python3
"""
Builds the oni-and-dragon stand for Yandex Station Midi as a printable STL.

    python3 tools/build_stand.py               # final, 0.3 mm voxels
    python3 tools/build_stand.py --res 0.6     # quick draft

The whole stand is one signed distance field sampled on a voxel grid and
turned into a mesh with marching cubes. That is what lets the mask relief,
the scaled dragon body and the plinth melt into each other with fillets
instead of being boolean-glued.

Before meshing, every layer is checked against the one above it: anything
that would hang in the air at steeper than 45 degrees gets a 45-degree
wedge of material under it (see `self_support`). The stand therefore prints
upright with no supports at all.

Axes: x to the right, y away from the viewer (the back of the shelf), z up.
The speaker stands centred on x = y = 0. Millimetres throughout.

Needs numpy, scipy, scikit-image, trimesh, fast-simplification.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- parameters

# Yandex Station Midi: 96 x 96 x 110 mm, USB-C power on the back.
SPK = 96.0
SPK_CLEAR = 1.0          # gap around the speaker in the pocket, per side
POCKET_R = 18.0          # pocket corner radius: fits a rounded square or a 96 mm disc
POCKET_DEPTH = 3.0

HP = 10.0                # plinth height
FRONT_Y = -54.0          # front edge of the plinth (speaker front is at -48)
BACK_Y = 54.0            # back edge behind the speaker (speaker back is at +48)

CABLE_X = 0.0            # where the USB-C plug leaves the back of the speaker
CABLE_W = 16.0           # width of the cut-out in the back rim

# The pillar on the left that carries the mask and the dragon.
ST_X0, ST_X1 = -104.0, -68.0
ST_Y0, ST_Y1 = -34.0, 0.0      # ST_Y0 is the face the mask hangs on
ST_TOP = 104.0
MASK_X = 0.5 * (ST_X0 + ST_X1)
MASK_Z = 64.0                  # centre of the mask
WING_X0 = -121.0               # left end of the plinth


# ------------------------------------------------------------------ helpers

def vmax(arrs):
    """Elementwise max of arrays that only broadcast together."""
    out = arrs[0]
    for a in arrs[1:]:
        out = np.maximum(out, a)
    return out


def vmin(arrs):
    out = arrs[0]
    for a in arrs[1:]:
        out = np.minimum(out, a)
    return out


def smin(a, b, k):
    """Polynomial smooth minimum: union with a fillet of size ~k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_rrect(px, py, cx, cy, hx, hy, r):
    """2D rounded rectangle."""
    qx = np.abs(px - cx) - (hx - r)
    qy = np.abs(py - cy) - (hy - r)
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return out + np.minimum(np.maximum(qx, qy), 0) - r


def sd_ellipsoid(px, py, pz, c, r):
    """Cheap ellipsoid distance; good enough near the surface."""
    k0 = np.sqrt(((px - c[0]) / r[0]) ** 2 + ((py - c[1]) / r[1]) ** 2 +
                 ((pz - c[2]) / r[2]) ** 2)
    return (k0 - 1.0) * min(r)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def dist_to_polyline(px, py, pts):
    """2D distance from points to a polyline given as (n, 2)."""
    d = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax, by - ay
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.minimum(d, np.hypot(px - ax - t * vx, py - ay - t * vy))
    return d


def spiral_groove(px, py, cx, cy, r0, pitch, turns, sign=1.0):
    """Distance to an Archimedean spiral r = r0 + pitch * theta / 2pi."""
    dx, dy = px - cx, sign * (py - cy)
    rho = np.hypot(dx, dy)
    phi = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    best = np.full(px.shape, np.inf)
    for n in range(int(np.ceil(turns)) + 1):
        theta = phi + 2 * np.pi * n
        rr = r0 + pitch * theta / (2 * np.pi)
        ok = theta <= turns * 2 * np.pi
        best = np.where(ok, np.minimum(best, np.abs(rho - rr)), best)
    # the open outer end and the inner start get round caps
    return np.minimum(best, np.hypot(dx - r0, dy))


# -------------------------------------------------------------------- grid

class Field:
    def __init__(self, lo, hi, h):
        self.h = h
        self.lo = np.asarray(lo, float)
        self.n = np.ceil((np.asarray(hi, float) - self.lo) / h).astype(int) + 1
        self.ax = [self.lo[i] + h * np.arange(self.n[i]) for i in range(3)]
        self.F = np.full(self.n, 50.0, np.float32)

    def window(self, blo, bhi):
        sl = []
        for i in range(3):
            a = max(0, int(np.floor((blo[i] - self.lo[i]) / self.h)))
            b = min(self.n[i], int(np.ceil((bhi[i] - self.lo[i]) / self.h)) + 1)
            sl.append(slice(a, b))
        sl = tuple(sl)
        X, Y, Z = np.meshgrid(self.ax[0][sl[0]], self.ax[1][sl[1]],
                              self.ax[2][sl[2]], indexing="ij", sparse=True)
        return sl, X, Y, Z

    def union(self, blo, bhi, fn, k=0.0):
        sl, X, Y, Z = self.window(blo, bhi)
        d = np.broadcast_to(fn(X, Y, Z), self.F[sl].shape)
        self.F[sl] = smin(self.F[sl], d.astype(np.float32), k)

    def cut(self, blo, bhi, fn, k=0.0):
        sl, X, Y, Z = self.window(blo, bhi)
        d = np.broadcast_to(fn(X, Y, Z), self.F[sl].shape)
        self.F[sl] = smax(self.F[sl], -d.astype(np.float32), k)

    def apply(self, blo, bhi, fn):
        """fn(F_window, X, Y, Z) -> new F_window."""
        sl, X, Y, Z = self.window(blo, bhi)
        self.F[sl] = fn(self.F[sl], X, Y, Z).astype(np.float32)


# ----------------------------------------------------------------- sweeps

class Sweep:
    """A tube along a smooth 3D curve with a radius profile and a local frame.

    `up` gives the direction the creature's back faces; the frame's N is the
    part of `up` perpendicular to the curve, so dorsal fins stand in a
    vertical plane wherever possible and print without overhangs.
    """

    def __init__(self, ctrl, radius_fn, step=0.2, up=None, post=None):
        ctrl = np.asarray(ctrl, float)
        chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))]
        cs = CubicSpline(chord, ctrl, bc_type="natural")
        tt = np.linspace(0, chord[-1], int(chord[-1] / 0.05) + 2)
        dense = cs(tt)
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        arc = np.r_[0, np.cumsum(seg)]
        self.length = arc[-1]
        s = np.linspace(0, self.length, int(self.length / step) + 2)
        self.C = np.stack([np.interp(s, arc, dense[:, i]) for i in range(3)], 1)
        self.s = s
        self.t = s / self.length
        self.r = radius_fn(self.t)
        if post is not None:
            post(self)
        T = np.gradient(self.C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        U = np.tile([0, 0, 1.0], (len(s), 1)) if up is None else up(self.C, T)
        N = U - (U * T).sum(1, keepdims=True) * T
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        self.T, self.N, self.B = T, N, np.cross(T, N)
        self.tree = cKDTree(self.C)

    def bbox(self, pad):
        return self.C.min(0) - self.r.max() - pad, self.C.max(0) + self.r.max() + pad

    def local(self, X, Y, Z, reach):
        """Nearest-sample local coordinates for every grid point in reach."""
        shape = np.broadcast_shapes(X.shape, Y.shape, Z.shape)
        P = np.stack([np.broadcast_to(a, shape).ravel() for a in (X, Y, Z)], 1)
        dist, idx = self.tree.query(P, distance_upper_bound=reach, workers=-1)
        ok = idx < len(self.C)
        i = idx[ok]
        d = P[ok] - self.C[i]
        a = (d * self.N[i]).sum(1)
        b = (d * self.B[i]).sum(1)
        return shape, ok, i, a, b, np.linalg.norm(d, axis=1)


def tube_fn(sw, scales=0.0, fin=None, cap=None):
    """Distance function of a sweep, optionally with scales and a fin."""
    fin_h = 0.0 if fin is None else fin["h"].max()
    reach = sw.r.max() + fin_h + 2.0

    def fn(X, Y, Z):
        shape, ok, i, a, b, rho = sw.local(X, Y, Z, reach)
        out = np.full(int(np.prod(shape)), 50.0, np.float32)
        r = sw.r[i]
        d = rho - r
        if scales > 0:
            theta = np.arctan2(b, a)                  # 0 = the creature's back
            u = sw.s[i] / 2.6
            v = theta * r / 2.6
            dA = np.hypot(u - np.round(u), v - np.round(v))
            dB = np.hypot(u + .5 - np.round(u + .5), v + .5 - np.round(v + .5))
            dm = np.minimum(dA, dB)
            bump = np.clip(1 - (dm / 0.62) ** 2, 0, 1)
            # scales on back and flanks, plain belly plates underneath
            side = smoothstep(2.4, 1.9, np.abs(theta))
            belly = 0.35 * np.clip(1 - np.abs(np.mod(sw.s[i] / 2.2, 1) - .5) * 4, 0, 1)
            amp = scales * np.clip(r / 3.0, 0, 1)
            d = d - amp * (side * bump + (1 - side) * belly)
        if fin is not None:
            h = fin["h"][i]
            th = fin["th"] * (1.15 - 0.75 * np.clip((a - r) / np.maximum(h, .1), 0, 1))
            f = vmax([np.abs(b) - th / 2, a - (r + h), (r - 1.2) - a])
            f = np.where(h > 0.2, f, 50.0)
            d = np.minimum(d, f)
        out[ok] = d
        return out.reshape(shape)

    return fn


def sawtooth(s, period, lean=0.75):
    """0..1 teeth whose steep face points toward increasing s."""
    x = np.mod(s / period, 1.0)
    return np.where(x < lean, x / lean, 1 - (x - lean) / (1 - lean)) ** 0.9


# ------------------------------------------------------------------ plinth

def plinth_2d(x, y):
    spk = sd_rrect(x, y, 0.0, 0.0, 54.0, 54.0, 12.0)
    wy0, wy1 = FRONT_Y, 14.0
    wing = sd_rrect(x, y, 0.5 * (WING_X0 - 20.0), 0.5 * (wy0 + wy1),
                    0.5 * (-20.0 - WING_X0), 0.5 * (wy1 - wy0), 10.0)
    return np.minimum(spk, wing)


def plinth(X, Y, Z):
    d2 = plinth_2d(X, Y)
    top_ch, bot_ch = 1.6, 0.6
    d = vmax([d2 + 0 * Z, Z - HP, -Z])
    d = np.maximum(d, (d2 + (Z - HP) + top_ch) / np.sqrt(2))
    d = np.maximum(d, (d2 - Z + bot_ch) / np.sqrt(2))
    return d


def seigaiha(u, v, R=3.4):
    """Distance to the arcs of a seigaiha (blue ocean waves) pattern."""
    ring = np.full(np.broadcast_shapes(u.shape, v.shape), np.inf)
    taken = np.zeros(ring.shape, bool)
    for j in range(-1, 8):                        # lower rows overlap upper ones
        vj = 0.4 + j * R * 0.5
        off = (j % 2) * R
        cx = np.round((u - off) / (2 * R)) * 2 * R + off
        dist = np.hypot(u - cx, v - vj)
        inside = (dist < R) & ~taken
        rr = vmin([np.abs(dist - R * f) for f in (0.98, 0.70, 0.42)])
        rr = np.minimum(rr, dist - R * 0.12)
        ring = np.where(inside, rr, ring)
        taken |= inside
    return ring


def front_waves(X, Y, Z):
    """Solid to remove: wave grooves on the flat front face of the plinth."""
    g = seigaiha(X - 7.0, Z - 1.4) - 0.32
    band = vmax([Z - (HP - 2.1), 1.3 - Z, X - 40.0, -108.0 - X])
    depth = np.maximum(Y - (FRONT_Y + 0.55), (FRONT_Y - 2.0) - Y)
    return vmax([g, band, depth])


# -------------------------------------------------------------- oni mask

def mask_silhouette(X, Z):
    """Outline of the mask: horned brow with a dip, broad cheeks, narrow chin."""
    aX = np.abs(X)
    wz = np.interp(Z, [-37, -31, -21, -8, 6, 20, 30, 37],
                   [6, 11, 18, 23.5, 26, 25.5, 23, 19])
    top = 37.0 - 5.5 * np.exp(-(X / 6.5) ** 2)
    return vmax([aX - wz, Z - top, -37.0 - Z])


def mask_height(x, z):
    """How far the mask stands proud of the pillar face, mm.

    The mask is 52 x 74 mm, wider than the pillar, centred on
    (MASK_X, MASK_Z). Modelled on the hannya/oni mask in the reference:
    heavy angry brows with a trim line, spirals on the forehead, a braided
    nose, an open grin with two rows of teeth and long fangs.
    """
    X = x - MASK_X
    Z = z - MASK_Z
    aX = np.abs(X)

    H = 3.5 + 7.5 * np.clip(1 - (X / 28.0) ** 2, 0, 1) ** 0.8 * \
        np.clip(1 - ((Z - 3) / 42.0) ** 2, 0, 1) ** 0.5
    # cheekbones
    H += 2.0 * np.exp(-((aX - 17.0) ** 2 + (Z + 3) ** 2) / 50.0)
    # forehead bulges above the eyes
    H += 1.6 * np.exp(-((aX - 11.5) ** 2 + (Z - 24) ** 2) / 45.0)

    # brows, angry: low at the nose, high at the temples
    brow = np.array([[2.5, 8.0], [8.5, 11.5], [15.5, 16.0], [23.0, 18.0]])
    H += 3.6 * np.exp(-(dist_to_polyline(aX, Z, brow) / 3.0) ** 2)
    trim = dist_to_polyline(aX, Z, brow + [0, 3.9])
    H -= 0.9 * smoothstep(0.85, 0.5, trim)

    # spiral swirls on the forehead
    sw = spiral_groove(aX, Z, 12.0, 27.0, 1.0, 2.0, 2.6)
    H -= 0.9 * smoothstep(0.6, 0.35, sw) * smoothstep(7.8, 6.8, np.hypot(aX - 12, Z - 27))
    # furrows between them
    fur = np.abs(aX - 1.7) + np.maximum(0, np.abs(Z - 26) - 6)
    H -= 0.8 * smoothstep(0.65, 0.35, fur)

    # cheek folds
    for off in (0.0, 3.6):
        fold = dist_to_polyline(aX, Z, np.array([[15.0, 1.0 - off], [20.0, -6 - off],
                                                 [23.5, -13 - off]]))
        H += 1.5 * np.exp(-(fold / 1.2) ** 2)

    # nose: bridge with braided chevrons, wide flared nostrils
    nose_w = np.interp(Z, [-10, 0, 9], [8.0, 5.0, 3.0])
    bridge = smoothstep(0.0, 1.0, (nose_w - aX) / 2.2) * smoothstep(-12, -7, Z) * \
        smoothstep(10, 7, Z)
    H += 3.2 * bridge
    chev = np.mod(Z + 0.75 * aX, 2.5)
    H -= 0.7 * smoothstep(0.6, 0.3, np.abs(chev - 1.25) - 0.45) * bridge * smoothstep(-6, -4, Z)
    for cx, cz, r, peak in ((5.4, -8.6, 4.0, 14.2), (0.0, -7.5, 3.8, 15.5)):
        dn = np.hypot(aX - cx, Z - cz)
        H = np.maximum(H, np.where(dn < r, peak - 3.5 + 3.5 * np.sqrt(
            np.clip(1 - (dn / r) ** 2, 0, 1)), -50))
    nost = np.hypot(aX - 5.0, (Z + 9.9) * 1.4)
    H = np.where(nost < 1.5, np.minimum(H, 11.0), H)

    # creases from the nostrils round the mouth
    crease = dist_to_polyline(aX, Z, np.array([[9.0, -9.0], [14.5, -15.0], [16.5, -21.5]]))
    H += 1.6 * np.exp(-(crease / 1.3) ** 2)

    # mouth: thick lips, deep cavity, two rows of teeth
    mouth = sd_rrect(X, Z - 0.018 * X * X, 0.0, -19.5, 13.5, 6.0, 4.5)   # a grin
    H += 2.2 * np.exp(-((mouth + 0.3) / 1.6) ** 2)
    H = np.where(mouth < 0, 2.0, H)
    # lower teeth grow out of the lower lip so nothing hangs in the air
    for zt0, zt1 in ((-14.6, -17.0), (-21.0, -28.0)):
        cx = np.round((X + 1.6) / 3.2) * 3.2 - 1.6
        tooth = sd_rrect(X, Z - 0.018 * X * X, cx, 0.5 * (zt0 + zt1), 1.3,
                         0.5 * (zt0 - zt1), 0.6)
        tooth = np.maximum(tooth, mouth + 0.5)
        H = np.where(tooth < 0, 6.0 + 1.2 * np.sqrt(np.clip(-tooth / 1.1, 0, 1)), H)

    # chin ornament
    chin = spiral_groove(X, Z, 0.0, -31.0, 0.6, 1.7, 1.8)
    H -= 0.8 * smoothstep(0.55, 0.3, chin) * smoothstep(5.0, 4.0, np.hypot(X, Z + 31))

    # eyes: deep slanted almonds (paint them black and they read as holes)
    ex, ez = aX - 9.8, Z - 4.8
    ang = np.radians(-17)
    eu = ex * np.cos(ang) - ez * np.sin(ang)
    ev = ex * np.sin(ang) + ez * np.cos(ang)
    eye = np.sqrt((eu / 7.2) ** 2 + (ev / (3.0 - 0.07 * eu)) ** 2) - 1
    H = np.where(eye < 0, np.minimum(H, 3.5), H)
    H -= 1.2 * smoothstep(0.3, 0.0, eye) * (eye >= 0)
    return H


def oni_mask(X, Y, Z):
    xs = np.asarray(X)[:, 0, 0]
    zs = np.asarray(Z)[0, 0, :]
    Xg, Zg = xs[:, None] - MASK_X, zs[None, :] - MASK_Z
    H = mask_height(xs[:, None], zs[None, :])[:, None, :]
    sil = mask_silhouette(Xg, Zg)[:, None, :]
    front = (ST_Y0 - H) - Y
    back = Y - (ST_Y0 + 5.0)
    return vmax([smax(sil, front, 3.0), back])


def pillar(X, Y, Z):
    core = sd_rrect(X, Y, MASK_X, 0.5 * (ST_Y0 + ST_Y1),
                    0.5 * (ST_X1 - ST_X0), 0.5 * (ST_Y1 - ST_Y0), 6.0)
    d = np.maximum(core + 0 * Z, (HP - 2.0) - Z)
    return smax(d, Z - ST_TOP, 1.5)


# ----------------------------------------------------------- the dragon

ZB = HP  # height of the plinth top

BODY_CTRL = [
    (-110.5, -14.0, ZB + 1.6),   # tail tip, lying along the left of the pillar
    (-111.5, -27.0, ZB + 2.3),
    (-109.0, -39.0, ZB + 3.3),
    (-98.0, -38.0, ZB + 4.6),    # under the mask
    (-80.0, -37.5, ZB + 5.6),
    (-69.0, -37.0, ZB + 7.5),
    (-63.5, -28.0, ZB + 13.0),   # up the right side, facing the speaker
    (-63.5, -14.0, ZB + 22.0),
    (-65.0, -1.0, ZB + 31.0),
    (-72.0, 3.5, ZB + 38.5),     # across the back
    (-86.0, 4.0, ZB + 48.0),
    (-100.0, 3.5, ZB + 57.5),
    (-107.5, -3.0, ZB + 66.0),   # the left side
    (-108.5, -13.0, ZB + 76.0),
    (-107.5, -17.0, ZB + 86.0),
    (-104.5, -15.0, ST_TOP + 0.5),  # over the top edge
    (-98.0, -9.5, ST_TOP + 3.9),
    (-92.0, -6.0, ST_TOP + 4.0),
    (-87.0, -9.5, ST_TOP + 4.8),  # neck into the head
]

HEAD_O = np.array([-86.0, -19.0, ST_TOP])
HEAD_YAW = np.radians(8.0)   # turned slightly toward the speaker


def body_radius(t):
    return np.interp(t, [0, 0.06, 0.25, 0.75, 0.93, 1.0],
                     [1.0, 2.5, 5.4, 6.3, 5.4, 5.0])


def body_rest(sw):
    """Where the body crawls along the plinth it lies on it, slightly sunk."""
    w = smoothstep(0.34, 0.26, sw.t)
    target = ZB + 0.85 * sw.r
    sw.C[:, 2] = (1 - w) * sw.C[:, 2] + w * target


def body_up(C, T):
    """The dragon's back faces up and away from the pillar."""
    cx = np.clip(C[:, 0], ST_X0 + 6, ST_X1 - 6)
    cy = np.clip(C[:, 1], ST_Y0 + 6, ST_Y1 - 6)
    out = np.stack([C[:, 0] - cx, C[:, 1] - cy, np.zeros(len(C))], 1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
    return out * 0.35 + np.array([0, 0, 1.0])


def head_local(X, Y, Z):
    c, s = np.cos(HEAD_YAW), np.sin(HEAD_YAW)
    fwd = np.array([s, -c, 0.0])
    lft = np.array([c, s, 0.0])
    dx, dy = X - HEAD_O[0], Y - HEAD_O[1]
    f = dx * fwd[0] + dy * fwd[1]
    l = dx * lft[0] + dy * lft[1]
    u = Z - HEAD_O[2]
    return f, l, u


def head_to_world(p):
    c, s = np.cos(HEAD_YAW), np.sin(HEAD_YAW)
    fwd = np.array([s, -c, 0.0])
    lft = np.array([c, s, 0.0])
    p = np.asarray(p, float)
    return HEAD_O + p[..., 0:1] * fwd + p[..., 1:2] * lft + p[..., 2:3] * [0, 0, 1]


def dragon_head(X, Y, Z):
    f, l, u = head_local(X, Y, Z)
    al = np.abs(l)
    E = lambda c, r: sd_ellipsoid(f, al, u, c, r)
    d = E((-2.0, 0.0, 6.4), (8.5, 7.2, 6.2))                      # skull
    d = smin(d, E((8.0, 0.0, 6.0), (10.5, 5.0, 3.9)), 2.5)        # snout
    d = smin(d, E((16.5, 0.0, 7.0), (3.4, 5.6, 3.3)), 1.5)        # nose pad
    d = smin(d, E((7.5, 0.0, 2.2), (10.5, 4.6, 2.9)), 1.5)        # lower jaw
    d = smin(d, E((2.8, 4.3, 9.6), (4.6, 2.6, 2.3)), 1.2)         # brow ridges
    d = smin(d, E((-3.0, 6.2, 4.6), (4.2, 2.4, 3.6)), 1.2)        # cheek frills
    d = smin(d, E((12.0, 3.6, 8.8), (5.0, 1.6, 1.2)), 1.0)        # snout ridges
    # V-shaped mouth line along the sides and the front: a 45-degree groove
    # so it prints without bridging
    w = 1.7 * smoothstep(1.0, 4.0, f)
    d = d + np.maximum(0.0, w - np.abs(u - 4.3))
    # eyes in their sockets
    d = np.minimum(d, sd_ellipsoid(f, al, u, (4.6, 5.0, 8.0), (2.1, 1.9, 1.9)))
    # nostrils
    d = np.maximum(d, -sd_ellipsoid(f, al, u, (19.4, 2.4, 8.4), (1.4, 1.0, 1.1)))
    # fangs hanging into the mouth line
    for fc, lc in ((15.0, 3.2), (10.5, 3.9)):
        q = np.hypot(f - fc, al - lc)
        rr = 0.85 * np.clip((u - 2.6) / 2.6, 0, 1)
        d = np.minimum(d, np.maximum(q - rr, np.maximum(u - 5.4, 2.6 - u)))
    # head sits on the pillar top, nothing below it
    d = np.maximum(d, -u - 1.0)
    return d


def head_sweeps():
    """Horns, mane tufts and whiskers as small sweeps in head coordinates."""
    out = []
    taper = lambda r0, r1: (lambda t: r0 + (r1 - r0) * t ** 0.9)
    for side in (1, -1):
        horn = [(-2.5, 3.4 * side, 10.4), (-5.8, 4.6 * side, 14.5),
                (-9.2, 5.6 * side, 19.0), (-11.6, 5.9 * side, 23.0)]
        out.append(("horn", horn, taper(2.1, 0.4)))
        for k, (dl, df) in enumerate(((7.2, -6.0), (5.0, -9.0), (2.2, -11.0))):
            tuft = [(df + 2, dl * side * 0.9, 2.6), (df - 3, (dl + 1.6) * side, 1.9),
                    (df - 6, (dl + 2.2) * side, 1.6), (df - 8.5, (dl + 0.8) * side, 1.3)]
            out.append(("mane", tuft, taper(2.4 - 0.2 * k, 0.5)))
        whisk = [(18.0, 4.8 * side, 6.0), (16.5, 7.0 * side, 2.4), (13.0, 10.5 * side, 0.9),
                 (6.0, 14.5 * side, 0.8), (-2.0, 15.5 * side, 0.8), (-8.0, 13.8 * side, 0.8)]
        out.append(("whisker", whisk, taper(0.95, 0.6)))
    return out


def tail_flames():
    """Flame tongues off the tail tip, lying flat on the plinth."""
    base = np.array(BODY_CTRL[0])
    out = []
    for dx, dy, bend in ((-3.5, 9.0, -2.0), (0.0, 11.0, 1.5), (3.6, 8.0, 3.5)):
        pts = [base + [0, 2, 0], base + [dx * 0.5, 5.0, 0],
               base + [dx + bend * 0.5, dy * 0.75, 0], base + [dx + bend, dy, 0]]
        out.append(pts)
    return out


PEARL = np.array([-55.5, -44.0, ZB + 2.6])
PEARL_R = 4.6

# legs lie on the plinth: (hip, knee/elbow, wrist) and where the toes end
LEGS = [
    # front leg, reaching for the pearl
    dict(path=[(-73.0, -38.0), (-66.5, -45.5), (-61.5, -47.5)], r=(2.8, 1.7),
         toes=[PEARL[:2] + 4.9 * np.array([np.cos(a), np.sin(a)])
               for a in np.radians([195, 228, 262])]),
    # hind leg by the tail
    dict(path=[(-107.5, -37.0), (-114.0, -41.5), (-116.0, -46.5)], r=(2.4, 1.5),
         toes=[(-118.5, -48.5), (-116.6, -50.8), (-113.6, -51.3)]),
]


def lie_on(z0):
    def post(sw):
        sw.C[:, 2] = z0 + 0.8 * sw.r
    return post


def fang_ctrl(side):
    """Long fangs from the corners of the grin, ending on the chin.

    The tip ends on the mask, never below it: a downward point that ends in
    the air would be the first thing printed on that layer, with nothing
    under it.
    """
    Zs = np.array([-10.5, -16.5, -23.0, -29.0, -33.0])
    Xs = np.array([12.6, 13.4, 13.3, 12.4, 11.2]) * side
    x, z = MASK_X + Xs, MASK_Z + Zs
    H = np.maximum(mask_height(x, z), 9.5)
    return np.stack([x, ST_Y0 - H, z], 1)


def fang_radius(t):
    return 3.1 * (1 - t) ** 0.85 + 0.35


def horn_ctrl(side):
    """Oni horns: out of the temples, outward, then curling up."""
    x0 = MASK_X + 17.0 * side
    p0 = np.array([x0, ST_Y0 - 4.0, MASK_Z + 31.0])
    steps = [(0, 0, 0), (5.5, -2.0, 11.0), (11.0, -2.5, 23.0), (13.5, -1.5, 35.0),
             (13.0, 0.0, 44.0)]
    return [p0 + np.array([dx * side, dy, dz]) for dx, dy, dz in steps]


# --------------------------------------------------------------- assembly

def self_support(F, h, k_bed, iso=0.0):
    """Adds 45-degree wedges under every overhang steeper than 45 degrees.

    Walks the layers from the top down: a layer must contain the layer
    above it shrunk by one voxel (one voxel sideways per voxel of height
    = 45 degrees). Whatever is missing is filled in.
    """
    solid = F < iso
    need = np.zeros(solid.shape[:2], bool)
    added = np.zeros_like(solid)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(solid.shape[2] - 1, k_bed - 1, -1):
        layer = solid[:, :, k]
        need = ndimage.binary_erosion(need, st, border_value=0)
        miss = need & ~layer
        added[:, :, k] = miss
        need = layer | miss
    F = F.copy()
    F[added] = np.minimum(F[added], -0.35 * h)
    return F, int(added.sum())


def overhang_report(F, h, k_bed, iso=0.0, min_vox=4):
    """Lists what would still print in mid-air: voxels with nothing within
    one voxel sideways in the layer below them (i.e. beyond 45 degrees)."""
    solid = F < iso
    st = np.ones((3, 3), bool)
    bad = np.zeros_like(solid)
    for k in range(k_bed + 1, solid.shape[2]):
        bad[:, :, k] = solid[:, :, k] & ~ndimage.binary_dilation(solid[:, :, k - 1], st)
    lab, n = ndimage.label(bad, np.ones((3, 3, 3), bool))
    out = []
    if n:
        sizes = ndimage.sum(bad, lab, range(1, n + 1))
        cms = ndimage.center_of_mass(bad, lab, range(1, n + 1))
        for sz, cm in sorted(zip(sizes, cms), key=lambda t: -t[0]):
            if sz >= min_vox:
                out.append((int(sz), cm))
    return int(bad.sum()), out


def build(res):
    t0 = time.time()
    lo = (WING_X0 - 4, FRONT_Y - 4, -2.0)
    hi = (58.0, BACK_Y + 4, 146.0)
    fld = Field(lo, hi, res)
    print(f"grid {tuple(fld.n)} = {np.prod(fld.n) / 1e6:.1f} M voxels")

    # plinth
    fld.union((WING_X0 - 2, FRONT_Y - 2, -1), (56, BACK_Y + 2, HP + 1), plinth)

    # pillar with the mask
    fld.union((ST_X0 - 4, ST_Y0 - 4, HP - 3), (ST_X1 + 4, ST_Y1 + 4, ST_TOP + 2),
              pillar, k=2.0)
    fld.union((MASK_X - 29, ST_Y0 - 18, MASK_Z - 40), (MASK_X + 29, ST_Y0 + 7, MASK_Z + 40),
              oni_mask, k=1.0)
    print(f"  plinth + mask  {time.time() - t0:5.1f}s")

    # oni horns with collars
    for side in (1, -1):
        hc = horn_ctrl(side)
        sw = Sweep(hc, lambda t: 6.0 * (1 - t) ** 0.85 + 0.45)
        lo_, hi_ = sw.bbox(2)
        fld.union(lo_, hi_, tube_fn(sw), k=1.2)
        # collar ring round the base of the horn
        n = sw.T[int(0.11 * len(sw.C))]
        c = sw.C[int(0.11 * len(sw.C))]
        rr = sw.r[int(0.11 * len(sw.C))] + 0.3

        def collar(X, Y, Z, c=c, n=n, rr=rr):
            dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
            ax = dx * n[0] + dy * n[1] + dz * n[2]
            rad = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - ax * ax, 0))
            return np.hypot(rad - rr, ax) - 1.3

        fld.union(c - 9, c + 9, collar, k=0.4)
    for side in (1, -1):
        sw = Sweep(fang_ctrl(side), fang_radius)
        sw.C[:, 1] += 0.3 * sw.r              # sunk into the face by a third
        sw.tree = cKDTree(sw.C)
        lo_, hi_ = sw.bbox(2)
        fld.union(lo_, hi_, tube_fn(sw), k=0.8)
    print(f"  horns, fangs   {time.time() - t0:5.1f}s")

    # dragon body with scales and a dorsal fin
    body = Sweep(BODY_CTRL, body_radius, up=body_up, post=body_rest)
    fh = 3.8 * np.interp(body.t, [0, .08, .2, .85, .95, 1], [0, 0, 1, 1, .7, .6])
    fh = fh * (0.45 + 0.55 * sawtooth(body.s, 4.2))
    lo_, hi_ = body.bbox(5)
    fld.union(lo_, hi_, tube_fn(body, scales=0.45, fin={"h": fh, "th": 1.5}), k=1.6)
    print(f"  dragon body    {time.time() - t0:5.1f}s  ({body.length:.0f} mm)")

    for pts in tail_flames():
        sw = Sweep(pts, lambda t: 1.9 * (1 - t) + 0.4)
        sw.C[:, 2] = ZB + sw.r * 0.75
        sw.tree = cKDTree(sw.C)
        lo_, hi_ = sw.bbox(2)
        fld.union(lo_, hi_, tube_fn(sw), k=0.8)

    # legs, claws, the pearl
    for leg in LEGS:
        r0, r1 = leg["r"]
        pts = [(x, y, ZB) for x, y in leg["path"]]
        sw = Sweep(pts, lambda t, r0=r0, r1=r1: r0 + (r1 - r0) * t, post=lie_on(ZB))
        lo_, hi_ = sw.bbox(2)
        fld.union(lo_, hi_, tube_fn(sw, scales=0.3), k=1.0)
        wx, wy = leg["path"][-1]
        for tx, ty in leg["toes"]:
            mid = (0.5 * (wx + tx) + 0.3 * (ty - wy), 0.5 * (wy + ty) - 0.3 * (tx - wx))
            sw = Sweep([(wx, wy, ZB), (*mid, ZB), (tx, ty, ZB)],
                       lambda t: 1.25 - 0.95 * t ** 1.3, post=lie_on(ZB))
            lo_, hi_ = sw.bbox(2)
            fld.union(lo_, hi_, tube_fn(sw), k=0.5)
    fld.union(PEARL - PEARL_R - 1, PEARL + PEARL_R + 1,
              lambda X, Y, Z: np.sqrt((X - PEARL[0]) ** 2 + (Y - PEARL[1]) ** 2 +
                                      (Z - PEARL[2]) ** 2) - PEARL_R, k=0.6)

    # head
    lo_ = HEAD_O + [-26, -26, -2]
    hi_ = HEAD_O + [26, 26, 30]
    fld.union(lo_, hi_, dragon_head, k=1.5)
    for kind, pts, rf in head_sweeps():
        wp = head_to_world(np.array(pts))
        sw = Sweep(wp, rf)
        if kind in ("whisker", "mane"):
            # anything lying on the pillar top rests on it, never floats
            floor = ST_TOP + sw.r * 0.55
            if kind == "mane":
                sw.C[:, 2] = floor
            else:
                sw.C[:, 2] = np.maximum(sw.C[:, 2], floor)
            sw.tree = cKDTree(sw.C)
        lo_, hi_ = sw.bbox(2)
        fld.union(lo_, hi_, tube_fn(sw), k=0.7)
    print(f"  head           {time.time() - t0:5.1f}s")

    # cuts: speaker pocket, cable slot, waves on the front
    pocket_half = SPK / 2 + SPK_CLEAR
    fld.cut((-pocket_half - 1, -pocket_half - 1, HP - POCKET_DEPTH - 1),
            (pocket_half + 1, pocket_half + 1, 146),
            lambda X, Y, Z: np.maximum(sd_rrect(X, Y, 0, 0, pocket_half, pocket_half,
                                                POCKET_R) + 0 * Z,
                                       (HP - POCKET_DEPTH) - Z))
    fld.cut((CABLE_X - CABLE_W, 30, HP - POCKET_DEPTH - 1), (CABLE_X + CABLE_W, BACK_Y + 4, 146),
            lambda X, Y, Z: vmax([
                sd_rrect(X, Y, CABLE_X, BACK_Y, CABLE_W / 2, 12.0, 2.0) + 0 * Z,
                (HP - POCKET_DEPTH) - Z]), k=0.6)
    fld.cut((-112, FRONT_Y - 3, 0), (44, FRONT_Y + 2, HP), front_waves)
    print(f"  cuts           {time.time() - t0:5.1f}s")

    # the speaker's space must stay empty whatever is near it
    spk_keep = lambda X, Y, Z: np.maximum(
        sd_rrect(X, Y, 0, 0, SPK / 2 + 0.5, SPK / 2 + 0.5, POCKET_R) + 0 * Z, (HP - POCKET_DEPTH) - Z)
    fld.cut((-55, -55, 0), (55, 55, 146), spk_keep)

    k_bed = int(np.ceil(-fld.lo[2] / res))       # first layer at z >= 0
    F, n_add = self_support(fld.F, res, k_bed)
    print(f"  self-support   {time.time() - t0:5.1f}s  (+{n_add * res ** 3 / 1000:.1f} cm3)")
    total, spots = overhang_report(F, res, k_bed)
    print(f"  still in the air: {total} voxels, {len(spots)} spots")
    for sz, cm in spots[:12]:
        p = fld.lo + np.array(cm) * res
        print(f"    {sz:5d} vox at x={p[0]:7.1f} y={p[1]:6.1f} z={p[2]:6.1f}")
    return fld, F


def to_mesh(fld, F):
    from skimage.measure import marching_cubes
    import trimesh

    pad = np.pad(F, 1, constant_values=50.0)
    # a level a hair off zero: flat faces that fall exactly on grid nodes
    # otherwise give zero-area triangles and non-manifold edges
    v, f, _, _ = marching_cubes(pad, 1.3e-3, spacing=(fld.h,) * 3)
    v += fld.lo - fld.h
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=float, default=0.3, help="voxel size, mm")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parents[1] /
                                               "stl" / "alisa_midi_oni_dragon_stand.stl"))
    ap.add_argument("--faces", type=int, default=600_000,
                    help="decimate to about this many triangles (0 = keep all)")
    args = ap.parse_args()

    fld, F = build(args.res)
    m = to_mesh(fld, F)
    print(f"mesh: {len(m.faces):,} triangles, watertight={m.is_watertight}")
    if args.faces and len(m.faces) > args.faces:
        import fast_simplification
        import trimesh
        v, f = fast_simplification.simplify(m.vertices, m.faces,
                                            target_reduction=1 - args.faces / len(m.faces))
        m = trimesh.Trimesh(v, f, process=True)
        # collapsing edges leaves a few slivers; drop them and close the gaps
        m.update_faces(m.nondegenerate_faces(height=1e-6))
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        print(f"decimated: {len(m.faces):,} triangles, watertight={m.is_watertight}")
    ext = m.bounds
    print(f"size: {ext[1, 0] - ext[0, 0]:.1f} x {ext[1, 1] - ext[0, 1]:.1f} x "
          f"{ext[1, 2] - ext[0, 2]:.1f} mm, volume {m.volume / 1000:.1f} cm3")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.export(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
