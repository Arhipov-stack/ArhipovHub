"""Oni (hannya) head that cradles the speaker.

Frame: z = 0 is the pocket floor (speaker bottom), the face looks towards +y,
the head is symmetric in x. The square socket for the neck is underneath.
"""
import numpy as np

import params as C
from sdf import (FAR, bezier, catmull, cut, cyl, ellipsoid, local, round_box,
                 smax, smin, sphere, torus, tube)

# ---------------------------------------------------------------- base mask
MASK_C = np.array([0.0, 30.0, 45.0])
MASK_R = np.array([72.0, 48.0, 85.0])
MASK_TOP = 104.0


def mask_volume(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # jaw narrows below the mouth
    rx = MASK_R[0] - 0.45 * np.clip(20.0 - z, 0, None)
    Q = np.stack([x * (MASK_R[0] / rx), y, z], 1)
    d = ellipsoid(Q, MASK_C, MASK_R)
    return smax(d, z - MASK_TOP, 4.0)


def surf_y(fn, x, z, y_hi=140.0, y_lo=-10.0):
    """Front-most y of surface fn at (x, z), by marching then bisection."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    z = np.atleast_1d(np.asarray(z, np.float32))
    ys = np.arange(y_hi, y_lo, -0.5, dtype=np.float32)
    P = np.stack(np.broadcast_arrays(x[:, None], ys[None, :], z[:, None]), -1).reshape(-1, 3)
    d = fn(P).reshape(len(x), len(ys))
    inside = d < 0
    k = np.argmax(inside, axis=1)
    hi = ys[np.maximum(k - 1, 0)]
    lo = ys[k]
    for _ in range(12):
        mid = (hi + lo) / 2
        dm = fn(np.stack([x, mid, z], 1))
        hi = np.where(dm >= 0, mid, hi)
        lo = np.where(dm < 0, mid, lo)
    return (hi + lo) / 2


def on_surf(fn, xz, lift):
    """Lift a 2D (x, z) curve onto surface fn, `lift` mm outwards (scalar or per point)."""
    xz = np.asarray(xz, np.float32)
    y = surf_y(fn, xz[:, 0], xz[:, 1]) + np.asarray(lift, np.float32)
    return np.stack([xz[:, 0], y, xz[:, 1]], 1)


def spiral_xz(c, R, turns, phase=0.0, sense=1, tail=None, n_per_turn=48):
    th = np.linspace(0.5 * np.pi, 2 * np.pi * turns, int(n_per_turn * turns))
    r = R * th / th[-1]
    pts = np.stack([c[0] + sense * r * np.cos(th + phase), c[1] + r * np.sin(th + phase)], 1)
    if tail is not None:
        pts = np.vstack([pts, catmull([pts[-1], *tail], 10)[1:]])
    return pts


# ---------------------------------------------------------------- features
def curve(fn, xz, lift):
    pts = catmull(xz, 8)
    if isinstance(lift, tuple):
        lift = np.linspace(lift[0], lift[1], len(pts))
    return on_surf(fn, pts, lift)


def build_features():
    """Pre-compute the curves that ride on the base mask surface."""
    S0 = mask_volume
    f = {}
    # forehead band that runs into the horn collars
    f["band"] = curve(S0, [(0, 95), (18, 96.5), (34, 95), (46, 91), (56, 86)], 1.5)
    # heavy angry brows, low at the nose, high at the temples
    f["brow"] = curve(S0, [(2, 70), (10, 74), (20, 79.5), (31, 84.5), (42, 88), (53, 88)], (3.5, 2.5))
    # eyelids
    f["lid_up"] = curve(S0, [(11, 65.5), (19, 70.5), (31, 73), (44, 70)], 1.0)
    f["lid_lo"] = curve(S0, [(12, 58.5), (22, 54.5), (34, 55), (44, 62)], 0.8)
    # nose bridge, growing towards the tip
    f["bridge"] = curve(S0, [(0, 73), (0, 62), (0, 50), (0, 42)], (2.0, 7.0))
    # cheek folds (three layers, like the carved original)
    f["fold1"] = curve(S0, [(18, 54), (30, 51), (42, 51), (55, 44)], (2.0, 1.0))
    f["fold2"] = curve(S0, [(22, 46), (34, 42), (45, 40), (57, 31)], (2.0, 1.0))
    f["fold3"] = curve(S0, [(25, 37), (35, 31), (43, 21), (48, 9)], (2.5, 1.2))
    # lips
    up = [(0, 25), (14, 24), (28, 20), (41, 12), (47, 6)]
    lo = [(0, -3), (14, -2), (28, 1.5), (41, 6.5), (47, 6)]
    f["lip_up"] = curve(S0, up, (4.0, 1.5))
    f["lip_lo"] = curve(S0, lo, (5.0, 1.5))
    f["lip_up_xz"] = np.array(up, np.float32)
    f["lip_lo_xz"] = np.array(lo, np.float32)
    # upper fangs, hanging in front of the lower lip down past the chin
    fang = bezier([(27, 0, 19), (31, 0, 4), (34, 0, -15), (32, 0, -38)], 30)
    fang[:, 1] = surf_y(S0, fang[:, 0], np.maximum(fang[:, 2], 2)) + np.linspace(1, 11, 30)
    f["fang"] = fang
    return f


F = None


def big_forms(P):
    """Mask + all large carved forms (before small decoration)."""
    d = mask_volume(P)

    d = smin(d, tube(P, F["band"], 4.5), 3.0)
    d = smin(d, tube(P, F["brow"], np.linspace(7.0, 8.5, len(F["brow"]))), 4.0)
    # round bosses at the brow ends that carry the big swirls
    yb = F["brow"][24, 1]
    d = smin(d, ellipsoid(P, (31, yb + 1, 84), (13, 7, 11.5)), 3.0)
    # glabella knot between the brows
    yg = F["brow"][0, 1]
    d = smin(d, ellipsoid(P, (0, yg - 1, 70), (10, 8, 9)), 3.0)

    # eye sockets: dig in, then lids on top
    ye = F["lid_up"][12, 1]
    d = cut(d, local(P, (6, 30, 46), (52, 130, 82),
                     lambda Q: ellipsoid(Q, (28, ye + 5, 63.5), (18, 13, 10))), 2.5)
    d = smin(d, tube(P, F["lid_up"], 3.4), 2.0)
    d = smin(d, tube(P, F["lid_lo"], 3.0), 2.0)

    # nose: bridge, bulb, flared wings, nostrils
    d = smin(d, tube(P, F["bridge"], np.linspace(6.0, 10.0, len(F["bridge"]))), 3.0)
    yn = F["bridge"][-1, 1]
    d = smin(d, sphere(P, (0, yn + 1, 36), 12.5), 3.0)
    d = smin(d, ellipsoid(P, (17, yn - 5, 32), (11, 9, 9.5)), 3.0)
    d = cut(d, ellipsoid(P, (9.5, yn + 1, 25.5), (5.5, 7, 3.5)), 1.0)

    # cheeks and folds
    yc = F["fold1"][20, 1]
    d = smin(d, ellipsoid(P, (42, yc - 5, 42), (16, 10, 15)), 5.0)
    for k in ("fold1", "fold2", "fold3"):
        d = smin(d, tube(P, F[k], np.linspace(4.2, 2.8, len(F[k]))), 3.0)

    # chin
    ych = F["lip_lo"][0, 1]
    d = smin(d, ellipsoid(P, (0, ych - 10, -20), (28, 14, 20)), 6.0)

    # lips
    d = smin(d, tube(P, F["lip_up"], np.linspace(6.0, 4.0, len(F["lip_up"]))), 2.0)
    d = smin(d, tube(P, F["lip_lo"], np.linspace(6.5, 4.0, len(F["lip_lo"]))), 2.0)
    return d


def mouth_hole(P):
    x, z = P[:, 0], P[:, 2]
    up = F["lip_up_xz"]
    lo = F["lip_lo_xz"]
    zu = np.interp(x, up[:, 0], up[:, 1])
    zl = np.interp(x, lo[:, 0], lo[:, 1])
    d = np.maximum.reduce([zl + 4.0 - z, z - (zu - 3.0), x - 40.0, 32.0 - P[:, 1], 1.0 - z])
    return d


def teeth(P):
    x, z = P[:, 0], P[:, 2]
    up = F["lip_up_xz"]
    lo = F["lip_lo_xz"]
    d = np.full(len(P), FAR, np.float32)
    for cx, w in ((3.9, 7.2), (11.8, 7.2), (19.6, 7.0), (26.8, 6.0), (32.6, 5.0)):
        zu = float(np.interp(cx, up[:, 0], up[:, 1]))
        zl = float(np.interp(cx, lo[:, 0], lo[:, 1]))
        yf = float(surf_y(mask_volume, cx, (zu + zl) / 2)[0]) - 2.0
        hu = 7.0 if cx < 25 else 5.0
        d = np.minimum(d, round_box(P, (cx, (yf + 46) / 2, zu - 1.5 - hu / 2), (w / 2, (yf - 46) / 2, hu / 2 + 1.5), 1.3))
        hl = 5.0 if cx < 25 else 4.0
        d = np.minimum(d, round_box(P, (cx, (yf + 46) / 2, zl + 2.0 + hl / 2), (w / 2, (yf - 46) / 2, hl / 2 + 1.5), 1.3))
    return d


def decorations(P, base):
    """Spirals, grooves and fangs that sit on the carved surface."""
    d = base

    # swirls on the brows, on the chin and on the cheeks
    sp = []
    sp.append((spiral_xz((30, 84.5), 11.0, 3.0, phase=0.3, sense=1,
                         tail=[(44, 90), (52, 89)]), 0.2, 1.5))
    sp.append((spiral_xz((12, -19), 7.0, 2.4, phase=2.6, sense=-1,
                         tail=[(4, -28), (-2, -30)]), 0.0, 1.1))
    for xz, lift, r in sp:
        pts = on_surf(SURF, xz, lift)
        d = smin(d, tube(P, pts, r), 0.6)

    # grain lines on the forehead band and above it
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    zone = (z > 87) & (z < 104) & (y > 20)
    g = 0.35 * np.sin(2 * np.pi * (z - 0.08 * x * x / 50.0) / 2.6)
    d = np.where(zone, d - g * np.clip((104 - z) / 3, 0, 1), d)

    # herringbone on the nose bridge
    zone = (np.abs(x) < 7) & (z > 44) & (z < 74) & (y > 50)
    g = 0.45 * np.sin(2 * np.pi * (z + 0.9 * x) / 3.2)
    d = np.where(zone, d - g * np.clip((7 - x) / 2, 0, 1), d)

    # fangs
    d = smin(d, tube(P, F["fang"], np.linspace(7.0, 1.3, len(F["fang"]))), 1.5)
    return d


SURF = None


def horn_axis():
    """Horn root: where the collar sits and which way the horn leaves the head."""
    o = np.array([C.PX + C.WALL + 9.0, 22.0, 86.0])
    u = np.array([1.0, 0.05, 0.42])
    return o, u / np.linalg.norm(u)


def cradle(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    hx, hy = C.PX + C.WALL, C.PY + C.WALL
    zb, zt = -C.FLOOR, C.RIM_Z
    d = round_box(P, (0, 0, (zb + zt) / 2), (hx, hy, (zt - zb) / 2), C.POCKET_R + C.WALL)
    # windows in the back and the rear of the sides: sound out, cable out, light weight
    for z0, z1 in ((3.0, 40.0), (52.0, 80.0)):
        d = cut(d, round_box(P, (0, -hy, (z0 + z1) / 2), (C.PX - 15, 12, (z1 - z0) / 2), 5.0))
        d = cut(d, round_box(P, (hx, -22.0, (z0 + z1) / 2), (12, 15.0, (z1 - z0) / 2), 5.0))
    # block under the floor that holds the neck socket
    d = smin(d, round_box(P, (0, -2, -18), (27, 30, 14), 5.0), 6.0)
    return d


def head_sdf(P):
    P = P.copy()
    P[:, 0] = np.abs(P[:, 0])
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    face = local(P, (-1, -20, -60), (90, 140, 112), lambda Q: decorations(Q, big_forms(Q)))

    # cut eyes and mouth through the mask (the speaker shows through, sound gets out)
    eye_c = np.array([28.0, 63.5])
    ang = np.deg2rad(12)
    ex = (x - eye_c[0]) * np.cos(ang) + (z - eye_c[1]) * np.sin(ang)
    ez = -(x - eye_c[0]) * np.sin(ang) + (z - eye_c[1]) * np.cos(ang)
    eye = (np.sqrt((ex / 13.5) ** 2 + (ez / 5.2) ** 2) - 1) * 5.2
    eye = np.maximum(eye, 30 - y)
    face = cut(face, eye, 0.8)
    face = cut(face, mouth_hole(P), 1.0)
    face = np.minimum(face, local(P, (-1, 40, -5), (40, 100, 30), teeth))

    d = smin(cradle(P), face, 5.0)

    # horn collar + boss
    o, u = horn_axis()
    d = smin(d, cyl(P, o - u * 12, o + u * 2, 16.5), 4.0)
    d = smin(d, torus(P, o + u * 0.5, u, 15.5, 3.8), 1.0)
    hole_r = C.HORN_PEG_D / 2 + C.HORN_HOLE_GAP
    d = cut(d, cyl(P, o - u * C.HORN_PEG_LEN, o + u * 10, hole_r))

    # speaker pocket, open to the top
    # only the vertical edges are rounded: a flat floor with a square edge, so the
    # speaker sits right down on it instead of riding up a fillet
    r = C.POCKET_R
    qx = np.abs(x) - (C.PX - r)
    qy = np.abs(y) - (C.PY - r)
    pocket = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    d = cut(d, np.maximum(pocket, -z))
    # neck socket with an entry chamfer
    s = C.PEG / 2 + C.SOCKET_GAP
    d = cut(d, round_box(P, (0, 0, -32 + C.SOCKET_DEPTH / 2), (s, s, C.SOCKET_DEPTH / 2), 0.5))
    ch = np.maximum.reduce([np.abs(x) - (s + 1.2) + (z + 32), np.abs(y) - (s + 1.2) + (z + 32), z + 32 - 1.2])
    d = cut(d, ch)
    d = np.maximum(d, -32 - z)   # flat bottom of the socket block
    return d


def setup():
    global F, SURF
    F = build_features()
    SURF = big_forms


HEAD_LO = (-95, -62, -48)
HEAD_HI = (95, 112, 112)
