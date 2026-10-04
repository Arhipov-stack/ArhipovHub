"""Horns (separate parts) and the dowel pins that hold them in the head."""
import numpy as np

import params as C
from head import horn_axis
from sdf import bezier, cut, cyl, smax, tube

HORN_R0 = 14.0     # radius at the root, sits inside the collar ring (R 15.5)


def horn_curve():
    """Centreline of the right horn in head coordinates."""
    o, u = horn_axis()
    b0 = o + u * 2.0                       # collar face
    pts = bezier([b0, b0 + u * 30, b0 + np.array([62.0, -4, 48]), b0 + np.array([40.0, -12, 102])], 64)
    return pts, b0, u


def horn_sdf(P):
    pts, b0, u = horn_curve()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    t = np.linspace(0, 1, len(pts))
    r = HORN_R0 * (1 - t) ** 0.85 + 0.9

    def detail(s, th):
        stri = 0.32 * np.sin(17 * th) * np.clip(1.15 - s * 1.3, 0, 1)
        rings = 0.55 * np.cos(2 * np.pi * s * L / 3.4) * np.clip((0.30 - s) / 0.08, 0, 1)
        return stri + rings

    d = tube(P, pts, r, pad=1.0, detail=detail)
    d = np.maximum(d, -((P - b0) @ u))            # flat root face
    # dowel hole in the root
    hole = cyl(P, b0 - u * 1, b0 + u * (C.HORN_PEG_LEN + 2.5), C.HORN_PEG_D / 2 + C.HORN_HOLE_GAP)
    return cut(d, hole, 0.0)


def horn_bounds():
    pts, b0, u = horn_curve()
    return pts.min(0) - 16, pts.max(0) + 16


def pin_sdf(P):
    """Dowel pin standing on z = 0: 10 mm, length = both holes minus 1 mm play."""
    L = C.HORN_PEG_LEN + 1.5 + C.HORN_PEG_LEN + 1.5
    r = C.HORN_PEG_D / 2 - 0.05
    d = cyl(P, (0, 0, 0), (0, 0, L), r)
    ch = 0.8
    rad = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    d = np.maximum(d, rad - (r - ch) - P[:, 2])               # bottom chamfer
    d = np.maximum(d, rad - (r - ch) - (L - P[:, 2]))         # top chamfer
    return d


PIN_LO = (-7, -7, -1)
PIN_HI = (7, 7, 2 * C.HORN_PEG_LEN + 5)
