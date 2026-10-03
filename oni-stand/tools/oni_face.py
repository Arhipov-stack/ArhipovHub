"""The oni face, sculpted from signed distance fields.

Frame: x to the right, y towards the back, z up; the face looks towards -y.
The face is a wide oni-gawara style mug: angry V brows, slanted almond eyes,
a bulbous nose, a grimace of clenched teeth with two tusks jutting forward,
and curls of hair on the temples. The eyes and the teeth are separate
inserts (they print in other colours), so the head has sockets for them.

Everything here is in the scale of the head stand; the shrine mask reuses it
scaled down.
"""
import numpy as np

from common import (length, sd_box, sd_cone, sd_ellipsoid, sd_poly2d,
                    sd_polyline, smax, smin)

TOP = 52.0                      # top of the head
FACE_C = np.array([0.0, -40.0, 24.0])
FACE_R = np.array([64.0, 30.0, 34.0])
SOCKET_BACK = -50.0             # back wall of the eye and mouth sockets


def ys(x, z):
    """y of the bare face surface (before features) at x, z."""
    q = 1 - (x / FACE_R[0]) ** 2 - ((z - FACE_C[2]) / FACE_R[2]) ** 2
    return FACE_C[1] - FACE_R[1] * np.sqrt(np.clip(q, 0, 1))


def on_face(x, z, out):
    return (x, float(ys(x, z)) - out, z)


# --- outlines in the x-z plane --------------------------------------------------

MOUTH_W = 35.0


def mouth_outline(n=90):
    x = np.linspace(-MOUTH_W, MOUTH_W, n)
    t = (x / MOUTH_W) ** 2
    s = np.clip(1 - t, 0, 1) ** 0.55
    upper = 9.0 + 8.0 * s - 3 * t
    lower = 9.0 - 5.0 * s - 3 * t
    return np.concatenate([np.stack([x, lower], 1), np.stack([x[::-1], upper[::-1]], 1)[1:-1]])


def bite_line(x):
    return 10.0 - 3 * (x / MOUTH_W) ** 2


def eye_outline(side, n=60):
    inner = np.array([side * 12.0, 27.0])
    outer = np.array([side * 35.0, 33.0])
    t = np.linspace(0, 1, n)
    axis = outer - inner
    L = np.linalg.norm(axis)
    a = axis / L
    perp = np.array([-a[1], a[0]]) * side       # points up for both eyes
    up = 6.2 * np.sin(np.pi * t ** 0.8) ** 0.8
    lo = 4.0 * np.sin(np.pi * t) ** 0.9
    base = inner + np.outer(t, axis)
    top = base + np.outer(up, perp)
    bot = base - np.outer(lo, perp)
    pts = np.concatenate([bot, top[::-1][1:-1]])
    if side < 0:
        pts = pts[::-1]
    return pts


# flame tongues on the brows: root x, z, tip x, z, root radius
BROW_FLAMES = [(13, 40.5, 19, 48.0, 2.8), (21, 43.0, 29, 50.5, 2.8), (30, 45.0, 40, 50.5, 2.6),
               (38, 46.0, 49, 46.5, 2.3)]


def eye_centre(side):
    return np.array([side * 22.5, 29.5])


TUSK_BASE_Z, TUSK_MID_Z, TUSK_TIP_Z = 5.5, 10.5, 19.5


def tusk_points(side):
    base = np.array(on_face(side * 20.0, TUSK_BASE_Z, -1.5))
    mid = np.array([side * 21.0, base[1] - 8.5, TUSK_MID_Z])
    tip = np.array([side * 22.0, mid[1] - 9.5, TUSK_TIP_Z])
    return [base, mid, tip], [3.4, 2.6, 1.0]


# --- 2-D fields, computed once per x-z grid --------------------------------------

class Planes:
    """Fields that depend only on x and z, broadcast along y."""

    def __init__(self, xs, zs):
        U, W = np.meshgrid(xs, zs, indexing="ij")
        self.YS = ys(U, W)
        self.mouth = sd_poly2d(U, W, mouth_outline())
        self.eyes = np.minimum(sd_poly2d(U, W, eye_outline(-1)),
                               sd_poly2d(U, W, eye_outline(1)))
        self.U, self.W = U, W

    def take(self, zsel):
        return {k: v[:, None, zsel] for k, v in (("YS", self.YS), ("mouth", self.mouth),
                                                 ("eyes", self.eyes), ("U", self.U),
                                                 ("W", self.W))}


def curls(Y, Z, centres):
    """Relief of spiral curls on a side wall, as a height in mm."""
    h = np.zeros_like(Y)
    for cy, cz, r0, turn in centres:
        dy, dz = Y - cy, Z - cz
        r = length(dy, dz)
        phi = np.arctan2(dz, dy) * turn
        phase = r / 2.6 - phi / (2 * np.pi)
        ridge = 0.5 + 0.5 * np.cos(2 * np.pi * phase)
        fade = np.clip((r0 - r) / 2.0, 0, 1)
        h = np.maximum(h, 1.1 * ridge * fade)
    return h


def head_field(X, Y, Z, P, block=((-64, -54, -4), (64, 64, 58))):
    """The head without its flat cuts (those are done with exact booleans)."""
    YS, mouth, eyes = P["YS"], P["mouth"], P["eyes"]

    d = sd_box(X, Y, Z, block[0], block[1], 10.0)
    d = smin(d, sd_ellipsoid(X, Y, Z, FACE_C, FACE_R), 10.0)

    # curls of hair on both temples
    side_x = block[1][0]
    cl = [(-30, 34, 10.5, 1), (-8, 22, 9.5, -1), (-36, 12, 8.5, -1), (14, 36, 9.0, 1),
          (34, 18, 9.5, 1)]
    relief = curls(Y, Z, cl) * np.clip(1 - (side_x - np.abs(X)) / 1.5, 0, 1)
    d = d - relief

    # cheeks, chin, bags under the eyes, smile folds
    for s in (-1, 1):
        d = smin(d, sd_ellipsoid(X, Y, Z, on_face(s * 42, 20, 0.5), (11, 3, 8)), 6)
        d = smin(d, sd_ellipsoid(X, Y, Z, on_face(s * 24, 22.5, 0.6), (10, 3.5, 2.6)), 3)
        d = smin(d, sd_cone(X, Y, Z, on_face(s * 16, 20.5, 1.5), on_face(s * 39, 5.5, 1.0),
                            2.8, 1.6), 2.5)

    # brows: a heavy angry V whose upper edge breaks into flame tongues
    for s in (-1, 1):
        pts = [on_face(s * 5.5, 35.0, 2.5), on_face(s * 28, 43.5, 4.5), on_face(s * 42, 46.0, 3.0)]
        brow = sd_polyline(X, Y, Z, pts, [5.2, 4.6, 3.0])
        for (x0, z0, x1, z1, r0) in BROW_FLAMES:
            tongue = sd_cone(X, Y, Z, on_face(s * x0, z0, 4.0), on_face(s * x1, z1, 2.2), r0, 0.7)
            brow = smin(brow, tongue, 1.6)
        axis_z = 35.0 + (np.abs(X) - 5.5) * 0.30
        strokes = np.sin(2 * np.pi * (Z - axis_z) / 2.6)
        d = smin(d, brow + 0.4 * strokes, 1.5)
        # frown creases between the brows
        crease = sd_cone(X, Y, Z, on_face(s * 3.2, 35.5, 0.5), on_face(s * 2.4, 47, 0.0), 1.0, 0.8)
        d = smax(d, -crease, 0.8)
    d = smin(d, sd_ellipsoid(X, Y, Z, on_face(0, 41, 1.5), (3.5, 3, 5)), 2)

    # nose
    nose = sd_cone(X, Y, Z, on_face(0, 36, 1.0), on_face(0, 25, 7.0), 4.0, 6.0)
    nose = smin(nose, sd_ellipsoid(X, Y, Z, on_face(0, 21, 7.5), (10, 7, 7)), 2.5)
    for s in (-1, 1):
        nose = smin(nose, sd_ellipsoid(X, Y, Z, on_face(s * 11, 18.5, 3.5), (6, 5, 5)), 2)
    d = smin(d, nose, 2.5)
    for s in (-1, 1):
        d = smax(d, -sd_ellipsoid(X, Y, Z, on_face(s * 6.5, 16.5, 7.0), (3.2, 3.5, 2.4)), 0.8)

    # lips around the mouth, eyelids around the eyes
    lips = np.maximum(np.abs(mouth - 1.0) - 1.4, (YS - 2.6) - Y)
    d = smin(d, lips, 1.5)
    lids = np.maximum(np.abs(eyes - 1.0) - 1.3, (YS - 2.2) - Y)
    d = smin(d, lids, 1.2)

    # sockets for the inserts
    d = smax(d, -np.maximum(mouth, Y - SOCKET_BACK), 0.4)
    d = smax(d, -np.maximum(eyes, Y - SOCKET_BACK), 0.4)

    # room in the upper lip for the tusks, open to the front
    for s in (-1, 1):
        pts, radii = tusk_points(s)
        m = ((np.abs(X - s * 21.0) < 6.5) & (Z > 0) & (Z < 26) & (Y < SOCKET_BACK)
             & (Y > -95))
        if not m.any():
            continue
        x, y, z = X[m], Y[m], Z[m]
        sweep = np.full(x.shape, np.inf)
        for shift in np.arange(0, 22, 0.8):
            sweep = np.minimum(sweep, sd_polyline(x, y + shift, z, pts, radii))
        d[m] = np.maximum(d[m], -(sweep - 0.3))
    return d


def teeth_field(X, Y, Z, P):
    """Clenched teeth with two tusks; slides into the mouth from the front."""
    YS, mouth, U = P["YS"], P["mouth"], P["U"]
    front = YS + 1.2
    xs_up = np.array([0, 7, 13, 18.5, 23, 27, 31])
    xs_lo = np.array([3.5, 10, 15.5, 21.5, 26, 30])
    zb = bite_line(U)
    gu = np.max([np.exp(-((np.abs(U) - x) / 0.45) ** 2) for x in xs_up], axis=0)
    gl = np.max([np.exp(-((np.abs(U) - x) / 0.45) ** 2) for x in xs_lo], axis=0)
    upper = Z > zb
    groove = 1.3 * np.where(upper, gu, gl) + 1.4 * np.exp(-((Z - zb) / 0.6) ** 2)
    d = np.maximum(np.maximum(mouth + 0.25, Y - SOCKET_BACK), (front + groove) - Y)
    for s in (-1, 1):
        pts, radii = tusk_points(s)
        d = smin(d, sd_polyline(X, Y, Z, pts, radii), 1.0)
    return d


def eye_field(X, Y, Z, P):
    """Domed eyeballs with an iris ring and a pupil, sliding into the sockets."""
    YS, eyes, U, W = P["YS"], P["eyes"], P["U"], P["W"]
    dome = 3.0 * np.sqrt(np.clip(-eyes / 3.5, 0, 1))
    front = YS - 0.5 - dome
    d = np.maximum(np.maximum(eyes + 0.25, Y - SOCKET_BACK), front - Y)
    for s in (-1, 1):
        ex, ez = eye_centre(s)
        r = length(U - ex, W - ez)
        pupil = np.maximum(r - 1.9, Y - (front + 1.4))
        ring = np.maximum(np.abs(r - 4.3) - 0.35, Y - (front + 0.5))
        d = np.maximum(d, -np.minimum(pupil, ring))
    return d
