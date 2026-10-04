"""Small numpy SDF kit: primitives, smooth booleans, tubes along curves, meshing.

Every primitive takes P with shape (N, 3) in millimetres and returns signed
distance (negative inside). Heavy primitives accept a bounding box and only
evaluate points inside it; points outside get FAR, which is harmless for
unions (nothing added) and for cuts (nothing removed).
"""
import numpy as np

FAR = np.float32(1e3)


def _len(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def cut(a, b, k=0.0):
    """a minus b."""
    return smax(a, -b, k) if k > 0 else np.maximum(a, -b)


def local(P, lo, hi, fn):
    """Evaluate fn only for points inside the box [lo, hi]."""
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    m = np.all((P >= lo) & (P <= hi), axis=1)
    out = np.full(len(P), FAR, np.float32)
    if m.any():
        out[m] = fn(P[m])
    return out


def round_box(P, c, half, r):
    q = np.abs(P - np.asarray(c, np.float32)) - (np.asarray(half, np.float32) - r)
    outside = _len(np.maximum(q, 0))
    inside = np.minimum(q.max(axis=1), 0)
    return outside + inside - r


def ellipsoid(P, c, r):
    p = (P - np.asarray(c, np.float32)) / np.asarray(r, np.float32)
    k0 = _len(p)
    k1 = _len(p / np.asarray(r, np.float32))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def sphere(P, c, r):
    return _len(P - np.asarray(c, np.float32)) - r


def cyl(P, a, b, r):
    """Capped cylinder between points a and b."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    ba = b - a
    L = np.linalg.norm(ba)
    u = ba / L
    pa = P - a
    t = pa @ u
    radial = _len(pa - t[:, None] * u) - r
    axial = np.abs(t - L / 2) - L / 2
    return np.minimum(np.maximum(radial, axial), 0) + _len(
        np.stack([np.maximum(radial, 0), np.maximum(axial, 0)], 1))


def torus(P, c, axis, R, r):
    axis = np.asarray(axis, np.float32)
    axis = axis / np.linalg.norm(axis)
    p = P - np.asarray(c, np.float32)
    h = p @ axis
    rad = _len(p - h[:, None] * axis)
    return np.sqrt((rad - R) ** 2 + h ** 2) - r


def frames(pts):
    """Parallel-transport frames (tangent, normal, binormal) along a polyline."""
    pts = np.asarray(pts, np.float64)
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N = [np.cross(T[0], ref)]
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, len(T)):
        n = N[-1] - T[i] * (N[-1] @ T[i])
        N.append(n / np.linalg.norm(n))
    N = np.array(N)
    B = np.cross(T, N)
    return T, N, B


def tube(P, pts, radii, pad=1.0, detail=None):
    """Tube with varying radius along a polyline.

    detail(t, theta) -> radial offset in mm (positive bulges out). t is the
    arc-length fraction 0..1, theta the angle around the centreline.
    """
    pts = np.asarray(pts, np.float32)
    radii = np.broadcast_to(np.asarray(radii, np.float32), (len(pts),))
    rmax = float(radii.max()) + pad + 2.0
    lo = pts.min(0) - rmax
    hi = pts.max(0) + rmax

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-6)
    if detail is not None:
        _, Nf, Bf = frames(pts)
        Nf = Nf.astype(np.float32)
        Bf = Bf.astype(np.float32)

    def fn(Q):
        best = np.full(len(Q), np.inf, np.float32)
        bt = np.zeros(len(Q), np.float32)
        bi = np.zeros(len(Q), np.int32)
        bh = np.zeros(len(Q), np.float32)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ba = b - a
            pa = Q - a
            h = np.clip((pa @ ba) / max(float(ba @ ba), 1e-9), 0, 1)
            d = _len(pa - h[:, None] * ba) - (radii[i] + (radii[i + 1] - radii[i]) * h)
            m = d < best
            best[m] = d[m]
            bi[m] = i
            bh[m] = h[m]
        if detail is not None:
            t = s[bi] + (s[bi + 1] - s[bi]) * bh
            c = pts[bi] + (pts[bi + 1] - pts[bi]) * bh[:, None]
            v = Q - c
            n = Nf[bi] * (1 - bh[:, None]) + Nf[bi + 1] * bh[:, None]
            bb = Bf[bi] * (1 - bh[:, None]) + Bf[bi + 1] * bh[:, None]
            theta = np.arctan2(np.einsum("ij,ij->i", v, bb), np.einsum("ij,ij->i", v, n))
            best = best - detail(t.astype(np.float32), theta.astype(np.float32))
        return best

    return local(P, lo, hi, fn)


def bezier(ctrl, n=40):
    ctrl = np.asarray(ctrl, np.float64)
    t = np.linspace(0, 1, n)[:, None]
    k = len(ctrl) - 1
    from math import comb
    return sum(comb(k, i) * (1 - t) ** (k - i) * t ** i * ctrl[i] for i in range(k + 1))


def catmull(pts, n=12):
    """Smooth curve through points (Catmull-Rom), n samples per span."""
    p = np.asarray(pts, np.float64)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)


def grid_eval(fn, lo, hi, step, slab=24):
    """Sample fn on a regular grid, slab by slab along z. Returns (vol, origin)."""
    # shift the grid off round numbers so flat faces never land exactly on grid planes
    lo = np.asarray(lo, np.float32) + np.float32(0.3137 * step)
    hi = np.asarray(hi, np.float32)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs = lo[0] + np.arange(n[0], dtype=np.float32) * step
    ys = lo[1] + np.arange(n[1], dtype=np.float32) * step
    zs = lo[2] + np.arange(n[2], dtype=np.float32) * step
    vol = np.empty((n[0], n[1], n[2]), np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for k0 in range(0, n[2], slab):
        zz = zs[k0:k0 + slab]
        P = np.empty((n[0], n[1], len(zz), 3), np.float32)
        P[..., 0] = X[..., None]
        P[..., 1] = Y[..., None]
        P[..., 2] = zz[None, None, :]
        vol[:, :, k0:k0 + len(zz)] = fn(P.reshape(-1, 3)).reshape(n[0], n[1], len(zz))
    return vol, lo


def lighten(m, faces):
    """Decimate to a face budget and make sure the result is still a closed solid."""
    import trimesh
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    if not m.is_watertight:
        # decimation can leave a handful of non-manifold edges; MeshFix closes them
        import pymeshfix
        parts = []
        for p in m.split(only_watertight=False):
            mf = pymeshfix.MeshFix(p.vertices, p.faces)
            mf.repair(joincomp=False, remove_smallest_components=False)
            parts.append(trimesh.Trimesh(mf.points, mf.faces, process=True))
        m = trimesh.util.concatenate(parts)
    m.fix_normals()
    if m.volume < 0:
        m.invert()
    return m


def mesh(fn, lo, hi, step, faces=None):
    import trimesh
    from skimage.measure import marching_cubes
    vol, origin = grid_eval(fn, lo, hi, step)
    # close the volume so the surface is always watertight
    vol[0, :, :] = vol[-1, :, :] = vol[:, 0, :] = vol[:, -1, :] = vol[:, :, 0] = vol[:, :, -1] = 1.0
    vol[vol == 0] = 1e-6   # exact zeros on grid-aligned planes make non-manifold edges
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(step, step, step), allow_degenerate=False)
    m = trimesh.Trimesh(v + origin, f[:, ::-1], process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:  # drop specks left by the voxel grid, keep real parts
        m = trimesh.util.concatenate([p for p in parts if len(p.faces) > 200])
    return lighten(m, faces) if faces else m
