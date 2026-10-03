"""Shared helpers: signed distance fields on numpy grids, meshing them, and
writing clean STL."""
import struct

import numpy as np
import shapely
from shapely.geometry import Polygon
from manifold3d import CrossSection, Manifold, Mesh, OpType
from skimage.measure import marching_cubes

# --- Station Midi -----------------------------------------------------------
STATION = 96.0        # footprint, mm (square)
STATION_H = 110.0
CLEARANCE = 1.25      # gap per side in the pocket
POCKET_R = 12.0       # pocket corner radius; must not exceed the station's
POCKET = STATION + 2 * CLEARANCE
NOTCH_W = 46.0        # cable gap at the back


# --- signed distance primitives (negative inside) ------------------------------

def length(*c):
    return np.sqrt(sum(x * x for x in c))


def sd_ellipsoid(X, Y, Z, c, r):
    px, py, pz = (X - c[0]) / r[0], (Y - c[1]) / r[1], (Z - c[2]) / r[2]
    k0 = length(px, py, pz)
    k1 = length(px / r[0], py / r[1], pz / r[2])
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_cone(X, Y, Z, a, b, ra, rb):
    """Capsule from a to b whose radius goes linearly from ra to rb."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / (ba @ ba), 0, 1)
    return length(px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h) - (ra + (rb - ra) * h)


def sd_polyline(X, Y, Z, pts, radii):
    d = None
    for i in range(len(pts) - 1):
        di = sd_cone(X, Y, Z, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = di if d is None else np.minimum(d, di)
    return d


def sd_box(X, Y, Z, lo, hi, r=0.0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2 - r
    qx, qy, qz = np.abs(X - c[0]) - h[0], np.abs(Y - c[1]) - h[1], np.abs(Z - c[2]) - h[2]
    out = length(np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0))
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_poly2d(U, V, pts):
    """Signed distance in a plane to a polygon outline, negative inside."""
    poly = Polygon(pts)
    u, v = U.ravel(), V.ravel()
    d = shapely.distance(shapely.points(u, v), poly.exterior)
    inside = shapely.contains_xy(poly, u, v)
    return np.where(inside, -d, d).reshape(U.shape)


# --- meshing ------------------------------------------------------------------

def mesh_field(field, lo, hi, step, chunk=24):
    """Evaluate field(X, Y, Z, (xs, zs, zslice)) on a grid in z-slabs and
    turn its zero level into a Manifold. The grid boundary is forced
    outside, so the surface always closes."""
    xs = np.arange(lo[0], hi[0] + step / 2, step)
    ys = np.arange(lo[1], hi[1] + step / 2, step)
    zs = np.arange(lo[2], hi[2] + step / 2, step)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    for k0 in range(0, len(zs), chunk):
        z = zs[k0:k0 + chunk]
        X, Y, Z = np.meshgrid(xs, ys, z, indexing="ij")
        vol[:, :, k0:k0 + chunk] = field(X, Y, Z, (xs, zs, slice(k0, k0 + chunk)))
    # a field exactly zero on a grid node makes marching cubes emit coincident
    # vertices from neighbouring cells (a pinch); nudge those nodes outside
    vol[np.abs(vol) < 1e-5] = 1e-5
    vol[0], vol[-1] = 1, 1
    vol[:, 0], vol[:, -1] = 1, 1
    vol[:, :, 0], vol[:, :, -1] = 1, 1
    V, F, _, _ = marching_cubes(vol, 0.0, spacing=(step, step, step),
                                allow_degenerate=False)
    V += np.array([xs[0], ys[0], zs[0]])
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    if m.is_empty() or m.volume() <= 0:
        m = Manifold(Mesh(vert_properties=V.astype(np.float32),
                          tri_verts=F[:, ::-1].astype(np.uint32)))
    if m.is_empty():
        raise RuntimeError(f"marching cubes gave a mesh manifold rejects: {m.status()}")
    return m


def rrect(w, h, r, cx=0.0, cy=0.0, seg=40):
    pts = []
    for x, y, a0 in ((cx + w / 2 - r, cy + h / 2 - r, 0),
                     (cx - w / 2 + r, cy + h / 2 - r, 90),
                     (cx - w / 2 + r, cy - h / 2 + r, 180),
                     (cx + w / 2 - r, cy - h / 2 + r, 270)):
        a = np.radians(a0 + np.linspace(0, 90, seg + 1))
        pts.append(np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1))
    return np.concatenate(pts)


def cs(pts):
    return CrossSection([np.asarray(pts, dtype=float)])


def prism(section, z0, z1):
    return section.extrude(z1 - z0).translate((0, 0, z0))


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def fit_test():
    """A thin frame with the exact pocket: drop the station in before
    committing to a long print."""
    inner = rrect(POCKET, POCKET, POCKET_R)
    outer = rrect(POCKET + 8, POCKET + 8, POCKET_R + 4)
    return prism(cs(outer), 0, 4) - prism(cs(inner), -1, 5)


# --- output -------------------------------------------------------------------

def float32_faces(man):
    """Triangles as the STL will store them. Rounding to float32 can fold a
    sliver into a pair of back-to-back faces or a collapsed one; both are
    dropped, and the result is checked to be closed."""
    mesh = man.to_mesh64()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    F = np.asarray(mesh.tri_verts)
    V, inv = np.unique(V, axis=0, return_inverse=True)
    F = inv.reshape(-1)[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    key = np.sort(F, axis=1)
    _, first, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    F = F[count[first.reshape(-1)] == 1]
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, ecount = np.unique(e, axis=0, return_counts=True)
    if not (ecount == 2).all():
        raise RuntimeError(f"{(ecount != 2).sum()} open or non-manifold edges")
    return V[F]


def write_stl(path, man):
    tri = float32_faces(man).astype("<f4")
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0).astype("<f4")
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                    ("attr", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(path.stem.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())


def bridge_pinches(man, gap=2e-4, r=0.12):
    """Two surfaces of a sculpt can graze each other within a fraction of a
    micron. That is valid in double precision, but float32 STL merges the
    two vertices into a non-manifold edge. Fill each such pinch with a
    speck of material (0.1 mm, invisible in print)."""
    from scipy.spatial import cKDTree
    mesh = man.to_mesh64()
    V = np.asarray(mesh.vert_properties)[:, :3]
    F = np.asarray(mesh.tri_verts)
    pairs = cKDTree(V).query_pairs(gap, output_type="ndarray")
    if len(pairs) == 0:
        return man
    edges = {tuple(e) for e in np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]],
                                                         F[:, [2, 0]]]), axis=1)}
    pinch = [V[a] for a, b in pairs if (min(a, b), max(a, b)) not in edges]
    if not pinch:
        return man
    specks = [Manifold.sphere(r, 12).translate(tuple(p)) for p in pinch]
    return Manifold.batch_boolean([man] + specks, OpType.Add)


def on_bed(man):
    """Centre on the Z axis and rest on z = 0. Slivers are collapsed and
    zero-volume crumbs that booleans of fine detail leave behind are
    dropped first."""
    man = man.simplify(1e-3)
    man = Manifold.compose([p for p in man.decompose() if p.volume() > 0.5])
    man = bridge_pinches(man)
    x0, y0, z0, x1, y1, _ = man.bounding_box()
    return man.translate((-(x0 + x1) / 2, -(y0 + y1) / 2, -z0))


def report(name, m):
    x0, y0, z0, x1, y1, z1 = m.bounding_box()
    print(f"{name:14s} {x1 - x0:6.1f} x {y1 - y0:6.1f} x {z1 - z0:6.1f} mm  "
          f"{m.volume() / 1000:6.1f} cm3  {m.num_tri():7d} tris")
