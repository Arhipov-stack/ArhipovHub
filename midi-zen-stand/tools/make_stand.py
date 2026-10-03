#!/usr/bin/env python3
"""
Builds the "zen garden" stand for Yandex Station Midi and writes the parts
as binary STL, each one already turned the way it prints and resting on z = 0.

    python3 tools/make_stand.py                 # -> stl/
    python3 tools/make_stand.py --assembly DIR  # also dump the parts in place

The station is the big rock of a karesansui garden: it sits in a pocket in
raked gravel, the rakes run in rings around it and around two moss islands,
and straight lines fill the rest. One island carries three stones, the other
a small bamboo grove. Base, moss and stones are separate parts so the garden
can be printed in three colours on a single-extruder printer.

Coordinates while building: x to the right, y towards the wall, z up,
origin at the front-left corner of the base.

Needs numpy, shapely and manifold3d.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from manifold3d import CrossSection, JoinType, Manifold, Mesh

# --- station --------------------------------------------------------------
STATION = 96.0        # footprint of Station Midi, mm (square)
CLEARANCE = 1.25      # gap per side between station and pocket wall
POCKET_R = 12.0       # pocket corner radius; must not exceed the station's
POCKET = STATION + 2 * CLEARANCE

# --- base -----------------------------------------------------------------
W, D = 160.0, 124.0   # footprint of the stand
CORNER_R = 16.0
RIM_W = 3.5           # border stones around the gravel
RIM_TOP = 12.0
CHAMFER = 0.8
FLOOR_Z = 3.0         # the station stands on this
FLOOR_HOLE_D = 60.0   # open centre under the station
SAND_Z = 9.0          # mean gravel level
RIPPLE_A = 0.8        # half the ridge-to-groove depth
RIPPLE_P = 4.4        # rake pitch
RINGS = 3             # rings raked around every rock before the straight lines
NOTCH_W = 46.0        # cable gap in the back wall

STX = W - RIM_W - 4.0 - POCKET / 2      # station centre
STY = D - RIM_W - 3.5 - POCKET / 2

# --- moss islands -----------------------------------------------------------
SEAT_Z = 5.0          # islands drop into recesses down to this level
ISLAND_CLEAR = 0.25
MOSS_EDGE = 5.6       # island height at its edge, above the seat
MOSS_DOME = 2.4
PEG_D, PEG_H = 6.0, 3.6          # base peg under the bamboo island
PEG_HOLE_D, PEG_HOLE_H = 6.5, 4.0

STONE_ISLAND = dict(cx=25.0, cy=32.0, rx=16.5, ry=12.5, rot=14, seed=3)
BAMBOO_ISLAND = dict(cx=23.0, cy=95.0, rx=14.0, ry=13.0, rot=-8, seed=7)

# --- bamboo (heights above the shelf) -----------------------------------------
STALKS = [  # dx, dy from island centre, radius, top, lean deg, lean azimuth deg
    (-3.5, 2.5, 5.0, 118.0, 1.5, 200),
    (5.5, 4.5, 4.3, 93.0, 3.0, 40),
    (1.0, -6.5, 3.7, 70.0, 4.5, 250),
]
TWIGS = [(250, 140), (110, 60), (205,)]   # leaf clusters per stalk, from the top node down
LEAVES = [(-48, 48, 18.0), (-14, 61, 23.0), (16, 54, 21.0), (47, 47, 17.0)]  # azimuth, elevation, length

GRID = 0.4            # heightfield resolution of the gravel, mm


# =============================================================================
# small helpers

def rrect(w, h, r, cx=0.0, cy=0.0, seg=40):
    pts = []
    for x, y, a0 in ((cx + w / 2 - r, cy + h / 2 - r, 0),
                     (cx - w / 2 + r, cy + h / 2 - r, 90),
                     (cx - w / 2 + r, cy - h / 2 + r, 180),
                     (cx + w / 2 - r, cy - h / 2 + r, 270)):
        a = np.radians(a0 + np.linspace(0, 90, seg + 1))
        pts.append(np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1))
    return np.concatenate(pts)


def blob(cx, cy, rx, ry, rot, seed, n=200):
    """An irregular oval, the outline of a moss island."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(t)
    for k, amp in ((2, 0.05), (3, 0.06), (4, 0.03), (5, 0.02)):
        r += amp * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    x, y = rx * r * np.cos(t), ry * r * np.sin(t)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def cs(pts):
    return CrossSection([np.asarray(pts, dtype=float)])


def prism(section, z0, z1):
    return section.extrude(z1 - z0).translate((0, 0, z0))


def flared(pts, z0, z1, grow):
    """Prism over a centred outline whose top is offset outwards by `grow`
    (negative shrinks it) - a chamfer for rounded rectangles."""
    pts = np.asarray(pts, float)
    c = (pts.max(0) + pts.min(0)) / 2
    w, h = pts.max(0) - pts.min(0)
    m = cs(pts - c).extrude(z1 - z0, scale_top=((w + 2 * grow) / w,
                                                  (h + 2 * grow) / h))
    return m.translate((c[0], c[1], z0))


def sdf_rrect(X, Y, cx, cy, hw, hh, r):
    qx = np.abs(X - cx) - (hw - r)
    qy = np.abs(Y - cy) - (hh - r)
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return out + np.minimum(np.maximum(qx, qy), 0) - r


def sdf_poly(X, Y, pts):
    """Signed distance to a polygon outline, negative inside."""
    poly = Polygon(pts)
    x, y = X.ravel(), Y.ravel()
    d = shapely.distance(shapely.points(x, y), poly.exterior)
    inside = shapely.contains_xy(poly, x, y)
    return np.where(inside, -d, d).reshape(X.shape)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smin(ds, k):
    ds = np.stack(ds)
    m = ds.min(0)
    return m - k * np.log(np.exp(-(ds - m) / k).sum(0))


def heightfield(xs, ys, H):
    """Watertight solid between z = 0 and z = H over a regular grid."""
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    tris = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]

    loop = np.concatenate([idx[:, 0], idx[-1, 1:], idx[-2::-1, -1],
                           idx[0, -2:0:-1]])
    nb = len(loop)
    bot = top[loop].copy()
    bot[:, 2] = 0
    centre = np.array([[xs.mean(), ys.mean(), 0.0]])
    verts = np.concatenate([top, bot, centre])
    t0, t1 = loop, np.roll(loop, -1)
    b0 = nx * ny + np.arange(nb)
    b1 = np.roll(b0, -1)
    ci = np.full(nb, len(verts) - 1)
    tris += [np.stack([t0, b0, b1], 1), np.stack([t0, b1, t1], 1),
             np.stack([ci, b1, b0], 1)]
    mesh = Mesh(vert_properties=verts.astype(np.float32),
                tri_verts=np.concatenate(tris).astype(np.uint32))
    return Manifold(mesh)


def capsule(p0, p1, r, seg=12):
    s = Manifold.sphere(r, seg)
    return Manifold.batch_hull([s.translate(tuple(p0)), s.translate(tuple(p1))])


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


def on_bed(man):
    """Centre on the Z axis and rest on z = 0. Slivers thinner than a micron
    are collapsed first: STL stores float32, and two vertices that close
    would merge in the slicer into a non-manifold edge."""
    man = man.simplify(1e-3)
    x0, y0, z0, x1, y1, _ = man.bounding_box()
    return man.translate((-(x0 + x1) / 2, -(y0 + y1) / 2, -z0))


# =============================================================================
# moss

def moss_surface(pts, X, Y, seed):
    """Height of a moss island above its seat: a low dome of small clumps."""
    rng = np.random.default_rng(seed)
    d_in = -sdf_poly(X, Y, pts)
    x0, y0 = pts.min(0) - 2
    x1, y1 = pts.max(0) + 2
    n = int((x1 - x0) * (y1 - y0) / 2.2)
    cx, cy = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    amp, sig = rng.uniform(0.45, 0.85, n), rng.uniform(0.9, 1.5, n)
    bumps = np.zeros_like(X)
    for i in range(n):
        g = amp[i] * np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2)
                            / (2 * sig[i] ** 2))
        np.maximum(bumps, g, out=bumps)
    edge = np.clip(d_in / 0.8, 0, 1)
    rounded = MOSS_EDGE - 0.8 * (1 - np.sqrt(1 - (1 - edge) ** 2))
    dome = MOSS_DOME * (1 - np.exp(-np.maximum(d_in, 0) / 4.0))
    return rounded + dome + bumps * smoothstep(0.3, 2.5, d_in)


def moss_island(pts, seed):
    """Island solid in world coordinates (bottom on the seat) and a function
    giving its top height at any point."""
    x0, y0 = pts.min(0) - 1
    x1, y1 = pts.max(0) + 1
    xs, ys = np.arange(x0, x1 + 0.25, 0.25), np.arange(y0, y1 + 0.25, 0.25)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = moss_surface(pts, X, Y, seed)
    solid = heightfield(xs, ys, np.maximum(H, 0.5)) ^ prism(cs(pts), -1, 30)

    def top_at(x, y):
        i = np.clip(np.round((x - x0) / 0.25).astype(int), 0, len(xs) - 1)
        j = np.clip(np.round((y - y0) / 0.25).astype(int), 0, len(ys) - 1)
        return SEAT_Z + H[i, j]

    return solid.translate((0, 0, SEAT_Z)), top_at


# =============================================================================
# stones

def stone(rx, ry, rz, seed, lean=(0.0, 0.0), taper=0.3):
    """A rugged garden stone: the hull of a jittered ellipsoid, narrowing
    upwards, with its softer edges rounded off and the bottom cut flat."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(30, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.85, 1.08, (len(v), 1))
    shrink = 1 - taper * (v[:, 2] + 1) / 2
    pts = np.stack([v[:, 0] * rx * shrink, v[:, 1] * ry * shrink, v[:, 2] * rz], 1)
    s = Manifold.hull_points(pts).smooth_out(40, 0.35).refine(5)
    s = s.rotate((lean[0], lean[1], rng.uniform(0, 180)))
    s = s.translate((0, 0, 0.45 * rz))
    return s.trim_by_plane((0, 0, 1), 0.0)


# =============================================================================
# bamboo

def stalk(r0, height, seed):
    """A culm standing on z = 0, cut flat just above its last node with a
    shallow hollow on top, like a cut bamboo."""
    rng = np.random.default_rng(seed)
    nodes, z = [], 0.0
    seg = 11.0
    while True:
        z += seg + rng.uniform(-1.0, 1.0)
        seg = min(seg + 3.0, 21.0)
        if z > height - 6:
            break
        nodes.append(z)
    top = height
    zs = np.unique(np.concatenate([np.linspace(0, top, int(top * 2)),
                                   *[n + np.linspace(-1.2, 1.2, 25) for n in nodes]]))
    zs = zs[(zs >= 0) & (zs <= top)]
    r = r0 * (1 - 0.12 * zs / top)
    for n in nodes:
        u = zs - n
        r = r + 0.45 * np.exp(-(u / 0.45) ** 2) - 0.18 * np.exp(-((u + 0.9) / 0.5) ** 2)
    rt = r[-1]
    outline = [(0.0, 0.0)] + list(zip(r, zs)) + [(rt - 1.0, top), (rt - 1.0, top - 1.4),
                                                 (0.0, top - 1.4)]
    return Manifold.revolve(cs(outline), 48), nodes


def leaf(base, azim, elev, length, width=5.0, thick=1.0):
    u = np.linspace(-1.0, length, 40)
    t = np.clip(u / length, 0, 1)
    half = np.maximum(width / 2 * np.sin(np.pi * t ** 0.65) ** 1.1, 0.0)
    half[0] = half[1] = 0.45
    outline = np.concatenate([np.stack([u, -half], 1),
                              np.stack([u[::-1], half[::-1]], 1)[1:]])
    plate = cs(outline).extrude(thick).translate((0, 0, -thick / 2))
    ph, el = np.radians(azim), np.radians(elev)
    a = np.array([np.cos(el) * np.cos(ph), np.cos(el) * np.sin(ph), np.sin(el)])
    b = np.array([-np.sin(ph), np.cos(ph), 0.0])
    n = np.cross(a, b)
    M = np.column_stack([a, b, n, base])
    return plate.transform(M)


def bamboo(island, cx, cy):
    parts = [island]
    for i, (dx, dy, r0, top, lean, lean_az) in enumerate(STALKS):
        z0 = SEAT_Z + 4.5
        s, nodes = stalk(r0, top - z0, seed=11 + i)
        ax = np.radians(lean_az)
        s = s.rotate((-lean * np.sin(ax), lean * np.cos(ax), 0))
        parts.append(s.translate((cx + dx, cy + dy, z0)))

        tilt = np.radians(lean)
        dirv = np.array([np.sin(tilt) * np.cos(ax), np.sin(tilt) * np.sin(ax), np.cos(tilt)])
        foot = np.array([cx + dx, cy + dy, z0])
        for zn, az in zip(nodes[::-1], TWIGS[i]):
            out = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
            p0 = foot + dirv * zn + out * (r0 - 0.6)
            p1 = p0 + 7.0 * (out * np.cos(np.radians(50)) + [0, 0, np.sin(np.radians(50))])
            parts.append(capsule(p0, p1, 0.85))
            for daz, el, ln in LEAVES:
                parts.append(leaf(p1, az + daz, el, ln))
    return Manifold.batch_boolean(parts, _union())


def _union():
    from manifold3d import OpType
    return OpType.Add


# =============================================================================
# base

def gravel_height(X, Y, islands):
    d_station = sdf_rrect(X, Y, STX, STY, POCKET / 2, POCKET / 2, POCKET_R)
    ds = [d_station] + [sdf_poly(X, Y, p) + ISLAND_CLEAR for p in islands]
    f = smin(ds, 1.2)
    rings = np.cos(2 * np.pi * np.maximum(f, 0) / RIPPLE_P)
    lines = np.cos(2 * np.pi * (Y - STY) / RIPPLE_P)
    edge = (RINGS + 0.5) * RIPPLE_P
    w = smoothstep(edge - 0.2 * RIPPLE_P, edge + 0.2 * RIPPLE_P, f)
    return SAND_Z + RIPPLE_A * ((1 - w) * rings + w * lines)


def base(islands):
    outer = rrect(W, D, CORNER_R, W / 2, D / 2)
    inner = rrect(W - 2 * RIM_W, D - 2 * RIM_W, CORNER_R - RIM_W, W / 2, D / 2)

    body = (prism(cs(outer), 0, RIM_TOP - CHAMFER)
            + flared(outer, RIM_TOP - CHAMFER, RIM_TOP, -CHAMFER))
    hole = (prism(cs(inner), -1, RIM_TOP - CHAMFER)
            + flared(inner, RIM_TOP - CHAMFER, RIM_TOP + 2, CHAMFER * (2 + CHAMFER) / CHAMFER))
    rim = body - hole

    xs = np.arange(RIM_W - 1.0, W - RIM_W + 1.0 + GRID, GRID)
    ys = np.arange(RIM_W - 1.0, D - RIM_W + 1.0 + GRID, GRID)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    gravel = heightfield(xs, ys, gravel_height(X, Y, islands))
    gravel = gravel ^ prism(cs(inner).offset(0.5, JoinType.Round), -1, 30)
    b = rim + gravel

    pocket = rrect(POCKET, POCKET, POCKET_R, STX, STY)
    b -= prism(cs(pocket), FLOOR_Z, 40)
    b -= Manifold.cylinder(10, FLOOR_HOLE_D / 2, FLOOR_HOLE_D / 2, 128).translate(
        (STX, STY, -1))
    b -= Manifold.cube((NOTCH_W, D, 40)).translate(
        (STX - NOTCH_W / 2, STY, FLOOR_Z))
    for p in islands:
        b -= prism(cs(p).offset(ISLAND_CLEAR, JoinType.Round), SEAT_Z, 40)
    bi = BAMBOO_ISLAND
    b += Manifold.cylinder(PEG_H + 0.5, PEG_D / 2, PEG_D / 2 - 0.3, 48).translate(
        (bi["cx"], bi["cy"], SEAT_Z - 0.5))
    return b


def fit_test():
    """A thin frame with the exact pocket: drop the station in before
    committing to the full base."""
    inner = rrect(POCKET, POCKET, POCKET_R)
    outer = rrect(POCKET + 8, POCKET + 8, POCKET_R + 4)
    return prism(cs(outer), 0, 4) - prism(cs(inner), -1, 5)


def station_mock():
    """Rough stand-in for the station, only for renders: body, light ring
    and top panel."""
    z = FLOOR_Z
    body = prism(cs(rrect(STATION, STATION, 16, STX, STY)), z, z + 106)
    ring = prism(cs(rrect(STATION - 3, STATION - 3, 14.5, STX, STY)), z + 106, z + 107.6)
    top = prism(cs(rrect(STATION - 2, STATION - 2, 15, STX, STY)), z + 107.6, z + 110)
    return body, ring, top


# =============================================================================

def build():
    pa = blob(**STONE_ISLAND)
    pb = blob(**BAMBOO_ISLAND)

    moss_a, top_a = moss_island(pa, STONE_ISLAND["seed"])
    moss_b, _ = moss_island(pb, BAMBOO_ISLAND["seed"])

    # three stones, tall one at the back of the island
    cx, cy = STONE_ISLAND["cx"], STONE_ISLAND["cy"]
    stone_specs = [  # dx, dy, rx, ry, rz, seed, lean
        (-3.0, 3.0, 6.2, 5.0, 11.0, 21, (6, -4)),
        (6.5, -2.5, 6.0, 4.8, 5.5, 22, (0, 8)),
        (-8.5, -4.5, 4.2, 3.5, 3.4, 23, (0, 0)),
    ]
    stones_world, stones_print = [], []
    for dx, dy, rx, ry, rz, seed, lean in stone_specs:
        s = stone(rx, ry, rz, seed, lean)
        x, y = cx + dx, cy + dy
        level = float(top_a(np.array(x), np.array(y)))
        sink = min(0.45 * rz, level - SEAT_Z - 1.4)
        zb = level - sink
        placed = s.translate((x, y, zb))
        below = placed.trim_by_plane((0, 0, -1), -(zb + sink + 0.8))
        seat = CrossSection(below.project().to_polygons()).offset(0.25, JoinType.Round)
        moss_a -= prism(seat, zb, zb + 30)
        stones_world.append(placed)
        stones_print.append(s)

    bamboo_part = bamboo(moss_b, BAMBOO_ISLAND["cx"], BAMBOO_ISLAND["cy"])
    bamboo_part -= Manifold.cylinder(PEG_HOLE_H, PEG_HOLE_D / 2, PEG_HOLE_D / 2, 48).translate(
        (BAMBOO_ISLAND["cx"], BAMBOO_ISLAND["cy"], SEAT_Z - 0.01))

    base_part = base([pa, pb])

    # stones laid out in a row for the plate
    row, x = [], 0.0
    for s in stones_print:
        x0, _, _, x1, _, _ = s.bounding_box()
        row.append(s.translate((x - x0, 0, 0)))
        x += (x1 - x0) + 4
    stones_plate = Manifold.batch_boolean(row, _union())

    world = {
        "base": base_part,
        "moss_stones": moss_a,
        "stones": Manifold.batch_boolean(stones_world, _union()),
        "bamboo": bamboo_part,
        **dict(zip(("station_mock", "station_glow", "station_top"), station_mock())),
    }
    printed = {
        "base": base_part,
        "moss_stones": moss_a,
        "stones": stones_plate,
        "bamboo": bamboo_part,
        "fit_test": fit_test(),
    }
    return world, printed


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(here / "stl"))
    ap.add_argument("--assembly", help="also write the parts in assembled position here")
    args = ap.parse_args()

    world, printed = build()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in printed.items():
        m = on_bed(m)
        write_stl(out / f"{name}.stl", m)
        x0, y0, z0, x1, y1, z1 = m.bounding_box()
        print(f"{name:12s} {x1 - x0:6.1f} x {y1 - y0:6.1f} x {z1 - z0:6.1f} mm  "
              f"{m.volume() / 1000:6.1f} cm3  {m.num_tri():7d} tris  genus {m.genus()}")
    if args.assembly:
        a = Path(args.assembly)
        a.mkdir(parents=True, exist_ok=True)
        for name, m in world.items():
            write_stl(a / f"{name}.stl", m.translate((-W / 2, -D / 2, 0)))


if __name__ == "__main__":
    main()
