#!/usr/bin/env python3
"""
Measures a Bambu Studio 3MF project the way a slicer sees it: part sizes,
wall thickness, and — the point of the exercise — which layers actually
have material hanging in the air.

    python3 tools/analyze_3mf.py Spirallampe_E27_02_A1.3mf

For every object it prints the bounding box, the solid volume, the mean
wall thickness, and then walks the part layer by layer comparing each
slice with the one below it. A layer whose new material sits more than a
millimetre away from anything underneath needs support; everything else
is a normal self-supporting slope and does not.

Needs numpy and scipy.
"""
import argparse
import json
import re
import sys
import zipfile

import numpy as np
from scipy import ndimage

VERTEX = re.compile(rb'<vertex x="([-0-9.eE+]+)" y="([-0-9.eE+]+)" z="([-0-9.eE+]+)"')
TRIANGLE = re.compile(rb'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"')
PIXEL = 0.2  # raster resolution for the layer comparison, mm


def load_meshes(path):
    """Yield (label, vertices, faces) for every mesh, oriented as it prints.

    A 3MF stores each mesh in its own coordinate system and places it on the
    plate with a transform in the build section. Bambu routinely flips a part
    over to print it — the lamp base is stored disc-up and printed disc-down —
    so the transform has to be applied, or the overhang analysis describes an
    orientation nobody prints.
    """
    zf = zipfile.ZipFile(path)
    root = zf.read("3D/3dmodel.model").decode()

    placement = {m.group(1): np.array(m.group(2).split(), float)[:9].reshape(3, 3)
                 for m in re.finditer(r'<item objectid="(\d+)"[^>]*transform="([^"]+)"', root)}

    names = {}
    try:
        cfg = zf.read("Metadata/model_settings.config").decode()
        for obj in re.finditer(r'<object id="(\d+)">(.*?)</object>', cfg, re.S):
            m = re.search(r'key="name" value="([^"]+)"', obj.group(2))
            if m:
                names[obj.group(1)] = m.group(1)
    except KeyError:
        pass

    # each mesh file is reached through a component of a placed root object
    owner = {}
    for obj in re.finditer(r'<object id="(\d+)"[^>]*>\s*<components>(.*?)</components>',
                           root, re.S):
        for comp in re.finditer(r'<component p:path="/([^"]+)"', obj.group(2)):
            owner[comp.group(1)] = obj.group(1)

    for entry in sorted(zf.namelist()):
        if not entry.startswith("3D/Objects/"):
            continue
        raw = zf.read(entry)
        V = np.array(VERTEX.findall(raw), dtype=np.float64)
        F = np.array(TRIANGLE.findall(raw), dtype=np.int64)
        if not len(V) or not len(F):
            continue
        oid = owner.get(entry)
        M = placement.get(oid)
        if M is not None:
            V = V @ M
            if np.linalg.det(M) < 0:      # a mirroring transform flips winding
                F = F[:, ::-1]
        label = names.get(oid) or entry.rsplit("/", 1)[-1]
        yield label, V, F


def bulk(V, F):
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1).sum() / 2
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    return vol, area


def contour(V, F, z):
    """Segments where the mesh crosses the plane z."""
    tri = V[F]
    n_above = (tri[:, :, 2] > z).sum(1)
    out = []
    for p in tri[(n_above == 1) | (n_above == 2)]:
        pts = []
        for i in range(3):
            a, b = p[i], p[(i + 1) % 3]
            if (a[2] > z) != (b[2] > z):
                pts.append((a + (z - a[2]) / (b[2] - a[2]) * (b - a))[:2])
        if len(pts) == 2:
            out.append(pts)
    return np.array(out)


def fill(segs, x0, y0, w, h):
    """Even-odd scanline fill of a cross-section into a boolean grid."""
    grid = np.zeros((h, w), bool)
    if not len(segs):
        return grid
    a, b = segs[:, 0], segs[:, 1]
    ys = np.arange(h) * PIXEL + y0 + PIXEL / 2

    # the scanlines each segment crosses, i.e. rows with y_lo <= y < y_hi
    lo = np.minimum(a[:, 1], b[:, 1])
    hi = np.maximum(a[:, 1], b[:, 1])
    first = np.searchsorted(ys, lo, side="left")
    last = np.searchsorted(ys, hi, side="left")
    count = np.maximum(last - first, 0)
    if not count.sum():
        return grid

    seg = np.repeat(np.arange(len(segs)), count)
    within = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    row = np.repeat(first, count) + within

    p, q = a[seg], b[seg]
    t = (ys[row] - p[:, 1]) / (q[:, 1] - p[:, 1])
    x = p[:, 0] + t * (q[:, 0] - p[:, 0])

    # sort crossings along each scanline, then fill between consecutive pairs
    order = np.lexsort((x, row))
    row, x = row[order], x[order]
    left = np.clip(((x[0::2] - x0) / PIXEL).astype(int), 0, w)
    right = np.clip(((x[1::2] - x0) / PIXEL).astype(int), 0, w)

    spans = np.zeros((h, w + 1), np.int32)
    np.add.at(spans, (row[0::2], left), 1)
    np.add.at(spans, (row[0::2], right), -1)
    return spans.cumsum(1)[:, :w] > 0


def unsupported(V, F, layer, flip=False):
    """Walk the part layer by layer; return (plate_z, area, overhang) rows."""
    P = V.copy()
    if flip:
        P[:, 2] = -P[:, 2]
    P[:, 2] -= P[:, 2].min()
    lo = P[:, :2].min(0) - 2
    hi = P[:, :2].max(0) + 2
    w = int((hi[0] - lo[0]) / PIXEL)
    h = int((hi[1] - lo[1]) / PIXEL)
    rows = []
    below = None
    # start one layer up: the bottom layer rests on the build plate, so it is
    # supported by definition and would otherwise read as a huge overhang
    for z in np.arange(2 * layer, P[:, 2].max(), layer):
        cur = fill(contour(P, F, z), lo[0], lo[1], w, h)
        if below is None:
            below = fill(contour(P, F, z - layer), lo[0], lo[1], w, h)
        new = cur & ~below
        reach = (ndimage.distance_transform_edt(~below) * PIXEL)[new].max() if new.any() else 0.0
        rows.append((z, cur.sum() * PIXEL ** 2, new.sum() * PIXEL ** 2, reach))
        below = cur
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="the .3mf project file")
    ap.add_argument("--layer", type=float, default=0.2, help="layer height, mm")
    ap.add_argument("--flip", action="store_true",
                    help="turn every part over, on top of its stored placement")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="overhang reach that counts as needing support, mm")
    ap.add_argument("--settings", action="store_true",
                    help="also print the slicer profile stored in the project")
    args = ap.parse_args()

    if args.settings:
        cfg = json.load(zipfile.ZipFile(args.archive).open("Metadata/project_settings.config"))
        for key in ("printer_model", "print_settings_id", "filament_type",
                    "layer_height", "wall_loops", "sparse_infill_density",
                    "sparse_infill_pattern", "enable_support", "support_type",
                    "support_threshold_angle", "nozzle_temperature",
                    "outer_wall_speed", "seam_position"):
            print(f"  {key:26} = {cfg.get(key, '-')}")
        print()

    for name, V, F in load_meshes(args.archive):
        vol, area = bulk(V, F)
        size = V.max(0) - V.min(0)
        print(f"{name}")
        print(f"  {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, "
              f"{vol/1000:.1f} cm3 solid, {vol/1000*1.24:.0f} g in PLA, "
              f"mean wall {2*vol/area:.2f} mm")
        rows = unsupported(V, F, args.layer, args.flip)
        bad = [r for r in rows if r[3] > args.threshold]
        if bad:
            for z, a, n, reach in bad:
                print(f"  needs support at plate z {z:6.1f} mm: "
                      f"{n:.0f} mm2 in the air, up to {reach:.1f} mm from anything below")
        else:
            print(f"  self-supporting: worst overhang reaches only "
                  f"{max(r[3] for r in rows):.2f} mm")
        print()


if __name__ == "__main__":
    sys.exit(main())
