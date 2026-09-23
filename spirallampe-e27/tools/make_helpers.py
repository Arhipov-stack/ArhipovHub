#!/usr/bin/env python3
"""
Generates the helper bodies for printing "Spirallampe E27" (Felix3DPrint).

Three cylinders, each in the coordinate system of the original 3MF object it
belongs to, so that they land in the right place when loaded as a part in
Bambu Studio / OrcaSlicer:

  enforcer_shade.stl        support enforcer, fills the bore under the
                            clamping ledge (the only ceiling in the shade)
  enforcer_ring.stl         support enforcer, fills the bore under the
                            socket lip of ring C
  modifier_shade_ledge.stl  modifier, marks the clamping ledge so it can be
                            given solid infill and extra walls

Heights are given in plate Z (0 = build plate) and converted to the object's
own Z, which is centred on the object for every part of this project.
"""
import struct
from pathlib import Path

SEG = 128  # facets around the circumference


def cylinder(radius, z_lo, z_hi, seg=SEG):
    """Closed cylinder on the Z axis. Returns a list of (n, v0, v1, v2)."""
    import math
    ring = [(radius * math.cos(2 * math.pi * i / seg),
             radius * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    tris = []
    for i in range(seg):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % seg]
        # side (two triangles, outward normal)
        tris.append(((x0, y0, z_lo), (x1, y1, z_lo), (x1, y1, z_hi)))
        tris.append(((x0, y0, z_lo), (x1, y1, z_hi), (x0, y0, z_hi)))
        # bottom cap (normal -Z) and top cap (normal +Z)
        tris.append(((0.0, 0.0, z_lo), (x1, y1, z_lo), (x0, y0, z_lo)))
        tris.append(((0.0, 0.0, z_hi), (x0, y0, z_hi), (x1, y1, z_hi)))
    return tris


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return nx / ln, ny / ln, nz / ln


def write_stl(path, tris, header):
    with open(path, "wb") as f:
        f.write(header.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<3f", *normal(a, b, c)))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


# (file, object half-height, plate z from, plate z to, radius, comment)
PARTS = [
    # The shade's ledge starts at plate z 25.000 and ring C's lip at 27.900.
    # Each enforcer runs a little past its ledge: no support can be generated
    # inside solid model anyway, and overshooting guarantees the column
    # reaches the underside.
    ("enforcer_shade.stl",       100.00,  0.0, 26.0, 34.0,
     "support enforcer: shade bore under the ledge at z 25.0"),
    ("enforcer_ring.stl",         14.95,  2.4, 28.5, 31.0,
     "support enforcer: ring C bore under the lip at z 27.9"),
    ("modifier_shade_ledge.stl", 100.00, 24.8, 28.2, 40.0,
     "modifier: solid infill zone over the ledge, plate z 25.0-28.0"),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "stl"
    out.mkdir(exist_ok=True)
    for name, half, z0, z1, r, comment in PARTS:
        tris = cylinder(r, z0 - half, z1 - half)
        write_stl(out / name, tris, comment)
        print(f"{name:28} d{2*r:5.1f} mm  plate z {z0:5.1f}..{z1:5.1f}  "
              f"object z {z0-half:7.2f}..{z1-half:7.2f}  {len(tris)} tris")
