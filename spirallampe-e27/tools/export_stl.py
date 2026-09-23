#!/usr/bin/env python3
"""
Exports every part of a Bambu Studio / Orca 3MF project as a binary STL,
already turned the way the project prints it and dropped onto z = 0.

    python3 tools/export_stl.py project.3mf -o out/

A 3MF keeps each mesh in its own coordinates and places it on the plate
with a transform, so a part exported straight out of the archive can come
out upside down — the lamp base is stored disc-up and printed disc-down.
Applying the transform means the STL lands in a slicer the same way it
sits in the original project, with nothing to rotate by hand.

Needs numpy.
"""
import argparse
import re
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_3mf import load_meshes  # noqa: E402


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "part"


def write_stl(path, V, F):
    tri = V[F].astype("<f4")
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0).astype("<f4")

    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                    ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(path.stem.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("-o", "--out", default="stl_export", help="output directory")
    ap.add_argument("--keep-origin", action="store_true",
                    help="leave the coordinates alone instead of centring on "
                         "the Z axis and resting the part on z = 0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for label, V, F in load_meshes(args.archive):
        if not args.keep_origin:
            lo, hi = V.min(0), V.max(0)
            V = V - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
        path = out / f"{safe(label)}.stl"
        write_stl(path, V, F)
        size = V.max(0) - V.min(0)
        print(f"{path.name:34} {len(F):7,} triangles  "
              f"{size[0]:6.1f} x {size[1]:6.1f} x {size[2]:6.1f} mm  "
              f"{path.stat().st_size/1e6:5.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
