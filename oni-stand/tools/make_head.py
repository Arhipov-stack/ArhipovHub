#!/usr/bin/env python3
"""
Variant 1: the stand is the head of a red oni. Station Midi stands on its
crown in a shallow pocket, two ivory horns rise on either side of it.

    python3 tools/make_head.py                 # -> stl/head/
    python3 tools/make_head.py --assembly DIR  # also dump the parts in place

Parts and colours:
  head      red      prints lying on its back, face up: no supports
  eyes      yellow   two inserts, back down
  teeth     ivory    clenched teeth with tusks, back down
  horns     ivory    flat base down
  pins      any      two dowels that hold the horns

Needs numpy, shapely, scikit-image and manifold3d.
"""
import argparse
from pathlib import Path

import numpy as np
from manifold3d import Manifold, OpType

from common import (CLEARANCE, NOTCH_W, POCKET, POCKET_R, STATION, STATION_H, box, cs,
                    fit_test, mesh_field, on_bed, prism, report, rrect, write_stl)
from oni_face import TOP, Planes, eye_field, head_field, teeth_field

STEP = 0.4
BACK = 60.0                 # flat back of the head
ST_Y = 3.0                  # station centre
POCKET_DEPTH = 3.0          # shallow: lying on its back the lip prints as a short overhang
FLOOR = TOP - POCKET_DEPTH
HORN_X, HORN_Y = 57.0, -40.0
PIN_R, PIN_HOLE_R, PIN_DEPTH = 2.9, 3.15, 10.0


def field_on(fn, xs, zs):
    planes = Planes(xs, zs)
    return lambda X, Y, Z, info: fn(X, Y, Z, planes.take(info[2]))


def sculpt(fn, lo, hi, step=STEP):
    xs = np.arange(lo[0], hi[0] + step / 2, step)
    zs = np.arange(lo[2], hi[2] + step / 2, step)
    return mesh_field(field_on(fn, xs, zs), lo, hi, step)


def horn(side):
    """A ringed horn on a flat base, leaning out and hooking back in at the
    tip, never more than ~40 degrees off vertical so it prints standing."""
    L, n = 62.0, 64
    t = np.linspace(0, 1, n)
    theta = np.radians(16 - 52 * t ** 1.25)        # off vertical, + = outwards
    phi = np.radians(-16 * t ** 2)                 # lean to the front
    step = L / (n - 1)
    d = np.stack([np.sin(theta), np.sin(phi), np.cos(theta) * np.cos(phi)], 1)
    p = np.concatenate([[np.zeros(3)], np.cumsum(d[:-1] * step, 0)])
    r = 7.2 * (1 - t) ** 0.8 + 0.6
    r = r * (1 + 0.05 * np.cos(2 * np.pi * t * 10) * (t < 0.85))
    balls = [Manifold.sphere(ri, 48).translate(tuple(pi)) for pi, ri in zip(p, r)]
    segs = [Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])]
    h = Manifold.batch_boolean(segs, OpType.Add).trim_by_plane((0, 0, 1), 0.0)
    h -= Manifold.cylinder(PIN_DEPTH - 2 + 0.01, PIN_HOLE_R, PIN_HOLE_R, 40).translate((0, 0, -0.01))
    if side < 0:
        h = h.mirror((1, 0, 0))
    return h


def pin():
    c = 0.5
    body = Manifold.cylinder(2 * PIN_DEPTH - 2 - 2 * c, PIN_R, PIN_R, 40).translate((0, 0, c))
    ends = [Manifold.cylinder(c, PIN_R - c, PIN_R, 40),
            Manifold.cylinder(c, PIN_R, PIN_R - c, 40).translate((0, 0, 2 * PIN_DEPTH - 2 - c))]
    return body + ends[0] + ends[1]


def station_mock(floor, cy):
    z = floor
    body = prism(cs(rrect(STATION, STATION, 16, 0, cy)), z, z + STATION_H - 4)
    ring = prism(cs(rrect(STATION - 3, STATION - 3, 14.5, 0, cy)), z + STATION_H - 4,
                 z + STATION_H - 2.4)
    top = prism(cs(rrect(STATION - 2, STATION - 2, 15, 0, cy)), z + STATION_H - 2.4,
                z + STATION_H)
    return body, ring, top


def build():
    raw = sculpt(head_field, (-70, -84, -6), (70, 68, 60))
    head = raw ^ box((-80, -100, 0), (80, BACK, TOP))
    head -= prism(cs(rrect(POCKET, POCKET, POCKET_R, 0, ST_Y)), FLOOR, TOP + 5)
    head -= box((-NOTCH_W / 2, ST_Y, FLOOR), (NOTCH_W / 2, BACK + 5, TOP + 5))
    for s in (-1, 1):
        head -= Manifold.cylinder(PIN_DEPTH + 1, PIN_HOLE_R, PIN_HOLE_R, 40).translate(
            (s * HORN_X, HORN_Y, TOP - PIN_DEPTH))

    eyes = sculpt(eye_field, (-38, -76, 22), (38, -48, 42))
    teeth = sculpt(teeth_field, (-34, -86, 1), (34, -48, 24))
    horns = [horn(s).translate((s * HORN_X, HORN_Y, TOP)) for s in (-1, 1)]

    world = {"head": head, "eyes": eyes, "teeth": teeth,
             "horns": Manifold.batch_boolean(horns, OpType.Add)}
    world.update(zip(("station_mock", "station_glow", "station_top"),
                     station_mock(FLOOR, ST_Y)))

    lay = lambda m: m.rotate((-90, 0, 0))          # back of every face part down
    plate_horns = Manifold.batch_boolean(
        [horn(-1).translate((-16, 0, 0)), horn(1).translate((16, 0, 0))], OpType.Add)
    printed = {"head": lay(head), "eyes": lay(eyes), "teeth": lay(teeth),
               "horns": plate_horns,
               "pins": pin().translate((-5, 0, 0)) + pin().translate((5, 0, 0)),
               "fit_test": fit_test()}
    return world, printed


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(here / "stl" / "head"))
    ap.add_argument("--assembly", help="also write the parts in assembled position here")
    args = ap.parse_args()

    world, printed = build()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in printed.items():
        m = on_bed(m)
        write_stl(out / f"{name}.stl", m)
        report(name, m)
    if args.assembly:
        a = Path(args.assembly)
        a.mkdir(parents=True, exist_ok=True)
        for name, m in world.items():
            write_stl(a / f"{name}.stl", m.simplify(1e-3))


if __name__ == "__main__":
    main()
