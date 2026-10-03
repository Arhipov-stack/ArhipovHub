#!/usr/bin/env python3
"""
Variant 2: a small Shinto shrine. Station Midi stands on a wooden podium
on a stone plinth; two vermilion posts rise at the front corners, a twisted
shimenawa rope with paper shide hangs between them, and under the rope
hangs the mask of a red oni - the same face as the head stand, scaled down.

    python3 tools/make_shrine.py                 # -> stl/shrine/
    python3 tools/make_shrine.py --assembly DIR  # also dump the parts in place

Parts and colours:
  plinth   grey stone
  podium   dark wood (brown or black)
  posts    vermilion red
  rope     straw / ivory (rope and shide in one piece, flat back down)
  mask     red, flat back down; horns and fangs can be painted ivory

Needs numpy, shapely, scikit-image and manifold3d.
"""
import argparse
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

from common import (NOTCH_W, POCKET, POCKET_R, box, cs, fit_test, on_bed, prism, report,
                    rrect, write_stl)
from make_head import horn, sculpt, station_mock
from oni_face import eye_field, eye_outline, head_field, mouth_outline, teeth_field

# --- layout (mm) --------------------------------------------------------------
ST_Y = 4.0                      # station centre
PL_W, PL_D = 130.0, 124.0       # plinth footprint, lower step
STEP_IN = 3.5                   # upper step is this much smaller per side
PL_H1, PL_H2 = 4.0, 8.0         # top of lower and upper step
BODY_W, BODY_D = 112.0, 110.0
BODY_TOP = 46.0
POCKET_DEPTH = 4.0
FLOOR = BODY_TOP - POCKET_DEPTH
BODY_Y = ST_Y                   # podium centred under the station
FRONT = BODY_Y - BODY_D / 2     # front face of the podium
POST = 10.0
POST_TOP = 64.0
CAP = 14.0
ROPE_Z_END, ROPE_Z_MID = 42.0, 37.0
ROPE_R = 5.0
MASK_SCALE = 0.5
MASK_TOP = 45.0          # top edge of the mask (without horns)
MASK_STANDOFF = 6.5     # the mask hangs in front of the rope
SEAT = 0.8                      # podium sits this deep in the plinth


def chamfered_slab(w, d, z0, z1, c, cy=0.0):
    pts = rrect(w, d, 3.0)
    body = prism(cs(pts), 0, z1 - z0 - c)
    top = cs(pts).extrude(c, scale_top=((w - 2 * c) / w, (d - 2 * c) / d)).translate(
        (0, 0, z1 - z0 - c))
    return (body + top).translate((0, cy, z0))


def plinth():
    p = chamfered_slab(PL_W, PL_D, 0, PL_H1, 0.8, ST_Y)
    # a few joints in the lower step, so it reads as cut stone
    for x in (-38.0, 6.0, 44.0):
        for y0 in (ST_Y - PL_D / 2 - 1, ST_Y + PL_D / 2 - STEP_IN + 0.3):
            p -= box((x - 0.4, y0, PL_H1 - 1.6), (x + 0.4, y0 + STEP_IN + 0.7, PL_H1 + 0.1))
    p += chamfered_slab(PL_W - 2 * STEP_IN, PL_D - 2 * STEP_IN, PL_H1 - 0.01, PL_H2, 0.8, ST_Y)
    # seat for the podium and the posts
    p -= prism(cs(rrect(BODY_W + 0.6, BODY_D + 0.6, 1.3, 0, BODY_Y)), PL_H2 - SEAT, PL_H2 + 1)
    for s in (-1, 1):
        x, y = s * (BODY_W / 2 - POST / 2), FRONT + POST / 2
        p -= box((x - POST / 2 - 0.3, y - POST / 2 - 0.3, PL_H2 - SEAT),
                 (x + POST / 2 + 0.3, y + POST / 2 + 0.3, PL_H2 + 1))
    return p


def podium():
    z0 = PL_H2 - SEAT
    b = box((-BODY_W / 2, FRONT, z0), (BODY_W / 2, FRONT + BODY_D, BODY_TOP))
    # notches for the front posts
    for s in (-1, 1):
        x = s * (BODY_W / 2 - POST / 2)
        b -= box((x - POST / 2 - 0.3, FRONT - 1, z0 - 1),
                 (x + POST / 2 + 0.3, FRONT + POST + 0.3, BODY_TOP + 1))
    # planks: vertical joints on the front and the sides, a beam on top and a sill
    inner = BODY_W / 2 - POST - 0.3
    for x in np.arange(-inner + 9.0, inner - 4, 9.0):
        b -= box((x - 0.5, FRONT - 1, z0 + 5), (x + 0.5, FRONT + 0.7, BODY_TOP - 6))
    for y in np.arange(FRONT + POST + 9.0, FRONT + BODY_D - 4, 9.0):
        for s in (-1, 1):
            x = s * BODY_W / 2
            b -= box((x - 0.7, y - 0.5, z0 + 5), (x + 0.7, y + 0.5, BODY_TOP - 6))
    b -= box((-BODY_W, FRONT - 1, BODY_TOP - 6.5), (BODY_W, FRONT + 0.6, BODY_TOP - 6))
    b -= box((-BODY_W, FRONT - 1, z0 + 4.5), (BODY_W, FRONT + 0.6, z0 + 5.0))
    for s in (-1, 1):
        b -= box((s * BODY_W / 2 - 0.6, FRONT - 1, BODY_TOP - 6.5),
                 (s * BODY_W / 2 + 0.6, FRONT + BODY_D + 1, BODY_TOP - 6))
    # station pocket and cable gap
    b -= prism(cs(rrect(POCKET, POCKET, POCKET_R, 0, ST_Y)), FLOOR, BODY_TOP + 5)
    b -= box((-NOTCH_W / 2, ST_Y, FLOOR), (NOTCH_W / 2, FRONT + BODY_D + 5, BODY_TOP + 5))
    return b


def post(side):
    x, y = side * (BODY_W / 2 - POST / 2), FRONT + POST / 2
    p = box((x - POST / 2, y - POST / 2, PL_H2 - SEAT), (x + POST / 2, y + POST / 2, POST_TOP))
    # cap with a chamfered underside so it prints without support
    c = (CAP - POST) / 2
    under = cs(rrect(POST, POST, 0.5)).extrude(c, scale_top=(CAP / POST, CAP / POST))
    cap = under + prism(cs(rrect(CAP, CAP, 0.8)), c, c + 2.5)
    cap += cs(rrect(CAP, CAP, 0.8)).extrude(2.5, scale_top=(0.35, 0.35)).translate((0, 0, c + 2.5))
    p += cap.translate((x, y, POST_TOP))
    # a ring (nuki slot look) just under the cap
    p -= box((x - POST, y - POST / 2 - 1, POST_TOP - 5), (x + POST, y - POST / 2 + 0.6, POST_TOP - 4.4))
    return p


def rope_curve(t):
    """Centre line of the rope, t in [0, 1] from left post to right post."""
    x0 = BODY_W / 2 - POST
    x = -x0 + 2 * x0 * t
    z = ROPE_Z_END - (ROPE_Z_END - ROPE_Z_MID) * (1 - (2 * t - 1) ** 2)
    return x, z


def rope():
    """Three twisted straw strands, cut flat at the back (it lies against the
    podium), with four zigzag paper shide hanging from it."""
    yc = FRONT - 1.0
    n = 260
    t = np.linspace(0, 1, n)
    x, z = rope_curve(t)
    segs = []
    for k in range(3):
        ph = 2 * np.pi * (t * (2 * (BODY_W / 2 - POST)) / 14.0) + k * 2 * np.pi / 3
        px = x
        py = yc + 2.3 * np.cos(ph)
        pz = z + 2.3 * np.sin(ph)
        balls = [Manifold.sphere(2.7, 20).translate((a, b, c)) for a, b, c in zip(px, py, pz)]
        segs += [Manifold.batch_hull([u, v]) for u, v in zip(balls, balls[1:])]
    r = Manifold.batch_boolean(segs, OpType.Add)
    for xs in (-41.0, -32.0, 32.0, 41.0):
        tt = (xs + (BODY_W / 2 - POST)) / (2 * (BODY_W / 2 - POST))
        _, zt = rope_curve(tt)
        r += shide(xs, zt - ROPE_R + 1.5)
    return r.trim_by_plane((0, -1, 0), -FRONT)


def shide(x, ztop, steps=4, drop=5.5, w=3.6, t=1.2):
    """A zigzag paper streamer hanging from ztop, flat against the podium."""
    rects = [cs(rrect(1.6, 2.5, 0.2, x - 1.1, ztop - 0.8))]
    for i in range(steps):
        off = 1.1 if i % 2 else -1.1
        z1, z0 = ztop - i * drop + 0.3, ztop - (i + 1) * drop
        rects.append(cs(rrect(w, z1 - z0, 0.2, x + off, (z0 + z1) / 2)))
    sec = CrossSection.batch_boolean(rects, OpType.Add)
    return sec.extrude(t).rotate((90, 0, 0)).translate((0, FRONT + 0.01, 0))


def mask_outline():
    u = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    n = 2.7
    ox = 52 * np.sign(np.cos(u)) * np.abs(np.cos(u)) ** (2 / n)
    oz = 27 + 28 * np.sign(np.sin(u)) * np.abs(np.sin(u)) ** (2 / n)
    ox *= 1 - 0.12 * (oz < 27) * ((27 - oz) / 28) ** 2      # narrower at the jaw
    return np.stack([ox, oz], 1)


def along_y(section, y_back, depth):
    """Extrude an x-z outline forward (towards -y) from y_back."""
    return section.extrude(depth).rotate((90, 0, 0)).translate((0, y_back, 0))


def mask():
    """The oni face of the head stand with its inserts, cut out as a mask,
    small horns on top, scaled down. A tapered block behind it keeps it off
    the podium so it hangs in front of the rope and still prints flat."""
    face = sculpt(head_field, (-60, -84, -4), (60, -44, 56))
    face += sculpt(eye_field, (-38, -76, 22), (38, -48, 42))
    face += sculpt(teeth_field, (-34, -86, 1), (34, -48, 24))
    back = -50.0
    # the inserts sit in open sockets with a gap; close the sockets from behind
    # so face, eyes and teeth fuse into one piece
    fill = CrossSection([mouth_outline(), eye_outline(-1), eye_outline(1)]).offset(1.0)
    face += along_y(fill, back + 0.5, 6.0)
    outline = CrossSection([mask_outline()])
    shell = face ^ along_y(outline, back + 0.01, 60)
    shell = shell ^ box((-70, -100, -2), (70, back, 60))
    for side in (-1, 1):
        h = horn(side).scale((0.34, 0.34, 0.34)).translate((side * 36.0, back - 2.5, 47.0))
        shell += h.trim_by_plane((0, -1, 0), -back)
    k = MASK_SCALE
    shell = shell.scale((k, k, k)).translate((0, FRONT - MASK_STANDOFF - back * k,
                                               MASK_TOP - 55 * k))
    small = outline.scale((k, k)).offset(-1.5)
    tiny = outline.scale((k, k)).offset(-1.5 - MASK_STANDOFF)
    zoff = MASK_TOP - 55 * k
    block = Manifold.batch_hull([along_y(small, FRONT - MASK_STANDOFF - 0.5, 0.01),
                                 along_y(tiny, FRONT, 0.01)]).translate((0, 0, zoff))
    return shell + block


def mask_cutout():
    """Room for the mask in the rope."""
    k = MASK_SCALE
    sec = CrossSection([mask_outline()]).scale((k, k)).offset(0.3)
    return along_y(sec, FRONT + 1, 20).translate((0, 0, MASK_TOP - 55 * k))


def build():
    parts = {"plinth": plinth(), "podium": podium(),
             "posts": post(-1) + post(1), "rope": rope() - mask_cutout(), "mask": mask()}
    world = dict(parts)
    world.update(zip(("station_mock", "station_glow", "station_top"), station_mock(FLOOR, ST_Y)))

    lay_flat = lambda m: m.rotate((-90, 0, 0))     # flat back down
    printed = {
        "plinth": parts["plinth"],
        "podium": parts["podium"],
        "posts": post(-1).translate((BODY_W / 2 - POST / 2 - 10, 0, 0))
        + post(1).translate((-(BODY_W / 2 - POST / 2) + 10, 0, 0)),
        "rope": lay_flat(parts["rope"]),
        "mask": lay_flat(parts["mask"]),
        "fit_test": fit_test(),
    }
    return world, printed


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(here / "stl" / "shrine"))
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
