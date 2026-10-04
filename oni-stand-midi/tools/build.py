"""Build every part, export STLs in print orientation, the assembly and previews.

    python3 tools/build.py            # final quality (0.4 mm voxels, ~6 min)
    python3 tools/build.py --draft    # quick check (0.9 mm)
"""
import argparse
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
import base  # noqa: E402
import head  # noqa: E402
import horn  # noqa: E402
import params as C  # noqa: E402
import render  # noqa: E402
import sdf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_bed(m):
    m = m.copy()
    m.apply_translation([-m.bounds[:, 0].mean(), -m.bounds[:, 1].mean(), -m.bounds[0, 2]])
    return m


def mirror_x(m):
    m = m.copy()
    m.apply_transform(np.diag([-1.0, 1, 1, 1]))
    m.fix_normals()
    return m


def report(name, m):
    ext = m.extents
    vol = m.volume / 1000
    print(f"  {name:12s} {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm  "
          f"{vol:7.1f} cm3  {len(m.faces):8d} tris  watertight={m.is_watertight}")
    return dict(name=name, size=ext.round(1).tolist(), volume_cm3=round(vol, 1),
                faces=len(m.faces), watertight=bool(m.is_watertight))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    step = 0.9 if a.draft else 0.4
    out = os.path.join(ROOT, "stl" if not a.draft else "stl_draft")
    prev = os.path.join(ROOT, "previews")
    os.makedirs(out, exist_ok=True)
    os.makedirs(prev, exist_ok=True)

    head.setup()
    base.setup()

    t = time.time()
    H = sdf.mesh(head.head_sdf, head.HEAD_LO, head.HEAD_HI, step, faces=700_000)
    print(f"head   {time.time() - t:5.0f} s")
    t = time.time()
    B = sdf.mesh(base.base_sdf, base.BASE_LO, base.BASE_HI, step, faces=450_000)
    print(f"base   {time.time() - t:5.0f} s")
    t = time.time()
    lo, hi = horn.horn_bounds()
    HR = sdf.mesh(horn.horn_sdf, lo, hi, step * 0.75, faces=150_000)
    print(f"horn   {time.time() - t:5.0f} s")
    PIN = sdf.mesh(horn.pin_sdf, horn.PIN_LO, horn.PIN_HI, 0.25, faces=20_000)
    HL = mirror_x(HR)

    # ---- assembly (head frame -> world)
    up = np.eye(4)
    up[2, 3] = base.HEAD_Z
    asm = [B.copy(), H.copy(), HR.copy(), HL.copy()]
    for m in asm[1:]:
        m.apply_transform(up)
    A = trimesh.util.concatenate(asm)

    # ---- fit check: nothing of the head may reach into the speaker pocket
    v = H.vertices
    r = C.POCKET_R
    qx = np.maximum(np.abs(v[:, 0]) - (C.PX - r), 0)
    qy = np.maximum(np.abs(v[:, 1]) - (C.PY - r), 0)
    above_floor = (v[:, 2] > 0.25) & (v[:, 2] < C.SPEAKER_H)   # floor vertices sit at z = 0
    pen = np.max(r - np.hypot(qx, qy)[above_floor], initial=0.0)
    print(f"pocket {2 * C.PX:.1f} x {2 * C.PY:.1f} mm, deepest wall intrusion: {max(pen, 0):.2f} mm")
    assert pen < 0.3

    # ---- print orientation
    _, b0, u = horn.horn_curve()
    rot = trimesh.geometry.align_vectors(u, [0, 0, -1])      # root face flat on the bed
    hr_p = HR.copy()
    hr_p.apply_transform(rot)
    parts = {
        "head": to_bed(H),
        "base": to_bed(B),
        "horn_right": to_bed(hr_p),
        "horn_left": to_bed(mirror_x(hr_p)),
        "horn_pin": to_bed(PIN),
        "assembly_preview": sdf.lighten(A, 500_000),
    }
    print("parts:")
    for k, m in parts.items():
        m.export(os.path.join(out, k + ".stl"))
        report(k, m)

    # ---- previews with a stand-in speaker
    spk = sdf.mesh(lambda P: sdf.round_box(P, (0, 0, C.SPEAKER_H / 2 + base.HEAD_Z),
                                           (C.SPEAKER_W / 2, C.SPEAKER_D / 2, C.SPEAKER_H / 2), 12.0),
                   (-55, -55, base.HEAD_Z - 5), (55, 55, base.HEAD_Z + 120), 1.0)
    clay = (0.78, 0.77, 0.75)
    dark = (0.33, 0.33, 0.36)
    px = max(step, 0.4) * 1.2
    views = [(0, 6, "front"), (35, 18, "three_quarter"), (155, 16, "back"), (-90, 4, "side")]
    ims = []
    for yaw, pitch, name in views:
        im = render.render([(A, clay), (spk, dark)], yaw=yaw, pitch=pitch, px=px)
        im.save(os.path.join(prev, f"assembly_{name}.png"))
        im.thumbnail((900, 900))
        ims.append(im)
    render.sheet(ims, os.path.join(prev, "assembly_sheet.png"), cols=2)
    face = render.render([(H, clay)], yaw=0, pitch=4, px=px * 0.8)
    face.save(os.path.join(prev, "head_front.png"))
    render.render([(B, clay)], yaw=25, pitch=22, px=px).save(os.path.join(prev, "base.png"))


if __name__ == "__main__":
    main()
