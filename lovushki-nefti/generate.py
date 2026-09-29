#!/usr/bin/env python3
"""Сборные модели ловушек нефти и газа: модель 2 (литологическая) и модель 4
(стратиграфическая, под поверхностью несогласия).

Каждая модель — блок 120 x 50 мм. Детали — профили в плоскости XZ,
вытянутые на всю глубину блока (50 мм по Y). Соседние детали сцеплены
шипами и пазами той же формы, что в моделях 1 и 3: шип 6 мм по вершине,
высота 2,5 мм, стенки 59°, зазор по всем граням (по умолчанию 0,25 мм).

Выход (папка stl/):
  2-litologicheskaya-zameschenie.stl   — раскладка для печати
  4-stratigraficheskaya-nesoglasie.stl — раскладка для печати
  model2/, model4/                     — детали по отдельности
  preview/*_sborka.stl                 — собранная модель для просмотра

    python3 generate.py [--clearance 0.25] [--render]
"""
import argparse
import math
import os

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

L, D = 120.0, 50.0            # длина блока (X) и глубина (Y), мм
STEP = 0.5                    # шаг дискретизации кривых, мм
TONGUE_TOP = 6.0
TONGUE_H = 2.5
TONGUE_ANGLE = 59.0
MIN_AREA = 0.5                # обрезки меньше этого (мм^2) выбрасываются

COLORS = {
    "seal_lower": (128, 118, 100),
    "seal_upper": (170, 160, 140),
    "seal_mid": (150, 140, 120),
    "clay": (100, 115, 90),
    "water": (70, 105, 150),
    "oil": (120, 72, 38),
    "gas": (200, 70, 55),
}

XS = np.arange(0.0, L + STEP / 2, STEP)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def region_between(lo, hi, x0=0.0, x1=L):
    """Область lo(x) < z < hi(x) на отрезке [x0, x1]; lo, hi — функции."""
    xs = XS[(XS >= x0) & (XS <= x1)]
    if xs[0] > x0:
        xs = np.r_[x0, xs]
    if xs[-1] < x1:
        xs = np.r_[xs, x1]
    top = [(x, hi(x)) for x in xs]
    bot = [(x, lo(x)) for x in xs[::-1]]
    return Polygon(top + bot).buffer(0)


def tongue(x0, z_base, up=True, slope=0.0):
    """Трапеция шипа с вершиной на 2,5 мм над границей в точке (x0, z_base),
    повёрнутая перпендикулярно границе с наклоном slope. На наклонах до
    ~30° обе стенки остаются круче горизонтали, так что деталь по-прежнему
    ставится сверху. Стенки продолжены вглубь детали-хозяина: лишнее срежется."""
    run = 1.0 / math.tan(math.radians(TONGUE_ANGLE))
    h = TONGUE_H
    deep = 8.0
    half = TONGUE_TOP / 2
    s = 1 if up else -1
    zt = z_base + s * h
    zb = zt - s * (h + deep)
    w = half + (h + deep) * run
    p = Polygon([(x0 - half, zt), (x0 + half, zt), (x0 + w, zb), (x0 - w, zb)])
    return affinity.rotate(p, math.degrees(math.atan(slope)), origin=(x0, z_base))


def largest(geom):
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if g.area >= MIN_AREA]
        return MultiPolygon(parts) if len(parts) > 1 else parts[0]
    return geom


def build(parts, tongues, clearance):
    """parts: [(key, name, color, region)] в порядке сборки.
    tongues: [(owner_key, target_key, x0, boundary_fn, up)].
    Шип прибавляется к owner и вырезается из target; затем каждая деталь
    отодвигается от всех ранее поставленных на величину зазора."""
    reg = {k: r for k, _, _, r in parts}
    for owner, target, x0, fz, up in tongues:
        slope = (fz(x0 + 1) - fz(x0 - 1)) / 2
        t = tongue(x0, fz(x0), up, slope)
        stick = t.intersection(reg[target])
        # выступ шипа должен целиком войти в деталь-мишень,
        # а над пазом должно остаться не меньше 1,5 мм материала
        full = TONGUE_H * (TONGUE_TOP + TONGUE_H / math.tan(math.radians(TONGUE_ANGLE)))
        assert stick.area > 0.8 * full, (owner, target, x0, stick.area, full)
        cap = stick.buffer(1.5 + clearance, join_style=2).difference(stick.buffer(0.01))
        cap = cap.difference(reg[owner].buffer(0.3)).intersection(box(0, 0, L, 100))
        lost = cap.difference(reg[target]).area
        assert lost < 0.5, (owner, target, x0, "тонкая стенка над пазом", lost)
        rest = reg[target].difference(stick)
        reg[owner] = reg[owner].union(stick).buffer(0)
        reg[target] = rest.buffer(0)
    out, placed = [], []
    for k, name, color, _ in parts:
        g = reg[k]
        if placed:
            g = g.difference(unary_union(placed).buffer(clearance, join_style=2, mitre_limit=3))
        g = largest(g.buffer(0))
        assert not isinstance(g, MultiPolygon), f"{k}: деталь распалась на куски"
        placed.append(reg[k])
        out.append((k, name, color, g))
    return out


# ---------------------------------------------------------------- модель 2
def model2():
    """Литологическая: пласт изогнут флексурой (высоко слева, низко справа),
    вверх по восстанию песчаник замещается глинами с «зубчатой» границей.
    Нефть упирается в глины, ниже — вода. ВНК горизонтален."""
    base = lambda x: 12.0 + 20.0 * (1 - smoothstep((x - 22.0) / 72.0))
    thick = 11.0
    roof = lambda x: base(x) + thick
    top = lambda x: roof(x) + 15.0 + 1.5 * math.sin(x / 120 * math.pi)
    owc = 24.0

    band = region_between(base, roof)
    lower = region_between(lambda x: 0.0, base)
    upper = region_between(roof, top)

    # Зубчатая граница замещения: три клина глин входят в песчаник
    # и три клина песчаника — в глины (пальцевидное замещение).
    x_mid = 40.0
    nz = 7
    zs = np.linspace(-2, thick + 2, nz)
    pts = []
    for i, dz in enumerate(zs):
        xo = 6.5 if i % 2 else -6.5
        # граница идёт перпендикулярно пласту, с учётом наклона
        slope = (roof(x_mid + 1) - roof(x_mid - 1)) / 2
        z = base(x_mid) + dz
        pts.append((x_mid + xo - slope * dz, z))
    poly_left = Polygon([(-1, -1)] + [(-1, 70)] + [(pts[-1][0], 70)] + pts[::-1] + [(pts[0][0], -1)]).buffer(0)
    clay = band.intersection(poly_left)
    sand = band.difference(poly_left)
    oil = sand.intersection(box(-1, owc, L + 1, 80))
    water = sand.intersection(box(-1, -1, L + 1, owc))

    parts = [
        ("seal_lower", "01 Непроницаемые породы (подошва)", COLORS["seal_lower"], lower),
        ("water", "02 Песчаник: вода", COLORS["water"], water),
        ("oil", "03 Песчаник: нефть", COLORS["oil"], oil),
        ("clay", "04 Глины замещения (экран)", COLORS["clay"], clay),
        ("seal_upper", "05 Непроницаемые породы (покрышка)", COLORS["seal_upper"], upper),
    ]
    tongues = [
        ("seal_lower", "clay", 12.0, base, True),
        ("seal_lower", "water", 106.0, base, True),
        ("water", "oil", 66.0, lambda x: owc, True),
        ("clay", "seal_upper", 14.0, roof, True),
        ("oil", "seal_upper", 54.0, roof, True),
        ("water", "seal_upper", 108.0, roof, True),
    ]
    return parts, tongues


# ---------------------------------------------------------------- модель 4
def model4():
    """Стратиграфическая: наклонная толща (пласт-коллектор между
    непроницаемыми слоями) срезана поверхностью размыва и перекрыта
    почти горизонтальными непроницаемыми породами. Залежь — у среза."""
    dip = 0.30
    base = lambda x: 6.0 + dip * (L - x)
    thick = 12.0
    roof = lambda x: base(x) + thick
    unc = lambda x: 38.0 - 0.06 * x                  # поверхность несогласия
    top = lambda x: 54.0 - 0.02 * x + 1.2 * math.sin(x / 120 * 2 * math.pi)
    goc, owc = 30.0, 22.0

    lower = region_between(lambda x: 0.0, lambda x: min(base(x), unc(x)))
    band = region_between(base, lambda x: max(base(x), min(roof(x), unc(x))))
    mid = region_between(roof, lambda x: max(roof(x), unc(x)))
    cover = region_between(lambda x: min(roof(x), unc(x)), top)
    cover = cover.difference(mid).difference(band)

    gas = band.intersection(box(-1, goc, L + 1, 80))
    oil = band.intersection(box(-1, owc, L + 1, goc))
    water = band.intersection(box(-1, -1, L + 1, owc))

    parts = [
        ("seal_lower", "01 Непроницаемые породы (подошва)", COLORS["seal_lower"], lower),
        ("water", "02 Пласт-коллектор: вода", COLORS["water"], water),
        ("oil", "03 Пласт-коллектор: нефть", COLORS["oil"], oil),
        ("gas", "04 Пласт-коллектор: газ", COLORS["gas"], gas),
        ("seal_mid", "05 Непроницаемый слой над пластом (срезан)", COLORS["seal_mid"], mid),
        ("seal_upper", "06 Покрышка над несогласием", COLORS["seal_upper"], cover),
    ]
    tongues = [
        ("seal_lower", "seal_upper", 8.0, unc, True),
        ("seal_lower", "oil", 58.0, base, True),
        ("seal_lower", "water", 108.0, base, True),
        ("water", "oil", 86.0, lambda x: owc, True),
        ("gas", "seal_upper", 46.0, unc, True),
        ("water", "seal_mid", 113.0, roof, True),
        ("seal_mid", "seal_upper", 100.0, unc, True),
    ]
    return parts, tongues


# ---------------------------------------------------------------- экспорт
def extrude_assembled(poly):
    """Деталь в собранном положении: профиль XZ, глубина по Y."""
    m = trimesh.creation.extrude_polygon(poly, D)       # профиль в XY, высота по Z
    # поворот на 90° вокруг X: (x, z_geo, depth) -> (x, D - depth, z_geo)
    m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, D], [0, 1, 0, 0], [0, 0, 0, 1]], float))
    return m


def extrude_print(poly):
    """Печатное положение: профиль лежит на столе."""
    return trimesh.creation.extrude_polygon(poly, D)


def export(name, file_stem, parts, outdir, bed=180.0, gap=4.0):
    pdir = os.path.join(outdir, name)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "preview"), exist_ok=True)
    plate, asm, report = [], [], []
    y = 0.0
    for k, title, color, g in parts:
        num = title.split()[0]
        pm = extrude_print(g)
        assert pm.is_watertight, k
        fname = f"{num}_{k}.stl"
        pm.export(os.path.join(pdir, fname))
        # раскладка: детали столбиком по Y стола
        minx, miny, maxx, maxy = g.bounds
        placed = pm.copy()
        placed.apply_translation([-minx, y - miny, 0])
        plate.append(placed)
        y += (maxy - miny) + gap
        am = extrude_assembled(g)
        am.visual.face_colors = list(color) + [255]
        asm.append(am)
        report.append((num, title, fname, pm.volume))
    plate_mesh = trimesh.util.concatenate(plate)
    ext = plate_mesh.extents
    offset = [(bed - ext[0]) / 2, (bed - ext[1]) / 2, 0]
    plate_mesh.apply_translation(offset - plate_mesh.bounds[0] * [1, 1, 0])
    plate_mesh.export(os.path.join(outdir, file_stem + ".stl"))
    asm_scene = trimesh.util.concatenate(asm)
    asm_scene.export(os.path.join(outdir, "preview", file_stem + "_sborka.stl"))
    return report, ext, asm


def render(parts, path, title, explode=0.0, size=(1100, 720)):
    """Изометрическое превью с z-буфером (без OpenGL)."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = size
    az, el = math.radians(-35), math.radians(28)
    # камера: смотрим с угла (-X, -Y, +Z)
    R1 = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
    R = R2 @ R1
    light = np.array([-0.35, -0.8, 0.9])
    light /= np.linalg.norm(light)
    meshes = []
    for i, (k, t, color, g) in enumerate(parts):
        m = extrude_assembled(g)
        m.apply_translation([0, 0, i * explode])
        meshes.append((m, np.array(color, float)))
    allv = np.vstack([m.vertices for m, _ in meshes]) @ R.T
    # экранные координаты: x -> вправо, z -> вверх, y -> глубина
    lo, hi = allv.min(0), allv.max(0)
    sc = 0.88 * min(W / (hi[0] - lo[0]), (H - 60) / (hi[2] - lo[2]))
    ox = (W - sc * (hi[0] - lo[0])) / 2
    oy = (H - 60 - sc * (hi[2] - lo[2])) / 2 + 50
    zbuf = np.full((H, W), np.inf)
    img = np.full((H, W, 3), 255.0)
    for m, color in meshes:
        v = m.vertices @ R.T
        sx = ox + (v[:, 0] - lo[0]) * sc
        sy = oy + (hi[2] - v[:, 2]) * sc
        dz = v[:, 1]
        sh = 0.45 + 0.55 * np.clip(m.face_normals @ light, 0, 1)
        for f, s_ in zip(m.faces, sh):
            x, y, z = sx[f], sy[f], dz[f]
            x0, x1 = int(max(x.min(), 0)), int(min(x.max() + 1, W - 1))
            y0, y1 = int(max(y.min(), 0)), int(min(y.max() + 1, H - 1))
            if x1 < x0 or y1 < y0:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
            b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
            c = 1 - a - b
            inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
            if not inside.any():
                continue
            zz = a * z[0] + b * z[1] + c * z[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zz < sub)
            sub[upd] = zz[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = color * s_
    # тонкие контуры по разрывам глубины — чтобы читались границы деталей
    edge = np.zeros((H, W), bool)
    fin = np.isfinite(zbuf)
    zb = np.where(fin, zbuf, 1e6)
    edge[1:, :] |= np.abs(zb[1:, :] - zb[:-1, :]) > 1.5
    edge[:, 1:] |= np.abs(zb[:, 1:] - zb[:, :-1]) > 1.5
    img[edge] *= 0.55
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((20, 15), title, fill=(30, 30, 30), font=font)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearance", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl"))
    ap.add_argument("--render", action="store_true", help="сохранить PNG-превью")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = [
        ("model2", "2-litologicheskaya-zameschenie", "Модель 2. Литологическая", model2),
        ("model4", "4-stratigraficheskaya-nesoglasie", "Модель 4. Стратиграфическая (несогласие)", model4),
    ]
    for key, stem, title, fn in models:
        p, t = fn()
        parts = build(p, t, a.clearance)
        report, ext, _ = export(key, stem, parts, a.out)
        print(f"\n{title}: раскладка {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} мм")
        for num, name, fname, vol in report:
            print(f"  {name:48s} {fname:22s} {vol / 1000:6.1f} см³")
        if a.render:
            pdir = os.path.join(a.out, "preview")
            render(parts, os.path.join(pdir, stem + ".png"), title)
            render(parts, os.path.join(pdir, stem + "_razbor.png"), title + " — разнесённые детали", explode=12)


if __name__ == "__main__":
    main()
