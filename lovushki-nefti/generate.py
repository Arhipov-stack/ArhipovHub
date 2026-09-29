#!/usr/bin/env python3
"""Сборные объёмные модели ловушек нефти и газа: модель 2 (литологическая)
и модель 4 (стратиграфическая, под поверхностью несогласия).

Каждая модель — блок 120 x 50 мм. Геологические границы — настоящие
поверхности z = f(x, y): пласт меняется не только вдоль блока, но и по его
глубине, так что передняя и задняя стенки выглядят по-разному.

Детали сцеплены шипами и пазами той же формы, что в моделях 1 и 3: шип 6 мм
по вершине, высота 2,5 мм, стенки 59°. Шип — гребень, который идёт по всей
глубине блока и повторяет поверхность пласта под ним, поэтому он всегда
прилегает к пласту. Паз в верхней детали — та же поверхность, отодвинутая
на зазор (по умолчанию 0,25 мм) по всем граням.

Все детали ставятся сверху. Печатаются лёжа на передней стенке (y = 0):
уклоны поверхностей по глубине пологие, поэтому поддержки не нужны.

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
from manifold3d import Manifold, Mesh

L, D = 120.0, 50.0            # длина (X) и глубина (Y) блока, мм
ZMAX = 100.0
PAD = 1.0                     # запас поверхностей за краем блока
TONGUE_TOP = 6.0
TONGUE_H = 2.5
TONGUE_ANGLE = 59.0
RUN = TONGUE_H / math.tan(math.radians(TONGUE_ANGLE))
MIN_WALL = 1.5                # минимум материала над пазом, мм

COLORS = {
    "seal_lower": (128, 118, 100),
    "seal_upper": (170, 160, 140),
    "seal_mid": (150, 140, 120),
    "clay": (100, 115, 90),
    "water": (70, 105, 150),
    "oil": (120, 72, 38),
    "gas": (200, 70, 55),
}


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bump(x0):
    """Профиль шипа по X: трапеция высотой 2,5 мм с центром в x0."""
    half = TONGUE_TOP / 2
    return lambda X, Y: TONGUE_H * np.clip((half + RUN - np.abs(X - x0)) / RUN, 0.0, 1.0)


def with_bumps(f, xs, sign=1.0):
    """Поверхность с шипами. sign = -1 — шип смотрит вниз (паз в нижней детали)."""
    bs = [bump(x) for x in xs]
    return lambda X, Y: f(X, Y) + sign * sum(b(X, Y) for b in bs)


def dilate(f, c):
    """Поверхность f, отодвинутая вверх на c по нормали (сдвиг шаром)."""
    if c <= 0:
        return f
    offs = [(dx, dy, math.sqrt(max(c * c - dx * dx - dy * dy, 0.0)))
            for dx in np.linspace(-c, c, 11) for dy in np.linspace(-c, c, 5)
            if dx * dx + dy * dy <= c * c + 1e-12]

    def g(X, Y):
        return np.max([f(X + dx, Y + dy) + dz for dx, dy, dz in offs], axis=0)
    return g


# ------------------------------------------------------------ тела-заготовки
def _grid(lo, hi, step, fine=(), fine_step=0.1, fine_r=5.6):
    g = [np.arange(lo, hi + step / 2, step)]
    for x0 in fine:
        g.append(np.arange(x0 - fine_r, x0 + fine_r + fine_step / 2, fine_step))
    g = np.unique(np.round(np.concatenate(g), 4))
    return g[(g >= lo) & (g <= hi)]


def heightfield(f, us, vs, w0, axes=(0, 1, 2)):
    """Замкнутое тело между плоскостью w = w0 и поверхностью w = f(u, v).
    axes: какие оси мира соответствуют (u, v, w)."""
    U, V = np.meshgrid(us, vs, indexing="ij")
    W = np.asarray(f(U, V), float)
    W = np.broadcast_to(W, U.shape)
    nu, nv = U.shape
    top = np.stack([U, V, W], -1).reshape(-1, 3)
    bot = np.stack([U, V, np.full_like(U, w0)], -1).reshape(-1, 3)
    verts = np.vstack([top, bot])
    idx = np.arange(nu * nv).reshape(nu, nv)
    off = nu * nv
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = [np.stack([a, b, c], -1), np.stack([a, c, d], -1)]
    tris += [np.stack([a, c, b], -1) + off, np.stack([a, d, c], -1) + off]
    faces = np.vstack([t.reshape(-1, 3) for t in tris])
    side = []
    for ring in (idx[:, 0], idx[-1, :], idx[::-1, -1], idx[0, ::-1]):
        p, q = ring[:-1], ring[1:]
        side.append(np.stack([p, q + off, q], -1))
        side.append(np.stack([p, p + off, q + off], -1))
    faces = np.vstack([faces] + side)
    if W.mean() < w0:                      # тело «под» плоскостью — вывернуть
        faces = faces[:, ::-1]
    world = np.zeros_like(verts)
    for i, ax in enumerate(axes):
        world[:, ax] = verts[:, i]
    perm_parity = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1}.get(tuple(axes), -1)
    if perm_parity < 0:
        faces = faces[:, ::-1]
    m = Manifold(Mesh(vert_properties=world.astype(np.float32),
                      tri_verts=faces.astype(np.uint32)))
    assert m.status().name == "NoError", m.status()
    return m


class Space:
    """Строитель тел для одной модели: сетка по X сгущается у шипов."""

    def __init__(self, tongue_xs, c):
        self.xs = _grid(-PAD, L + PAD, 0.5, fine=tongue_xs)
        self.ys = _grid(-PAD, D + PAD, 1.0)
        self.c = c

    def below(self, f):
        return heightfield(f, self.xs, self.ys, -5.0)

    def above(self, f):
        return heightfield(f, self.xs, self.ys, ZMAX)

    def above_gap(self, f):
        """Над поверхностью с зазором: так ставится следующая деталь."""
        return self.above(dilate(f, self.c))

    def left_of(self, xb, shrink=0.0):
        """x < xb(y, z) — граница, заданная как x от (y, z)."""
        ys = _grid(-PAD, D + PAD, 0.5)
        zs = _grid(-5.0, ZMAX, 0.5)
        return heightfield(lambda Y, Z: xb(Y, Z) - shrink, ys, zs, -5.0, axes=(1, 2, 0))

    def right_of(self, xb):
        ys = _grid(-PAD, D + PAD, 0.5)
        zs = _grid(-5.0, ZMAX, 0.5)
        return heightfield(xb, ys, zs, L + 5.0, axes=(1, 2, 0))


BLOCK = Manifold.cube([L, D, ZMAX])


def finish(parts):
    out = []
    for key, title, color, solid in parts:
        solid = solid ^ BLOCK
        bodies = [b for b in solid.decompose() if b.volume() > 5.0]
        assert len(bodies) == 1, f"{key}: {len(bodies)} кусков"
        out.append((key, title, color, bodies[0]))
    return out


# ---------------------------------------------------------------- модель 2
def model2(c):
    """Литологическая. Пласт изогнут флексурой: слева высоко, справа низко;
    ось флексуры идёт наискосок и слегка выгнута по глубине блока. Вверх по
    восстанию песчаник замещается глинами: граница зубчатая в плане и
    наклонная в разрезе (глины надвинуты на песчаник клиньями).
    Нефть упирается в глины, ниже вода. ВНК горизонтален."""
    thick = 12.5
    owc = 21.0

    def base(X, Y):
        xc = 22.0 + 3.0 * (Y / D - 0.5)
        return 5.0 + 27.0 * (1 - smoothstep((X - xc) / 72.0)) + 0.8 * np.sin(np.pi * Y / D)

    roof = lambda X, Y: base(X, Y) + thick
    top = lambda X, Y: roof(X, Y) + 15.0 + 1.5 * np.sin(X / L * np.pi) - 0.8 * np.sin(np.pi * Y / D)

    def xb(Y, Z):
        # зубцы в плане: три зубца на глубину блока, размах ±3,5 мм
        # (стенки не круче ~40° к оси Y — печатаются без поддержек);
        # в разрезе граница наклонена — чем выше, тем дальше вправо
        t = (Y / (D / 3)) % 1.0
        saw = 3.5 * (4 * np.abs(t - 0.5) - 1)
        return 41.0 + saw + 0.45 * (Z - 36.0)

    tx_base = [12.0, 106.0]
    tx_owc = [63.0]
    tx_roof = [14.0, 60.0]
    tx_roof_down = [108.0]
    sp = Space(tx_base + tx_owc + tx_roof + tx_roof_down, c)
    base_T = with_bumps(base, tx_base)
    owc_T = with_bumps(lambda X, Y: np.full_like(X, owc), tx_owc)
    # справа кровля почти у ВНК: шип вверх ушёл бы в нефть, поэтому он смотрит вниз, в воду
    roof_T = with_bumps(with_bumps(roof, tx_roof), tx_roof_down, sign=-1)

    band = sp.above_gap(base_T) ^ sp.below(roof_T)
    sand = sp.right_of(xb)
    parts = [
        ("seal_lower", "01 Непроницаемые породы (подошва)", COLORS["seal_lower"], sp.below(base_T)),
        ("water", "02 Песчаник: вода", COLORS["water"], band ^ sand ^ sp.below(owc_T)),
        ("oil", "03 Песчаник: нефть", COLORS["oil"], band ^ sand ^ sp.above_gap(owc_T)),
        ("clay", "04 Глины замещения (экран)", COLORS["clay"], band ^ sp.left_of(xb, shrink=c * 1.1)),
        ("seal_upper", "05 Непроницаемые породы (покрышка)", COLORS["seal_upper"],
         sp.above_gap(roof_T) ^ sp.below(top)),
    ]
    tongues = [  # (x0, деталь с шипом, деталь с пазом)
        (12.0, "seal_lower", "clay"), (106.0, "seal_lower", "water"),
        (63.0, "water", "oil"),
        (14.0, "clay", "seal_upper"), (60.0, "oil", "seal_upper"), (108.0, "seal_upper", "water"),
    ]
    return finish(parts), tongues


# ---------------------------------------------------------------- модель 4
def model4(c):
    """Стратиграфическая. Наклонная толща (пласт-коллектор между
    непроницаемыми слоями) срезана волнистой поверхностью размыва и
    перекрыта почти горизонтальными непроницаемыми породами. Простирание
    пластов косое к блоку, поэтому линия среза в плане идёт наискосок.
    Залежь — у среза: газ, ниже нефть, ниже вода; контакты горизонтальны."""
    thick = 12.0
    goc, owc = 30.0, 22.0

    base = lambda X, Y: 4.0 + 0.34 * (L - X) + 0.07 * (Y - D / 2)
    roof = lambda X, Y: base(X, Y) + thick
    unc = lambda X, Y: 37.0 - 0.05 * X + 1.2 * np.sin(np.pi * Y / D) + 0.6 * np.sin(X / 25.0)
    top = lambda X, Y: 54.0 - 0.02 * X + 1.2 * np.sin(X / L * 2 * np.pi) + 0.8 * np.cos(np.pi * Y / D)

    tx_unc = [10.0, 50.0, 100.0]
    tx_base = [64.0, 110.0]
    tx_roof = [114.0]
    tx_owc = [88.0]
    sp = Space(tx_unc + tx_base + tx_roof + tx_owc, c)
    unc_T = with_bumps(unc, tx_unc)
    base_T = with_bumps(base, tx_base)
    # шип слоя над пластом смотрит вниз: вверх он поднял бы кровлю выше ВНК
    roof_T = with_bumps(roof, tx_roof, sign=-1)
    owc_T = with_bumps(lambda X, Y: np.full_like(X, owc), tx_owc, sign=-1)
    goc_f = lambda X, Y: np.full_like(X, goc)
    lower_top = lambda X, Y: np.minimum(base_T(X, Y), unc_T(X, Y))
    band_top = lambda X, Y: np.minimum(roof_T(X, Y), unc_T(X, Y))

    band = sp.above_gap(lower_top) ^ sp.below(band_top)
    parts = [
        ("seal_lower", "01 Непроницаемые породы (подошва)", COLORS["seal_lower"], sp.below(lower_top)),
        ("water", "02 Пласт-коллектор: вода", COLORS["water"], band ^ sp.below(owc_T)),
        ("oil", "03 Пласт-коллектор: нефть", COLORS["oil"],
         band ^ sp.above_gap(owc_T) ^ sp.below(goc_f)),
        ("gas", "04 Пласт-коллектор: газ", COLORS["gas"], band ^ sp.above_gap(goc_f)),
        ("seal_mid", "05 Непроницаемый слой над пластом (срезан)", COLORS["seal_mid"],
         sp.above_gap(band_top) ^ sp.below(unc_T)),
        ("seal_upper", "06 Покрышка над несогласием", COLORS["seal_upper"],
         sp.above_gap(unc_T) ^ sp.below(top)),
    ]
    tongues = [
        (10.0, "seal_lower", "seal_upper"), (50.0, "gas", "seal_upper"), (100.0, "seal_mid", "seal_upper"),
        (64.0, "seal_lower", "oil"), (110.0, "seal_lower", "water"),
        (114.0, "seal_mid", "water"), (88.0, "oil", "water"),
    ]
    return finish(parts), tongues


# ---------------------------------------------------------------- проверки
def column(solid, x, y):
    """Толщина тела по вертикали в точке (x, y), мм."""
    col = Manifold.cube([0.05, 0.05, ZMAX + 10]).translate([x - 0.025, y - 0.025, -5])
    return (solid ^ col).volume() / 0.0025


def check(parts, tongues, c):
    by = {k: s for k, _, _, s in parts}
    ys = [1.0, 12.0, 25.0, 38.0, 49.0]
    problems = []
    for x0, owner, target in tongues:
        for y in ys:
            # над вершиной шипа должна остаться стенка
            for dx in (-TONGUE_TOP / 2 + 0.3, 0.0, TONGUE_TOP / 2 - 0.3):
                t = column(by[target], x0 + dx, y)
                if t < MIN_WALL:
                    problems.append(f"x={x0 + dx:.1f} y={y:.0f}: над пазом в {target} {t:.2f} мм")
            # под шипом и сбоку от паза — сплошной материал обеих деталей
            half = TONGUE_TOP / 2 + RUN + c + 0.3
            for dx in (-half, half):
                for k in (owner, target):
                    if column(by[k], x0 + dx, y) < 1.0:
                        problems.append(f"x={x0 + dx:.1f} y={y:.0f}: шип {owner}->{target} выходит за {k}")
    keys = list(by)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = (by[keys[i]] ^ by[keys[j]]).volume()
            if ov > 1e-3:
                problems.append(f"{keys[i]} и {keys[j]} пересекаются: {ov:.3f} мм³")
    return problems


def to_trimesh(solid):
    # плоские участки сетки сливаются; форма меняется не больше чем на 0,02 мм
    m = solid.simplify(0.02).to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def print_pose(tm):
    """Деталь кладётся на переднюю стенку (y = 0): y становится высотой."""
    r = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])   # (x,y,z)->(x,-z,y)
    tm = tm.copy()
    tm.apply_transform(r)
    tm.apply_translation(-tm.bounds[0])
    return tm


def overhang(tm, limit=45.0):
    """Доля площади, нависающей круче limit° от вертикали (без опоры на стол)."""
    n = tm.face_normals
    z = tm.triangles_center[:, 2]
    bad = (n[:, 2] < -math.cos(math.radians(limit))) & (z > 0.3)
    return tm.area_faces[bad].sum() / tm.area


def export(key, stem, parts, outdir, bed=180.0, gap=4.0):
    pdir = os.path.join(outdir, key)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "preview"), exist_ok=True)
    plate, asm, report = [], [], []
    y = 0.0
    for k, title, color, solid in parts:
        num = title.split()[0]
        tm = to_trimesh(solid)
        assert tm.is_watertight, k
        pm = print_pose(tm)
        fname = f"{num}_{k}.stl"
        pm.export(os.path.join(pdir, fname))
        placed = pm.copy()
        placed.apply_translation([0, y, 0])
        plate.append(placed)
        y += pm.extents[1] + gap
        am = tm.copy()
        am.visual.face_colors = list(color) + [255]
        asm.append(am)
        report.append((title, fname, pm.volume, pm.extents, overhang(pm)))
    plate_mesh = trimesh.util.concatenate(plate)
    ext = plate_mesh.extents
    assert ext[0] <= bed and ext[1] <= bed, f"раскладка не влезает на стол: {ext}"
    plate_mesh.apply_translation([(bed - ext[0]) / 2, (bed - ext[1]) / 2, 0])
    plate_mesh.export(os.path.join(outdir, stem + ".stl"))
    trimesh.util.concatenate(asm).export(os.path.join(outdir, "preview", stem + "_sborka.stl"))
    return report, ext


# ---------------------------------------------------------------- превью
def render(parts, path, title, explode=0.0, size=(1100, 720), az_deg=-35, el_deg=28):
    """Изометрическое превью с z-буфером (без OpenGL)."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = size
    az, el = math.radians(az_deg), math.radians(el_deg)
    R1 = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
    R = R2 @ R1
    light = np.array([-0.35, -0.8, 0.9])
    light /= np.linalg.norm(light)
    meshes = []
    for i, (k, t, color, solid) in enumerate(parts):
        m = to_trimesh(solid)
        m.apply_translation([0, 0, i * explode])
        meshes.append((m, np.array(color, float)))
    allv = np.vstack([m.vertices for m, _ in meshes]) @ R.T
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
            cc = 1 - a - b
            inside = (a >= -1e-6) & (b >= -1e-6) & (cc >= -1e-6)
            if not inside.any():
                continue
            zz = a * z[0] + b * z[1] + cc * z[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zz < sub)
            sub[upd] = zz[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = color * s_
    edge = np.zeros((H, W), bool)
    zb = np.where(np.isfinite(zbuf), zbuf, 1e6)
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
    ap.add_argument("--no-check", action="store_true", help="пропустить проверку шипов")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = [
        ("model2", "2-litologicheskaya-zameschenie", "Модель 2. Литологическая", model2),
        ("model4", "4-stratigraficheskaya-nesoglasie", "Модель 4. Стратиграфическая (несогласие)", model4),
    ]
    failed = False
    for key, stem, title, fn in models:
        parts, tongues = fn(a.clearance)
        if not a.no_check:
            probs = check(parts, tongues, a.clearance)
            for p in probs:
                print("  ПРОБЛЕМА:", p)
            failed |= bool(probs)
        report, ext = export(key, stem, parts, a.out)
        print(f"\n{title}: раскладка {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} мм")
        for name, fname, vol, e, oh in report:
            print(f"  {name:44s} {fname:22s} {vol / 1000:6.1f} см³  "
                  f"{e[0]:.0f}x{e[1]:.0f}x{e[2]:.0f}  нависания {oh * 100:.1f}%")
        if a.render:
            pdir = os.path.join(a.out, "preview")
            render(parts, os.path.join(pdir, stem + ".png"), title)
            render(parts, os.path.join(pdir, stem + "_szadi.png"), title + " — вид сзади", az_deg=145)
            render(parts, os.path.join(pdir, stem + "_razbor.png"), title + " — разнесённые детали", explode=12)
    if failed:
        raise SystemExit("есть проблемы с шипами — см. выше")


if __name__ == "__main__":
    main()
