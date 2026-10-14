"""Наружная ручка двери Toyota Corolla Ceres (AE100/AE101).

Та же ручка стоит на Corolla E100 1993–1997 и RAV4 1996–2000
(оригинальный номер передней левой с личинкой — 69220-12160).

Форма в плане снята с фото (tools/outline_photo.json, см. trace_photo.py):
вытянутое «яйцо», узкое спереди и широкое у личинки. Механизм тоже по
фото: лопатка закрывает верхнюю половину проёма, под ней карман под
пальцы; ось — стальной штифт сзади вверху, на двух кронштейнах, с
пружиной кручения; рядом с личинкой на лопатке рычаг под тягу замка.

Детали:
  * корпус (bezel)  — накладка с проёмом, карманом, кронштейнами оси,
                      бобышками крепления и гнездом личинки;
  * лопатка (flap)  — подвижная часть: пластина, загиб под пальцы,
                      проушины на ось и рычаг на тягу;
  * штифт Ø4 мм и пружина — покупные или с донора.

Система координат (вид на дверь снаружи):
  X — вдоль машины, +X в сторону личинки и замка (задний край двери,
      при виде снаружи он слева);
  Y — по нормали к двери, +Y наружу, Y=0 — наружная поверхность обшивки;
  Z — вверх.

В таком виде это ручка ПРАВОЙ стороны (у праворульной Ceres —
водительская). Левая, как на фото, — зеркало по X.

Масштаб фото задаётся одной длиной накладки (P.length). Глубины по Y
по фото анфас не определить — они оценочные.
"""

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial import ConvexHull

PHOTO = json.loads(Path(__file__).with_name("outline_photo.json").read_text())


@dataclass(frozen=True)
class P:
    length: float = 203.0        # длина накладки — калибрует весь контур с фото

    # Накладка
    proud: float = 9.0           # выступ лица накладки над обшивкой
    front_fillet: float = 3.5    # скругление наружной кромки
    open_fillet: float = 1.0     # скругление кромки проёма
    floor_y: float = 2.0         # дно мелкой части проёма (узкий конец), над обшивкой
    deep_from_x: float = -60.0   # глубокий карман от этого X до конца проёма
    cup_clear: float = 1.0       # карман шире проёма на столько (ободок нависает)
    cup_wall: float = 2.2
    cup_depth: float = 24.0      # глубина кармана за обшивкой

    # Лопатка
    gap: float = 0.8             # зазор лопатки в проёме и по всем движущимся поверхностям
    paddle_recess: float = 2.5   # лицо лопатки ниже лица накладки
    paddle_t: float = 3.0
    lip_depth: float = 9.0       # загиб низа лопатки внутрь, от лица лопатки
    lip_t: float = 2.5
    lug_t: float = 4.0           # проушины на ось
    lug_r: float = 4.5
    spring_room: float = 12.0    # место под пружину между левым кронштейном и проушиной
    lever_len: float = 28.0      # рычаг на тягу: от оси назад, выходит за карман
    rod_hole_d: float = 5.0
    swing_deg: float = 22.0      # ход лопатки
    stop_x: tuple = (-40.0, 45.0)   # упоры закрытого положения за загибом лопатки
    stop_w: float = 4.0
    stop_clear: float = 0.2

    # Ось
    pivot_y: float = -1.5        # ось сразу за обшивкой
    pivot_z: float = 23.0
    pin_d: float = 4.0
    pin_clear: float = 0.3
    bracket_r: float = 4.0

    # Крепление: бобышки под термовставки M4 (болты изнутри двери)
    boss_d: float = 9.0
    boss_len: float = 6.0
    insert_d: float = 5.6
    insert_depth: float = 7.0

    # Личинка
    key: bool = False
    key_tube_wall: float = 3.0
    key_tube_len: float = 16.0

    # --- производные ---
    @property
    def s(self):
        x0, x1, _, _ = PHOTO["outer_bbox"]
        return self.length / (x1 - x0)

    def xz(self, u, v):
        x0, x1, y0, y1 = PHOTO["outer_bbox"]
        return ((u - (x0 + x1) / 2) * self.s, ((y0 + y1) / 2 - v) * self.s)

    @property
    def face_y(self):
        return self.proud - self.paddle_recess

    @property
    def pivot(self):
        return (self.pivot_y, self.pivot_z)

    @property
    def key_xz(self):
        return self.xz(*PHOTO["key_center"])

    @property
    def key_d(self):
        return PHOTO["key_diam"] * self.s

    @property
    def brackets_x(self):
        return [tuple(sorted(self.xz(u, 0)[0] for u in b)) for b in PHOTO["brackets_u"]]

    @property
    def lugs_x(self):
        """Проушины: левая — через место под пружину, правая — вплотную к кронштейну."""
        (_, l1), (r0, _) = self.brackets_x
        left0 = l1 + self.spring_room
        right1 = r0 - 0.5
        return ((left0, left0 + self.lug_t), (right1 - self.lug_t, right1))

    @property
    def bosses(self):
        kx, kz = self.key_xz
        pts = [(-80.0, -1.0), (kx - 7.0, kz + 24.0)]
        if self.key:
            pts.append((kx + 9.0, kz + 20.0))   # винт фиксатора личинки
        return pts

    def split_z(self, x):
        (u0, v0), (u1, v1) = PHOTO["split_line"]
        (x0, z0), (x1, z1) = self.xz(u0, v0), self.xz(u1, v1)
        return z0 + (z1 - z0) * (x - x0) / (x1 - x0)


# ---------- 2D-заготовки в плоскости XZ ----------

def _xz_wp(y):
    """Рабочая плоскость XZ на уровне y; extrude(d) идёт в -Y."""
    return cq.Workplane(cq.Plane(origin=(0, y, 0), xDir=(1, 0, 0), normal=(0, -1, 0)))


def _contour(p: P, key):
    return [p.xz(u, v) for u, v in PHOTO[key]]


def _prism(p: P, key, y0, y1, offset=0.0):
    """Контур с фото (inner/outer), раздутый на offset, выдавленный от y1 вниз до y0."""
    wp = _xz_wp(y1).spline(_contour(p, key), periodic=True).close()
    if offset:
        wp = wp.offset2D(offset, "arc")
    return wp.extrude(y1 - y0)


def _box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0)
            .translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)))


def _above_split(p: P, y0, y1, dz=0.0, band=None):
    """Полуплоскость над нижней кромкой лопатки (или полоса высотой band)."""
    xa, xb = -200.0, 200.0
    za, zb = p.split_z(xa) + dz, p.split_z(xb) + dz
    top_a, top_b = (za + band, zb + band) if band else (200.0, 200.0)
    return (_xz_wp(y1).polyline([(xa, za), (xb, zb), (xb, top_b), (xa, top_a)]).close()
            .extrude(y1 - y0))


def _yz_hull(points, x0, x1):
    """Выпуклая оболочка точек (y, z), выдавленная по X от x0 до x1."""
    pts = np.array(points)
    hull = pts[ConvexHull(pts).vertices]
    return (cq.Workplane("YZ", origin=(x0, 0, 0))
            .polyline([tuple(q) for q in hull]).close().extrude(x1 - x0))


def _circle_pts(c, r, n=24):
    return [(c[0] + r * np.cos(a), c[1] + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


# ---------- лопатка ----------

def _paddle(p: P, grow=0.0) -> cq.Workplane:
    """Лопатка в закрытом положении. grow>0 — раздутая копия для выреза зазоров."""
    g = grow
    fy = p.face_y
    # Пластина: проём минус зазор, выше нижней кромки
    plate = (_prism(p, "inner", fy - p.paddle_t - g, fy + g, offset=-p.gap + g)
             .intersect(_above_split(p, fy - p.paddle_t - g - 1, fy + g + 1, dz=-g)))
    # Загиб низа внутрь — только в глубокой части, у мелкого конца ему некуда
    lip = (_prism(p, "inner", fy - p.lip_depth - g, fy - p.paddle_t + 0.01, offset=-p.gap + g)
           .intersect(_above_split(p, fy - p.lip_depth - g - 1, fy + 1, dz=-g, band=p.lip_t + 2 * g))
           .intersect(_box(p.deep_from_x + p.cup_clear + p.gap - g, 300, -100, 100, -100, 100)))
    body = plate.union(lip)

    # Проушины на ось: от оси до пластины
    py, pz = p.pivot
    for i, (x0, x1) in enumerate(p.lugs_x):
        pts = _circle_pts((py, pz), p.lug_r + g) + [
            (fy - p.paddle_t + 0.5, pz + 3.0 + g), (fy - p.paddle_t + 0.5, pz - 12 - g),
        ]
        if i == 1:   # у личинки — рычаг на тягу
            pts += _circle_pts((py - p.lever_len, pz - 2), p.lug_r + g)
        body = body.union(_yz_hull(pts, x0 - g, x1 + g))

    if grow == 0:
        (x0, _), (_, x1) = p.lugs_x
        pin = (cq.Workplane("YZ", origin=(x0 - 1, 0, 0)).center(py, pz)
               .circle(p.pin_d / 2 + p.pin_clear).extrude(x1 - x0 + 2))
        rod = (cq.Workplane("YZ", origin=(p.lugs_x[1][0] - 1, 0, 0)).center(py - p.lever_len, pz - 2)
               .circle(p.rod_hole_d / 2).extrude(p.lug_t + 2))
        body = body.cut(pin).cut(rod)
    return body


def _rotate(shape, p: P, angle):
    """Положительный угол выводит низ лопатки наружу (+Y)."""
    py, pz = p.pivot
    return shape.rotate((0, py, pz), (1, py, pz), angle)


def flap(p: P = P(), angle: float = 0.0) -> cq.Workplane:
    body = _paddle(p)
    return _rotate(body, p, angle) if angle else body


@lru_cache(maxsize=None)
def _sweep(p: P):
    """Объём, который заметает раздутая на зазор лопатка на всём ходе."""
    base = _paddle(p, grow=p.gap)
    env = base
    steps = int(np.ceil(p.swing_deg / 2.0))
    for i in range(1, steps + 1):
        env = env.union(_rotate(base, p, p.swing_deg * i / steps))
    return env


# ---------- корпус ----------

def bezel(p: P = P()) -> cq.Workplane:
    face = _prism(p, "outer", 0, p.proud)
    face = face.faces(">Y").edges().fillet(p.front_fillet)

    # Проём: сквозь лицо до дна мелкой части
    body = face.cut(_prism(p, "inner", p.floor_y, p.proud + 1))
    inner = np.array(_contour(p, "inner"))
    (ox0, oz0), (ox1, oz1) = inner.min(0), inner.max(0)
    try:
        sel = cq.selectors.BoxSelector((ox0 - 1, p.proud - 0.1, oz0 - 1), (ox1 + 1, p.proud + 0.1, oz1 + 1))
        body = body.edges(sel).fillet(p.open_fillet)
    except Exception:
        pass   # скругление кромки проёма — косметика, без него деталь та же

    # Глубокий карман за обшивкой
    cup = (_prism(p, "inner", -p.cup_depth - p.cup_wall, p.floor_y + 0.01, offset=p.cup_clear + p.cup_wall)
           .intersect(_box(p.deep_from_x - p.cup_wall, 300, -100, 100, -100, 100)))
    body = body.union(cup)
    cav = (_prism(p, "inner", -p.cup_depth, p.floor_y + 0.5, offset=p.cup_clear)
           .intersect(_box(p.deep_from_x, 300, -100, 100, -100, 100)))
    body = body.cut(cav)

    # Личинка: отверстие и трубка-направляющая
    kx, kz = p.key_xz
    if p.key:
        tube = _xz_wp(0).center(kx, kz).circle(p.key_d / 2 + p.key_tube_wall).extrude(p.key_tube_len)
        body = body.union(tube)
        hole = (_xz_wp(p.proud + 1).center(kx, kz).circle(p.key_d / 2)
                .extrude(p.proud + p.key_tube_len + 2))
        body = body.cut(hole)

    # Зазоры под ход лопатки: вырезаем всё, что она заметает
    body = body.cut(_sweep(p))

    # Кронштейны оси: от задней стенки кармана до оси
    py, pz = p.pivot
    for x0, x1 in p.brackets_x:
        pts = _circle_pts((py, pz), p.bracket_r) + [
            (-p.cup_depth - p.cup_wall, pz + p.bracket_r), (-p.cup_depth - p.cup_wall, pz - 8),
        ]
        body = body.union(_yz_hull(pts, x0, x1))
    # Упоры: рёбра от задней стенки до загиба лопатки, держат её закрытой
    lip_back = p.face_y - p.lip_depth - p.stop_clear
    for sx in p.stop_x:
        z = p.split_z(sx)
        body = body.union(_box(sx - p.stop_w / 2, sx + p.stop_w / 2, -p.cup_depth - 0.5, lip_back,
                               z - 2.0, z + p.lip_t))

    # Отверстие под штифт: через торец кармана и оба кронштейна (штифт вставляется слева)
    _, (_, bx1) = p.brackets_x
    start = p.deep_from_x - p.cup_wall - 5
    pin = (cq.Workplane("YZ", origin=(start, 0, 0)).center(py, pz)
           .circle(p.pin_d / 2).extrude(bx1 - start))
    body = body.cut(pin)

    # Бобышки крепления
    for bx, bz in p.bosses:
        boss = _xz_wp(0).center(bx, bz).circle(p.boss_d / 2).extrude(p.boss_len)
        body = body.union(boss)
        ins = (_xz_wp(-p.boss_len + p.insert_depth).center(bx, bz).circle(p.insert_d / 2)
               .extrude(p.insert_depth + 1))
        body = body.cut(ins)

    return body


def mirrored(shape: cq.Workplane) -> cq.Workplane:
    """Ручка левой стороны — зеркало по X."""
    return shape.mirror("YZ")


VARIANTS = {
    "key":   replace(P(), key=True),   # передние двери с личинкой
    "plain": P(),                      # двери без личинки
}
