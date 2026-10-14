"""Наружная ручка двери Toyota Corolla Ceres (AE100/AE101), черновая модель.

Тип ручки — «лопатка» (flap): пластина поднимается вверх-наружу и
поворачивается вокруг горизонтальной оси у верхнего края. Ручка состоит
из двух печатных деталей и стального штифта:

  * корпус (bezel)  — накладка на дверь с чашкой под пальцы;
  * лопатка (flap)  — подвижная часть с рычагом под тягу замка;
  * штифт Ø4 мм     — ось лопатки, проходит через торцы чашки.

Система координат (вид на дверь снаружи):
  X — вдоль машины, +X в сторону замка (задний край двери, при виде
      снаружи он слева);
  Y — по нормали к двери, +Y наружу, Y=0 — наружная поверхность обшивки;
  Z — вверх.

В таком виде получается ручка ПРАВОЙ стороны машины: у японской
праворульной Ceres это водительская дверь. Левая сторона — зеркало по X
(export_all делает оба варианта).

Все размеры ниже — оценка по типовым ручкам Toyota того поколения, а не
замер. Перед печатью их надо сверить с оригиналом (см. README).
"""

from dataclasses import dataclass, replace

import cadquery as cq


@dataclass(frozen=True)
class P:
    # Проём под лопатку
    open_len: float = 120.0      # длина проёма по X
    open_h: float = 38.0         # высота проёма по Z
    open_r_bot: float = 6.0      # скругление нижних углов проёма

    # Корпус-накладка
    margin_x: float = 14.0       # поле накладки от проёма до края по X
    margin_z: float = 9.0        # поле накладки над/под проёмом
    bezel_r: float = 10.0        # скругление углов накладки (вид спереди)
    proud: float = 5.0           # насколько лицо накладки выступает над обшивкой
    edge_fillet: float = 2.5     # скругление кромки лица накладки
    wall: float = 2.2            # стенки чашки
    cup_depth: float = 22.0      # глубина чашки от обшивки внутрь двери

    # Крепление: бобышки под термовставки M5 (болт изнутри двери)
    boss_d: float = 10.0
    boss_len: float = 4.0        # выступ бобышки за обшивку
    insert_d: float = 6.4        # отверстие под термовставку M5
    insert_depth: float = 8.0

    # Личинка замка (только передние двери)
    key: bool = False
    key_extra: float = 30.0      # насколько удлиняется накладка под личинку
    key_hole_d: float = 22.0
    key_tube_d: float = 28.0
    key_tube_len: float = 16.0

    # Лопатка
    gap: float = 0.8             # зазор лопатки в проёме
    flap_t: float = 3.5          # толщина пластины лопатки
    barrel_r: float = 4.5        # радиус втулки оси
    pin_d: float = 4.0           # штифт оси (стальной пруток / гвоздь Ø4)
    pin_clear: float = 0.3       # люфт штифта во втулке лопатки
    finger_gap: float = 14.0     # открытая щель под пальцы снизу
    lip_depth: float = 7.0       # загиб низа лопатки внутрь (зацеп для пальцев)
    lip_t: float = 3.0

    # Рычаг на тягу замка
    arm_from_end: float = 10.0   # центр рычага от торца проёма со стороны замка
    arm_w: float = 6.0
    arm_h: float = 9.0
    arm_reach: float = 36.0      # вылет рычага внутрь двери от оси
    rod_hole_d: float = 5.0      # под клипсу/конец тяги
    swing_deg: float = 25.0      # ход лопатки (для проверки и рендера)

    # Упоры закрытого положения: лопатка ложится на них изнутри
    stop_w: float = 3.5          # насколько упор заходит под лопатку по X
    stop_h: float = 5.0
    stop_depth: float = 4.0

    # --- производные величины ---
    @property
    def bezel_x0(self):
        return -self.open_len / 2 - self.margin_x

    @property
    def bezel_x1(self):
        return self.open_len / 2 + self.margin_x + (self.key_extra if self.key else 0)

    @property
    def bezel_h(self):
        return self.open_h + 2 * self.margin_z

    @property
    def pivot(self):
        """(Y, Z) оси лопатки: втулка касается плоскости лица накладки."""
        return (self.proud - self.barrel_r, self.open_h / 2 - self.gap - self.barrel_r)

    @property
    def key_x(self):
        return self.open_len / 2 + self.margin_x + self.key_extra / 2 - 2

    @property
    def boss_xs(self):
        inner = self.open_len / 2 + self.wall
        left = -(inner + self.margin_x / 2)
        right = (self.bezel_x1 - self.margin_x / 2) if self.key else inner + self.margin_x / 2
        return (left, right)


def _xz_box(x0, x1, z0, z1, y0, y1):
    """Прямоугольный брусок по границам."""
    return (cq.Workplane("XY")
            .box(x1 - x0, y1 - y0, z1 - z0)
            .translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)))


def _opening_prism(p: P, grow: float, y0: float, y1: float):
    """Профиль проёма (вид спереди), выдавленный по Y от y0 до y1.

    grow расширяет профиль во все стороны (для внешних стенок чашки).
    Нижние углы скруглены, верхние — острые: там стоит втулка оси.
    """
    x0, x1 = -p.open_len / 2 - grow, p.open_len / 2 + grow
    z0, z1 = -p.open_h / 2 - grow, p.open_h / 2 + grow
    body = _xz_box(x0, x1, z0, z1, y0, y1)
    r = p.open_r_bot + grow
    return body.edges("|Y").edges("<Z").fillet(r)


def bezel(p: P = P()) -> cq.Workplane:
    x0, x1, h = p.bezel_x0, p.bezel_x1, p.bezel_h

    # Накладка: плита от обшивки до лица, скруглённая в плане и по кромке
    plate = (_xz_box(x0, x1, -h / 2, h / 2, 0, p.proud)
             .edges("|Y").fillet(p.bezel_r)
             .faces(">Y").edges().fillet(p.edge_fillet))

    # Чашка за проёмом, уходит сквозь вырез в обшивке внутрь двери
    cup_outer = _opening_prism(p, p.wall, -p.cup_depth - p.wall, p.proud)
    body = plate.union(cup_outer)
    body = body.cut(_opening_prism(p, 0, -p.cup_depth, p.proud + 1))

    # Упоры на торцах чашки: держат лопатку заподлицо, когда пружина её прижимает
    zb = -p.open_h / 2 + p.finger_gap
    back = p.proud - p.flap_t
    for sx in (-1, 1):
        x_wall = sx * p.open_len / 2
        stop = _xz_box(min(x_wall, x_wall - sx * p.stop_w), max(x_wall, x_wall - sx * p.stop_w),
                       zb + p.lip_t + 0.5, zb + p.lip_t + 0.5 + p.stop_h,
                       back - p.stop_depth, back - 0.1)
        body = body.union(stop)

    # Прорезь в дне чашки под рычаг лопатки (с запасом на ход вниз)
    py, pz = p.pivot
    ax = p.open_len / 2 - p.arm_from_end
    slot_w = p.arm_w + 2 * 1.0
    body = body.cut(_xz_box(ax - slot_w / 2, ax + slot_w / 2,
                            pz - p.arm_h / 2 - 20, pz + p.arm_h / 2 + 1.5,
                            -p.cup_depth - p.wall - 1, -p.cup_depth + 1))

    # Отверстия под штифт оси в торцах чашки (посадка с натягом)
    pin = (cq.Workplane("YZ").circle(p.pin_d / 2)
           .extrude(p.open_len + 2 * p.wall + 4)
           .translate((-(p.open_len / 2 + p.wall + 2), py, pz)))
    body = body.cut(pin)

    # Личинка замка: отверстие в накладке и трубка-направляющая
    if p.key:
        tube = (cq.Workplane("XZ").circle(p.key_tube_d / 2)
                .extrude(p.key_tube_len)  # XZ выдавливается в -Y
                .translate((p.key_x, 0, 0)))
        body = body.union(tube)
        hole = (cq.Workplane("XZ").circle(p.key_hole_d / 2)
                .extrude(p.key_tube_len + p.proud + 2)
                .translate((p.key_x, p.proud + 1, 0)))
        body = body.cut(hole)

    # Бобышки крепления с отверстиями под термовставки
    for bx in p.boss_xs:
        boss = (cq.Workplane("XZ").circle(p.boss_d / 2).extrude(p.boss_len)
                .translate((bx, 0, 0)))
        body = body.union(boss)
        ins = (cq.Workplane("XZ").circle(p.insert_d / 2).extrude(p.insert_depth)
               .translate((bx, -p.boss_len + p.insert_depth, 0)))
        body = body.cut(ins)

    return body


def flap(p: P = P(), angle: float = 0.0) -> cq.Workplane:
    """Лопатка в закрытом положении; angle>0 — открыта на этот угол."""
    py, pz = p.pivot
    xl = p.open_len / 2 - p.gap
    zb = -p.open_h / 2 + p.finger_gap      # нижний край лопатки

    # Втулка оси на всю длину
    barrel = (cq.Workplane("YZ").circle(p.barrel_r).extrude(2 * xl)
              .translate((-xl, py, pz)))

    # Пластина: лицо заподлицо с накладкой
    plate = _xz_box(-xl, xl, zb, pz, p.proud - p.flap_t, p.proud)
    plate = (plate.edges("|Y").edges("<Z").fillet(p.open_r_bot - p.gap)
             .faces(">Y").edges("not >Z").fillet(1.2))

    # Загиб низа внутрь — за него цепляются пальцы
    lip_y0, lip_y1 = p.proud - p.flap_t - p.lip_depth, p.proud - p.flap_t + 0.01
    lip = (_xz_box(-xl, xl, zb, pz, lip_y0, lip_y1)
           .edges("|Y").edges("<Z").fillet(p.open_r_bot - p.gap)
           .intersect(_xz_box(-xl - 1, xl + 1, zb - 1, zb + p.lip_t, lip_y0 - 1, lip_y1 + 1)))

    body = barrel.union(plate).union(lip)

    # Рычаг на тягу: уходит от оси внутрь двери через прорезь в чашке
    ax = p.open_len / 2 - p.arm_from_end
    arm = (_xz_box(ax - p.arm_w / 2, ax + p.arm_w / 2,
                   pz - p.arm_h / 2, pz + p.arm_h / 2,
                   py - p.arm_reach, py)
           .edges("|X").edges("<Y").fillet(p.arm_h / 2 - 0.01))
    rod = (cq.Workplane("YZ").circle(p.rod_hole_d / 2).extrude(p.arm_w + 2)
           .translate((ax - p.arm_w / 2 - 1, py - p.arm_reach + p.arm_h / 2, pz)))
    body = body.union(arm).cut(rod)

    # Отверстие под штифт (свободная посадка)
    pin = (cq.Workplane("YZ").circle(p.pin_d / 2 + p.pin_clear)
           .extrude(2 * xl + 2).translate((-xl - 1, py, pz)))
    body = body.cut(pin)

    if angle:
        # Положительный угол выводит низ лопатки наружу (+Y)
        body = body.rotate((0, py, pz), (1, py, pz), angle)
    return body


def mirrored(shape: cq.Workplane) -> cq.Workplane:
    """Ручка левой стороны — зеркало по X."""
    return shape.mirror("YZ")


VARIANTS = {
    "key":   replace(P(), key=True),   # передние двери с личинкой
    "plain": P(),                      # двери без личинки
}
