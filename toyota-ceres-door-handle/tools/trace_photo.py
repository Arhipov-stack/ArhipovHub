"""Снимает контуры ручки с фото (вид спереди) и сохраняет их в outline_photo.json.

Запуск:  python3 -I tools/trace_photo.py <фото.jpg>

Фото — ручка FL (передняя левая) анфас на белом фоне, личинка справа.
Сами фото в репозиторий не кладём (чужие, с водяными знаками), только
снятые с них контуры в пикселях. В миллиметры их переводит handle.py.

Что снимается:
  * проём (лопатка + карман под пальцы) — самая тёмная область, порог 50;
  * наружный край накладки — проём, раздутый на ширину ободка, плюс эллипс
    вокруг личинки, и выпуклая оболочка (напрямую с маски нельзя: сверху
    и снизу из-за ракурса видна подошва накладки);
  * отверстие личинки, нижняя кромка лопатки, положение кронштейнов оси.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as nd
from scipy.spatial import ConvexHull

OUT = Path(__file__).with_name("outline_photo.json")

# Замерено на этом фото вручную (пиксели), см. README
KEY_CENTER = (894.0, 418.0)
KEY_DIAM_PX = 104.0
KEY_END_ELLIPSE = (880.0, 385.0, 136.0, 150.0)   # cx, cy, полуоси по x и y
RIM_PX = 26          # ширина ободка вокруг проёма
RIM_SHIFT_PX = 3     # нижний ободок шире верхнего
SPLIT_LINE = ((200.0, 393.0), (760.0, 374.0))   # нижняя кромка лопатки
BRACKETS_U = ((318.0, 396.0), (652.0, 722.0))   # кронштейны оси (по u)
FACE_TOP_V = 228.0   # верх лица накладки над серединой (без подошвы)


def polar_contour(mask, n=72):
    """Контур звёздной области: радиус границы по лучам из центроида."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    pts = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        dx, dy = np.cos(a), np.sin(a)
        r = 0.0
        while True:
            x, y = int(round(cx + (r + 1) * dx)), int(round(cy + (r + 1) * dy))
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
                break
            r += 1
        pts.append((cx + r * dx, cy + r * dy))
    pts = np.array(pts)
    # Лёгкое сглаживание по соседям, чтобы сплайн не ловил шум JPEG
    sm = (np.roll(pts, 1, 0) + 2 * pts + np.roll(pts, -1, 0)) / 4
    return sm


def main(photo):
    rgb = np.asarray(Image.open(photo).convert("RGB")).astype(int)
    g = rgb.mean(2)
    h, w = g.shape

    dark = nd.binary_opening(g < 50, iterations=2)
    lab, _ = nd.label(dark)
    inner = nd.binary_fill_holes(lab == lab[450, 500])

    yy, xx = np.mgrid[0:h, 0:w]
    disk = (np.hypot(*np.mgrid[-RIM_PX:RIM_PX + 1, -RIM_PX:RIM_PX + 1]) <= RIM_PX)
    grown = nd.binary_dilation(inner, structure=disk)
    grown = np.roll(grown, RIM_SHIFT_PX, axis=0)
    ex, ey, ea, eb = KEY_END_ELLIPSE
    ell = ((xx - ex) / ea) ** 2 + ((yy - ey) / eb) ** 2 <= 1
    union = grown | ell
    ys, xs = np.nonzero(union)
    pts = np.column_stack([xs, ys])
    hull = ConvexHull(pts)
    outer = np.zeros_like(union)
    poly = pts[hull.vertices]
    from matplotlib.path import Path as MPath
    outer = MPath(poly).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(h, w)

    data = {
        "source": Path(photo).name,
        "note": "pixels, u to the right, v down; front view of the FL handle",
        "inner": polar_contour(inner).round(1).tolist(),
        "outer": polar_contour(outer).round(1).tolist(),
        "key_center": KEY_CENTER,
        "key_diam": KEY_DIAM_PX,
        "split_line": SPLIT_LINE,
        "brackets_u": BRACKETS_U,
        "face_top_v": FACE_TOP_V,
    }
    ys, xs = np.nonzero(outer)
    data["outer_bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
    ys, xs = np.nonzero(inner)
    data["inner_bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
    OUT.write_text(json.dumps(data, indent=1))
    print("saved", OUT, "outer bbox", data["outer_bbox"], "inner bbox", data["inner_bbox"])


if __name__ == "__main__":
    main(sys.argv[1])
