"""Собирает самодостаточную HTML-страницу с 3D-просмотром ручки.

Запуск из корня проекта:  python3 tools/build_viewer.py <выходной.html>
Строит детали правой стороны заново и сохраняет их грубее, чем для печати,
чтобы страница грузилась на телефоне (левая в просмотрщике — зеркало).
"""

import base64
import json
import sys
import tempfile
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).parent))
import handle as h  # noqa: E402



def main(out):
    data = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, p in h.VARIANTS.items():
            data[name] = {}
            for part, shape in (("bezel", h.bezel(p)), ("flap", h.flap(p))):
                f = Path(tmp) / f"{name}_{part}.stl"
                cq.exporters.export(shape, str(f), tolerance=0.12, angularTolerance=0.25)
                data[name][part] = base64.b64encode(f.read_bytes()).decode()
    p = h.VARIANTS["key"]
    outer = __import__("numpy").array(h._contour(p, "outer"))
    inner = __import__("numpy").array(h._contour(p, "inner"))
    (ox0, oz0), (ox1, oz1) = outer.min(0), outer.max(0)
    (ix0, iz0), (ix1, iz1) = inner.min(0), inner.max(0)
    (bx0, _), (_, bx1) = p.brackets_x
    dims = [
        ("Накладка, длина × высота", f"{ox1 - ox0:.0f} × {oz1 - oz0:.0f} мм"),
        ("Выступ над обшивкой", f"{p.proud:.0f} мм"),
        ("Проём (лопатка + карман)", f"{ix1 - ix0:.0f} × {iz1 - iz0:.0f} мм"),
        ("Глубина кармана за обшивкой", f"{p.cup_depth + p.cup_wall:.0f} мм"),
        ("Ось: от обшивки внутрь / над центром", f"{-p.pivot_y:.0f} / {p.pivot_z:.0f} мм"),
        ("Штифт оси", f"Ø{p.pin_d:.0f} × {bx1 - (p.deep_from_x - p.cup_wall):.0f} мм"),
        ("Ход лопатки", f"{p.swing_deg:.0f}°"),
        ("Рычаг тяги: вылет / ход тяги", f"{p.lever_len:.0f} / {p.lever_len * __import__('math').sin(__import__('math').radians(p.swing_deg)):.0f} мм"),
        ("Личинка замка", f"Ø{p.key_d:.0f} мм"),
    ]
    rows = "".join(f"<tr><td>{k}</td><td class=\"v\">{v}</td></tr>" for k, v in dims)
    html = (Path(__file__).with_name("viewer_template.html").read_text()
            .replace("__DATA__", json.dumps(data))
            .replace("__PIVOT__", json.dumps(list(p.pivot)))
            .replace("__SWING__", f"{p.swing_deg:.0f}")
            .replace("__DIMS__", rows))
    Path(out).write_text(html)
    print(out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
