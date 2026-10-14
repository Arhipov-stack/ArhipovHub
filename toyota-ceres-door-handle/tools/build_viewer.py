"""Собирает самодостаточную HTML-страницу с 3D-просмотром ручки.

Запуск из корня проекта:  python3 tools/build_viewer.py <выходной.html>
Берёт STL правой стороны из stl/ (левая в просмотрщике — зеркало).
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import handle as h  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(out):
    data = {}
    for name in h.VARIANTS:
        data[name] = {part: base64.b64encode((ROOT / "stl" / f"{name}_right_{part}.stl").read_bytes()).decode()
                      for part in ("bezel", "flap")}
    p = h.VARIANTS["key"]
    dims = [
        ("Накладка, длина (с личинкой / без)", f"{p.bezel_x1 - p.bezel_x0:.0f} / {2 * -h.P().bezel_x0:.0f} мм"),
        ("Накладка, высота", f"{p.bezel_h:.0f} мм"),
        ("Выступ над обшивкой", f"{p.proud:.0f} мм"),
        ("Проём под лопатку", f"{p.open_len:.0f} × {p.open_h:.0f} мм"),
        ("Глубина чашки за обшивкой", f"{p.cup_depth + p.wall:.0f} мм"),
        ("Крепление, межосевое", f"{p.boss_xs[1] - p.boss_xs[0]:.0f} / {h.P().boss_xs[1] - h.P().boss_xs[0]:.0f} мм"),
        ("Ход лопатки", f"{p.swing_deg:.0f}°"),
        ("Вылет рычага от оси", f"{p.arm_reach:.0f} мм"),
        ("Штифт оси", f"Ø{p.pin_d:.0f} мм"),
        ("Личинка замка", f"Ø{p.key_hole_d:.0f} мм"),
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
