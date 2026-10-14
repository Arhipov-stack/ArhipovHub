"""Экспорт STEP/STL всех вариантов ручки.

Рендеры снимаются из просмотрщика (build_viewer.py), см. README.

Запуск из корня проекта:  python3 tools/export_all.py
"""

import sys
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).parent))
import handle as h  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def export():
    for name, p in h.VARIANTS.items():
        parts = {"bezel": h.bezel(p), "flap": h.flap(p)}
        for side in ("right", "left"):
            for part, shape in parts.items():
                s = shape if side == "right" else h.mirrored(shape)
                stem = f"{name}_{side}_{part}"
                cq.exporters.export(s, str(ROOT / "step" / f"{stem}.step"))
                cq.exporters.export(s, str(ROOT / "stl" / f"{stem}.stl"),
                                    tolerance=0.02, angularTolerance=0.1)
                print("exported", stem)


if __name__ == "__main__":
    export()
