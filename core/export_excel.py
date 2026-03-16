# core/export_excel.py
import os
from openpyxl import Workbook
from core.system import Y_ORDER, AUX_ORDER

def _is_pressure(name: str) -> bool:
    return name.startswith("p") or name.startswith("dp") or name.startswith("dP")

def save_to_excel(res, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("data")

    dy_names = tuple("d" + n for n in Y_ORDER)

    headers = ["time_s"]
    headers += [f"{n}_bar" if _is_pressure(n) else n for n in Y_ORDER]
    headers += [f"{n}_bar" if _is_pressure(n) else n for n in AUX_ORDER]
    headers += [f"{n}_bar_s" if _is_pressure(n) else n for n in dy_names]
    ws.append(headers)

    t, y, dy, aux = res.t, res.y, res.dy, res.aux

    for i in range(len(t)):
        row = [float(t[i])]

        for j, n in enumerate(Y_ORDER):
            v = float(y[i, j])
            row.append(v * 1e-5 if _is_pressure(n) else v)

        for j, n in enumerate(AUX_ORDER):
            v = float(aux[i, j])
            row.append(v * 1e-5 if _is_pressure(n) else v)

        for j, n in enumerate(dy_names):
            v = float(dy[i, j])
            row.append(v * 1e-5 if _is_pressure(n) else v)

        ws.append(row)

    wb.save(filepath)
