from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d


@dataclass(slots=True)
class UnitSystem:
    name: str
    g0: float
    delta_p: float
    delta_rho: float
    delta_j: float
    delta_d: float
    delta_f: float
    delta_v: float
    delta_r: float
    delta_jinert: float
    delta_c: float
    delta_rkav: float
    delta_torq: float
    delta_cp: float


def si_or_sgs(name: str = "SI") -> UnitSystem:
    if name == "SI":
        return UnitSystem(
            name="SI",
            g0=1.0,
            delta_p=1.0,
            delta_rho=1.0,
            delta_j=1.0,
            delta_d=1.0,
            delta_f=1.0,
            delta_v=1.0,
            delta_r=1.0,
            delta_jinert=1.0,
            delta_c=1.0,
            delta_rkav=1.0,
            delta_torq=1.0,
            delta_cp=1.0,
        )
    if name == "SGS":
        return UnitSystem(
            name="SGS",
            g0=980.665,
            delta_p=10 ** (-5) / 0.980665,
            delta_rho=10 ** -6,
            delta_j=10 ** (-2) / 980.665,
            delta_d=10 ** 2,
            delta_f=10 ** 4,
            delta_v=10 ** 6,
            delta_r=100.0 / 9.80665,
            delta_jinert=10 ** 4,
            delta_c=980.665 * 10 ** 2,
            delta_rkav=10 ** -4 / 9.80665,
            delta_torq=10 ** 2 / 9.80665,
            delta_cp=10 ** 7,
        )
    raise ValueError(f"Unsupported unit system: {name}")


def func_import(path: str | Path, sheet: int, rows: Any, cols: Any) -> Any:
    df = pd.read_excel(path, sheet_name=sheet - 1, header=None)
    if rows == "All":
        row_idx = slice(None)
    elif isinstance(rows, range):
        row_idx = [r - 1 for r in rows]
    elif isinstance(rows, list):
        row_idx = [r - 1 for r in rows]
    else:
        row_idx = rows - 1

    if isinstance(cols, range):
        col_idx = [c - 1 for c in cols]
    elif isinstance(cols, list):
        col_idx = [c - 1 for c in cols]
    else:
        col_idx = cols - 1

    out = df.iloc[row_idx, col_idx]
    if isinstance(out, pd.DataFrame):
        return out.to_numpy()
    if isinstance(out, pd.Series):
        return out.to_numpy()
    return out


def calc_coef(value: float, kind: str, units: UnitSystem) -> float | str:
    mapping = {
        "-": value,
        "pressure": value * units.delta_p,
        "density": value * units.delta_rho,
        "inertia": value * units.delta_j * 10.0,
        "diameter": value * units.delta_d,
        "area": value * units.delta_f,
        "volume": value * units.delta_v,
        "gas constant": value * units.delta_r,
        "inertia turb": value * units.delta_jinert,
        "compliance": value * units.delta_c,
        "resist cavitation": value * units.delta_rkav,
        "heat capacity": value * units.delta_cp,
    }
    return mapping.get(kind, "<<< NoTypeVarInExcel >>>")


def load_general_params(path: str | Path, units: UnitSystem) -> dict[str, float]:
    raw = pd.read_excel(path, sheet_name=0, header=None, usecols=[0, 1, 2])
    raw = raw.dropna(how="all")

    result: dict[str, float] = {}

    skip_keys = {
        "",
        "nan",
        "name",
        "parameter",
        "var",
    }

    skip_values = {
        "",
        "nan",
        "value",
        "value si",
        "value sgs",
    }

    skip_kinds = {
        "",
        "nan",
        "type",
        "kind",
    }

    for _, row in raw.iterrows():
        key_raw = row.iloc[0]
        value_raw = row.iloc[1]
        kind_raw = row.iloc[2]

        key = str(key_raw).strip()
        value_text = str(value_raw).strip()
        kind = str(kind_raw).strip()

        key_l = key.lower()
        value_l = value_text.lower()
        kind_l = kind.lower()

        if key_l in skip_keys:
            continue
        if value_l in skip_values:
            continue
        if kind_l in skip_kinds:
            continue

        if pd.isna(value_raw):
            continue

        try:
            if isinstance(value_raw, str):
                value_raw = value_raw.replace(",", ".").strip()
            value = float(value_raw)
        except (TypeError, ValueError):
            continue

        converted = calc_coef(value, kind_l, units)
        if isinstance(converted, str):
            raise ValueError(f"Unknown parameter type for {key!r}: {kind!r}")

        result[key] = converted
        result[f"{key}Nom"] = converted

    return result

def load_valve_cyclogram(path: str | Path, units: UnitSystem) -> dict[str, list[float]]:
    raw = pd.read_excel(path, sheet_name=2, header=None)
    rows = raw.iloc[1:12, 2:10].dropna(how="all")
    result: dict[str, list[float]] = {}
    for _, row in rows.iterrows():
        name = str(row.iloc[0]).strip()
        if name == "nan":
            continue
        vals = [float(v) for v in row.iloc[1:8].tolist()]
        vals[4] *= units.delta_f
        vals[5] *= units.delta_f
        result[name] = vals
    return result


def load_pressure_vapor_tables(path: str | Path, units: UnitSystem) -> tuple[interp1d, interp1d]:
    raw = pd.read_excel(path, sheet_name=3, header=None)
    ox = raw.iloc[2:24, 0:2].dropna()
    fu = raw.iloc[2:11, 4:6].dropna()
    f_ox = interp1d(ox.iloc[:, 0].to_numpy(), ox.iloc[:, 1].to_numpy() * units.delta_p,
                    kind="linear", fill_value="extrapolate")
    f_fu = interp1d(fu.iloc[:, 0].to_numpy(), fu.iloc[:, 1].to_numpy() * units.delta_p,
                    kind="quadratic", fill_value="extrapolate")
    return f_ox, f_fu


def interp2d_from_sheet(
    path: str | Path,
    sheet_number: int,
    last_row: int,
    last_column: int,
    coef_row: float,
    coef_col: float,
) -> RegularGridInterpolator:
    sheet = pd.read_excel(path, sheet_name=sheet_number - 1, header=None)
    values = sheet.iloc[:last_row, :last_column].to_numpy(dtype=float).T
    rows = sheet.iloc[:last_row, last_column].to_numpy(dtype=float) * coef_row
    cols = sheet.iloc[last_row, :last_column].to_numpy(dtype=float) * coef_col
    return RegularGridInterpolator((cols, rows), values, method="linear", bounds_error=False, fill_value=None)


def load_sheet_vector(path: str | Path, sheet: int, rows: range, column: int) -> np.ndarray:
    data = func_import(path, sheet, rows, column)
    return np.asarray(data, dtype=float)
