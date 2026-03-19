from pathlib import Path

import numpy as np
import pandas as pd

from core.io_wolfram import UnitSystem


def load_general_params(path: str | Path, units: UnitSystem) -> dict[str, float]:
    raw = pd.read_excel(path, sheet_name=0, header=None, usecols=[0, 1, 2])
    raw = raw.dropna(how="all")

    result: dict[str, float] = {}

    skip_keys = {"", "nan", "name", "parameter", "var"}
    skip_values = {"", "nan", "value", "value si", "value sgs"}
    skip_kinds = {"", "nan", "type", "kind"}

    for _, row in raw.iterrows():
        key_raw = row.iloc[0]
        value_raw = row.iloc[1]
        kind_raw = row.iloc[2]

        key = str(key_raw).strip()
        value_text = str(value_raw).strip()
        kind = str(kind_raw).strip()

        if key.lower() in skip_keys:
            continue
        if value_text.lower() in skip_values:
            continue
        if kind.lower() in skip_kinds:
            continue
        if pd.isna(value_raw):
            continue

        try:
            if isinstance(value_raw, str):
                value_raw = value_raw.replace(",", ".").strip()
            value = float(value_raw)
        except (TypeError, ValueError):
            continue

        converted = calc_coef(value, kind.lower(), units)
        if isinstance(converted, str):
            continue

        result[key] = converted
        result[f"{key}Nom"] = converted

    return result

def _clean_numeric_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = df.copy()
    out.iloc[:, 0] = pd.to_numeric(out.iloc[:, 0], errors="coerce")
    out.iloc[:, 1] = pd.to_numeric(out.iloc[:, 1], errors="coerce")
    out = out.dropna(subset=[out.columns[0], out.columns[1]])
    out = out.drop_duplicates(subset=[out.columns[0]], keep="first")
    out = out.sort_values(out.columns[0])
    x = out.iloc[:, 0].to_numpy(dtype=np.float64)
    y = out.iloc[:, 1].to_numpy(dtype=np.float64)
    return x, y


def load_pressure_vapor_arrays(
    path: str | Path,
    units: UnitSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_excel(path, sheet_name=3, header=None)

    x_ox, y_ox = _clean_numeric_xy(raw.iloc[2:24, 0:2])
    x_fu, y_fu = _clean_numeric_xy(raw.iloc[2:11, 4:6])

    return (
        x_ox,
        y_ox * units.delta_p,
        x_fu,
        y_fu * units.delta_p,
    )


def load_interp2d_arrays(
    path: str | Path,
    sheet_number: int,
    last_row: int,
    last_column: int,
    coef_row: float,
    coef_col: float,
    coef_val: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sheet = pd.read_excel(path, sheet_name=sheet_number - 1, header=None)

    values = sheet.iloc[:last_row, :last_column].to_numpy(dtype=np.float64).T * coef_val
    rows = sheet.iloc[:last_row, last_column].to_numpy(dtype=np.float64) * coef_row
    cols = sheet.iloc[last_row, :last_column].to_numpy(dtype=np.float64) * coef_col

    return (
        np.asarray(cols, dtype=np.float64),
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
def _clean_numeric_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    out = df.copy()
    out.iloc[:, 0] = pd.to_numeric(out.iloc[:, 0], errors="coerce")
    out.iloc[:, 1] = pd.to_numeric(out.iloc[:, 1], errors="coerce")
    out = out.dropna(subset=[out.columns[0], out.columns[1]])
    out = out.drop_duplicates(subset=[out.columns[0]], keep="first")
    out = out.sort_values(out.columns[0])
    x = out.iloc[:, 0].to_numpy(dtype=np.float64)
    y = out.iloc[:, 1].to_numpy(dtype=np.float64)
    return x, y


def load_pressure_vapor_arrays(
    path: str | Path,
    units: UnitSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_excel(path, sheet_name=3, header=None)

    x_ox, y_ox = _clean_numeric_xy(raw.iloc[2:24, 0:2])
    x_fu, y_fu = _clean_numeric_xy(raw.iloc[2:11, 4:6])

    return (
        x_ox,
        y_ox * units.delta_p,
        x_fu,
        y_fu * units.delta_p,
    )


def load_interp2d_arrays(
    path: str | Path,
    sheet_number: int,
    last_row: int,
    last_column: int,
    coef_row: float,
    coef_col: float,
    coef_val: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sheet = pd.read_excel(path, sheet_name=sheet_number - 1, header=None)

    values = sheet.iloc[:last_row, :last_column].to_numpy(dtype=np.float64).T * coef_val
    rows = sheet.iloc[:last_row, last_column].to_numpy(dtype=np.float64) * coef_row
    cols = sheet.iloc[last_row, :last_column].to_numpy(dtype=np.float64) * coef_col

    return (
        np.asarray(cols, dtype=np.float64),
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
