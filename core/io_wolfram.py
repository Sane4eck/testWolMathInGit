# core/io_wolfram.py
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class UnitSystem:
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

def si_or_sgs(name: str) -> UnitSystem:
    if name == "SI":
        return UnitSystem(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    if name == "SGS":
        return UnitSystem(
            980.665,
            1e-5 / 0.980665,
            1e-6,
            1e-2 / 980.665,
            1e2,
            1e4,
            1e6,
            100 / 9.80665,
            1e4,
            980.665 * 1e2,
            1e-4 / 9.80665,
            1e2 / 9.80665,
            1e7,
        )
    raise ValueError(f"Unknown unit system: {name}")
