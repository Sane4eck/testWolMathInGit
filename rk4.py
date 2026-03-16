from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = np.ndarray
RhsFunc = Callable[[float, ArrayLike], ArrayLike]


def rk4_step(rhs: RhsFunc, y: ArrayLike, t: float, h: float) -> ArrayLike:
    """Exact Python equivalent of Mathematica SolRungeKutta4."""
    y = np.asarray(y, dtype=float)
    k1 = h * np.asarray(rhs(t, y), dtype=float)
    k2 = h * np.asarray(rhs(t + 0.5 * h, y + 0.5 * k1), dtype=float)
    k3 = h * np.asarray(rhs(t + 0.5 * h, y + 0.5 * k2), dtype=float)
    k4 = h * np.asarray(rhs(t + h, y + k3), dtype=float)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def euler_step(rhs: RhsFunc, y: ArrayLike, t: float, h: float) -> ArrayLike:
    y = np.asarray(y, dtype=float)
    return y + h * np.asarray(rhs(t, y), dtype=float)


def euler2_step(rhs: RhsFunc, y: ArrayLike, t: float, h: float) -> ArrayLike:
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(rhs(t, y), dtype=float)
    k2 = np.asarray(rhs(t + h, y + h * k1), dtype=float)
    return y + 0.5 * h * (k1 + k2)


def euler2_modified_step(rhs: RhsFunc, y: ArrayLike, t: float, h: float) -> ArrayLike:
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(rhs(t, y), dtype=float)
    k2 = np.asarray(rhs(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    return y + h * k2
