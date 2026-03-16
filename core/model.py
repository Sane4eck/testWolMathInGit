# core/model.py
import numpy as np
from numba import njit

from core.system import (
    NY, NAUX,
    rhs, clamp_y_inplace,
    initial_y
)

# @njit(cache=True)
@njit(cache=False)
def simulate_rk4(endTime, dt, stepPrint, y0, p):
    n = int(endTime / dt)
    nsave = n // stepPrint + 1

    t_out   = np.empty(nsave, dtype=np.float64)
    y_out   = np.empty((nsave, NY), dtype=np.float64)
    dy_out  = np.empty((nsave, NY), dtype=np.float64)
    aux_out = np.empty((nsave, NAUX), dtype=np.float64)

    y = y0.copy()
    t = 0.0

    k1 = np.empty(NY, dtype=np.float64)
    k2 = np.empty(NY, dtype=np.float64)
    k3 = np.empty(NY, dtype=np.float64)
    k4 = np.empty(NY, dtype=np.float64)
    aux = np.empty(NAUX, dtype=np.float64)
    yt  = np.empty(NY, dtype=np.float64)

    k = 0
    for i in range(n):
        t = i * dt
        rhs(t, y, p, k1, aux)

        if i % stepPrint == 0:
            t_out[k] = t
            for j in range(NY):
                y_out[k, j]  = y[j]
                dy_out[k, j] = k1[j]
            for j in range(NAUX):
                aux_out[k, j] = aux[j]
            k += 1

        # k2
        for j in range(NY):
            yt[j] = y[j] + 0.5*dt*k1[j]
        rhs(t + 0.5*dt, yt, p, k2, aux)

        # k3
        for j in range(NY):
            yt[j] = y[j] + 0.5*dt*k2[j]
        rhs(t + 0.5*dt, yt, p, k3, aux)

        # k4
        for j in range(NY):
            yt[j] = y[j] + dt*k3[j]
        rhs(t + dt, yt, p, k4, aux)

        # update
        for j in range(NY):
            y[j] = y[j] + (dt/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])

        clamp_y_inplace(y)
        # t += dt

    return t_out[:k], y_out[:k], dy_out[:k], aux_out[:k]


class HydraulicModel:
    def simulate(self, state0, params, dt, endTime, countPoint=1000, backend="numba"):
        stepPrint = max(int(abs((state0.time - endTime)) / (dt * countPoint)), 1)

        # y0
        if getattr(state0, "y", None) is None:
            y0 = initial_y()
        else:
            y0 = np.asarray(state0.y, dtype=np.float64)

        p = params.as_tuple()
        return simulate_rk4(endTime, dt, stepPrint, y0, p)
