# core/physics.py
from numba import njit
import math


# @njit(cache=True)
# def examples(a: float, b: float) -> float:
#     """ приклад """
#     res = a / b
#     return res

@njit(cache=True)
def ggaz(P1: float, P2: float, Mu: float, F: float, T1: float, k: float, R: float) -> float:
    """
    Аналог Mathematica:
    GGAZ[P1_,P2_,Mu_,F_,T1_,k_,R_]

    Повертає GG (витрата газу) з урахуванням критичного відношення тисків.
    """
    # g0 якщо знадобиться використання СГС
    if P1 > 1.0e4:
        g0 = 1.0
    else:
        g0 = 980.665

    # якщо немає перепаду в правильний бік — витрата 0
    if P1 <= P2: return 0.0

    PI = P2 / P1
    PIcrit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))

    # “задушення” (choking): PI не менше критичного
    if PI < PIcrit: PI = PIcrit

    term = (g0 * (2.0 * k) / (R * T1 * (k - 1.0))) * (PI ** (2.0 / k) - PI ** ((k + 1.0) / k))
    if term <= 0.0:
        return 0.0

    return P1 * Mu * F * math.sqrt(term)


# @njit(cache=True)
@njit(cache=False)
def linear_law(t, val0, valN, t1, t2):
    if t <= t1:
        return val0
    elif t <= t2:
        return val0 + (valN - val0) / (t2 - t1) * (t - t1)
    else:
        return valN


@njit(cache=True)
def f_valve(f1: float, f2: float, t1: float, t2: float, dt1: float, dt2: float, t: float) -> float:
    """
    площа клапану від циклограми
    """
    # поза інтервалом
    if t < t1 or t >= (t2 + dt2):
        return f1
    # підйом
    if t >= t1 and t < (t1 + dt1):
        if dt1 <= 0.0:
            return f2
        return f1 + (f2 - f1) * (t - t1) / dt1
    # плато
    if t >= (t1 + dt1) and t < t2:
        return f2
    # спад
    if t>= t2 and t < (t2 + dt2):
        if dt2 <= 0.0:
            return f1
        return f2 - (f2 - f1) * (t - t2) / dt2
    return 0

from numba import njit
import math


@njit(cache=True)
def base_pump_eta_q(QtoN: float, coef0: float, coef1: float, coef2: float) -> float:
    den = coef1 * coef1 - QtoN * (2.0 * coef1 - coef2)
    if abs(den) < 1e-30:
        return 0.0
    return coef0 * (coef2 - QtoN) * QtoN / den


@njit(cache=True)
def base_pump_eta_m(m_pump_out: float, rho: float, rpm: float, coef0: float, coef1: float, coef2: float) -> float:
    q_to_n = m_pump_out / (rho * 1.0e-3 * rpm)
    return base_pump_eta_q(q_to_n, coef0, coef1, coef2)


@njit(cache=True)
def base_turb_eta(u_to_c: float, coef0: float, coef1: float) -> float:
    return coef0 * u_to_c + coef1 * u_to_c * u_to_c


@njit(cache=True)
def base_pump_h(m_pump_out: float, rho: float, rpm: float, coef0: float, coef1: float, coef2: float) -> float:
    return coef0 * rho * rpm * rpm + coef1 * rpm * m_pump_out + coef2 * m_pump_out * m_pump_out / rho


@njit(cache=True)
def torque_pump(pressure_out: float, mass_flow: float, rho: float, eta: float, rpm: float) -> float:
    if eta <= 1.0e-12 or rpm <= 1.0e-12:
        return 0.0
    power = pressure_out * (mass_flow / rho) / eta
    return 60.0 * power / (2.0 * math.pi * rpm)


@njit(cache=True)
def torque_turb(p_out: float, p_in: float, m_turb_nom: float, rpm_nom: float,
                eta_turb_nom: float, k_nom: float, r_nom: float, t_nom: float) -> float:
    if p_in <= 1.0e-12 or rpm_nom <= 1.0e-12 or k_nom <= 1.0:
        return 0.0
    lad = k_nom / (k_nom - 1.0) * r_nom * t_nom * (1.0 - (p_out / p_in) ** ((k_nom - 1.0) / k_nom))
    return 60.0 * m_turb_nom * lad * eta_turb_nom / (2.0 * math.pi * rpm_nom)

@njit(cache=True)
def func_filling(
    flag_fill: float,
    flag_val_open: float,
    val_is_open: float,
    vol: float,
    vol_nom: float,
    resis_val_nom: float,
    f_val_max: float,
    resis_pipe1: float,
    resis_pipe2: float,
    inert_pipe1: float,
    inert_pipe2: float,
    p1: float,
    p2: float,
    m: float,
    rho: float,
    f_val: float,
):
    m_fill = m
    m_jet = 0.0
    dm = 0.0
    dvol = 0.0

    if val_is_open > 0.5:
        ratio_vol = 0.0
        if vol_nom > 1.0e-30:
            ratio_vol = vol / vol_nom
        if ratio_vol < 0.0:
            ratio_vol = 0.0

        if flag_fill < 0.5 and ratio_vol >= 1.0:
            ratio_vol = 1.0
            flag_fill = 1.0

        if f_val <= 0.0:
            f_val = 1.0e-10

        resis_val = resis_val_nom * (f_val_max / f_val) ** 2
        fl1 = ratio_vol ** 0.1
        resis_pipe = resis_pipe1 + resis_val + fl1 * resis_pipe2 + 1.0e-20
        inert_pipe = inert_pipe1 + fl1 * inert_pipe2 + 1.0e-20

        if flag_val_open > 0.5:
            dm = (p1 - p2 - resis_pipe * m_fill * abs(m_fill)) / inert_pipe
        else:
            dm = 0.0
            m_fill = 0.0

        if flag_fill > 0.5:
            dvol = 0.0
            m_jet = m_fill
        else:
            fl2 = 1.0 - 0.98 * ratio_vol
            if fl2 < 0.0:
                fl2 = 0.0
            dvol = (m_fill * fl2) / rho
            m_jet = m_fill * (1.0 - fl2)
    else:
        if m_fill < 0.0:
            m_fill = 0.0
        m_jet = m_fill

    return flag_fill, dm, dvol, m_fill, m_jet
# @njit(cache=True)
# def examples(a: float, b: float) -> float:
#     """ приклад """
#     res = a / b
#     return res

from numba import njit
import numpy as np


@njit(cache=False)
def interp1_linear_clamped(x, xp, fp):
    n = xp.size
    if n == 0:
        return 0.0
    if n == 1:
        return fp[0]

    if x <= xp[0]:
        x0 = xp[0]
        x1 = xp[1]
        y0 = fp[0]
        y1 = fp[1]
    elif x >= xp[n - 1]:
        x0 = xp[n - 2]
        x1 = xp[n - 1]
        y0 = fp[n - 2]
        y1 = fp[n - 1]
    else:
        i = 0
        while i < n - 1 and not (xp[i] <= x <= xp[i + 1]):
            i += 1
        x0 = xp[i]
        x1 = xp[i + 1]
        y0 = fp[i]
        y1 = fp[i + 1]

    dx = x1 - x0
    if abs(dx) < 1.0e-30:
        return y0
    return y0 + (y1 - y0) * (x - x0) / dx


@njit(cache=False)
def interp1_quadratic_local(x, xp, fp):
    n = xp.size
    if n < 3:
        return interp1_linear_clamped(x, xp, fp)

    if x <= xp[1]:
        i0, i1, i2 = 0, 1, 2
    elif x >= xp[n - 2]:
        i0, i1, i2 = n - 3, n - 2, n - 1
    else:
        i1 = 1
        while i1 < n - 1 and not (xp[i1] <= x <= xp[i1 + 1]):
            i1 += 1
        i0 = i1 - 1
        i2 = i1 + 1

    x0, x1, x2 = xp[i0], xp[i1], xp[i2]
    y0, y1, y2 = fp[i0], fp[i1], fp[i2]

    d01 = x0 - x1
    d02 = x0 - x2
    d12 = x1 - x2
    if abs(d01) < 1.0e-30 or abs(d02) < 1.0e-30 or abs(d12) < 1.0e-30:
        return interp1_linear_clamped(x, xp, fp)

    l0 = ((x - x1) * (x - x2)) / (d01 * d02)
    l1 = ((x - x0) * (x - x2)) / ((x1 - x0) * d12)
    l2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    return y0 * l0 + y1 * l1 + y2 * l2


@njit(cache=False)
def interp2_bilinear_clamped(x, y, xg, yg, table):
    nx = xg.size
    ny = yg.size

    if nx < 2 or ny < 2:
        return 0.0

    if x <= xg[0]:
        ix = 0
    elif x >= xg[nx - 1]:
        ix = nx - 2
    else:
        ix = 0
        while ix < nx - 1 and not (xg[ix] <= x <= xg[ix + 1]):
            ix += 1

    if y <= yg[0]:
        iy = 0
    elif y >= yg[ny - 1]:
        iy = ny - 2
    else:
        iy = 0
        while iy < ny - 1 and not (yg[iy] <= y <= yg[iy + 1]):
            iy += 1

    x0 = xg[ix]
    x1 = xg[ix + 1]
    y0 = yg[iy]
    y1 = yg[iy + 1]

    q00 = table[ix, iy]
    q10 = table[ix + 1, iy]
    q01 = table[ix, iy + 1]
    q11 = table[ix + 1, iy + 1]

    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1.0e-30 or abs(dy) < 1.0e-30:
        return q00

    tx = (x - x0) / dx
    ty = (y - y0) / dy

    return (
        q00 * (1.0 - tx) * (1.0 - ty)
        + q10 * tx * (1.0 - ty)
        + q01 * (1.0 - tx) * ty
        + q11 * tx * ty
    )

from numba import njit
import numpy as np


@njit(cache=False)
def interp1_linear_clamped(x, xp, fp):
    n = xp.size
    if n == 0:
        return 0.0
    if n == 1:
        return fp[0]

    if x <= xp[0]:
        i = 0
    elif x >= xp[n - 1]:
        i = n - 2
    else:
        i = 0
        while i < n - 1 and not (xp[i] <= x <= xp[i + 1]):
            i += 1

    x0 = xp[i]
    x1 = xp[i + 1]
    y0 = fp[i]
    y1 = fp[i + 1]

    dx = x1 - x0
    if abs(dx) < 1.0e-30:
        return y0
    return y0 + (y1 - y0) * (x - x0) / dx


@njit(cache=False)
def interp1_quadratic_local(x, xp, fp):
    n = xp.size
    if n < 3:
        return interp1_linear_clamped(x, xp, fp)

    if x <= xp[1]:
        i0, i1, i2 = 0, 1, 2
    elif x >= xp[n - 2]:
        i0, i1, i2 = n - 3, n - 2, n - 1
    else:
        i1 = 1
        while i1 < n - 1 and not (xp[i1] <= x <= xp[i1 + 1]):
            i1 += 1
        i0 = i1 - 1
        i2 = i1 + 1

    x0, x1, x2 = xp[i0], xp[i1], xp[i2]
    y0, y1, y2 = fp[i0], fp[i1], fp[i2]

    d01 = x0 - x1
    d02 = x0 - x2
    d12 = x1 - x2
    if abs(d01) < 1.0e-30 or abs(d02) < 1.0e-30 or abs(d12) < 1.0e-30:
        return interp1_linear_clamped(x, xp, fp)

    l0 = ((x - x1) * (x - x2)) / (d01 * d02)
    l1 = ((x - x0) * (x - x2)) / ((x1 - x0) * d12)
    l2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    return y0 * l0 + y1 * l1 + y2 * l2


@njit(cache=False)
def interp2_bilinear_clamped(x, y, xg, yg, table):
    nx = xg.size
    ny = yg.size
    if nx < 2 or ny < 2:
        return 0.0

    if x <= xg[0]:
        ix = 0
    elif x >= xg[nx - 1]:
        ix = nx - 2
    else:
        ix = 0
        while ix < nx - 1 and not (xg[ix] <= x <= xg[ix + 1]):
            ix += 1

    if y <= yg[0]:
        iy = 0
    elif y >= yg[ny - 1]:
        iy = ny - 2
    else:
        iy = 0
        while iy < ny - 1 and not (yg[iy] <= y <= yg[iy + 1]):
            iy += 1

    x0 = xg[ix]
    x1 = xg[ix + 1]
    y0 = yg[iy]
    y1 = yg[iy + 1]

    q00 = table[ix, iy]
    q10 = table[ix + 1, iy]
    q01 = table[ix, iy + 1]
    q11 = table[ix + 1, iy + 1]

    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1.0e-30 or abs(dy) < 1.0e-30:
        return q00

    tx = (x - x0) / dx
    ty = (y - y0) / dy

    return (
        q00 * (1.0 - tx) * (1.0 - ty)
        + q10 * tx * (1.0 - ty)
        + q01 * (1.0 - tx) * ty
        + q11 * tx * ty
    )
