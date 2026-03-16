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
