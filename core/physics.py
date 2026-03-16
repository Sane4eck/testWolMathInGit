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

# @njit(cache=True)
# def examples(a: float, b: float) -> float:
#     """ приклад """
#     res = a / b
#     return res
