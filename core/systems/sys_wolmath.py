import numpy as np
from numba import njit

from core.physics import (
    ggaz,
    linear_law,
    f_valve,
    base_pump_eta_q,
    base_pump_eta_m,
    base_turb_eta,
    base_pump_h,
    torque_pump,
    torque_turb,
    func_filling,
)

Y_ORDER = (
    "xCVf1", "xCVf2", "xCVf3", "xCVf4", "xCVo1",
    "uCVf1", "uCVf2", "uCVf3", "uCVf4", "uCVo1",
    "masbf", "masbo",
    "pf1", "pf4", "pf5", "pf6", "pf7",
    "poB", "po1", "po3", "po4",
    "pGg", "pGv", "pCh",
    "mfTx1", "mf2xCh", "mf3x4", "mf4x6", "mf6xIc",
    "mf4x5", "mf5xIg", "mfBx7", "mf7x6", "mf7x5",
    "moTx1", "mo2x3", "mo3xGg", "mo2xGg", "mo3x4",
    "mo4xIc", "mo4xIg", "moBx4",
    "volVf1xCh", "volVo1xGg",
    "omega",
)

AUX_ORDER = (
    "mGg", "mGv", "mCh",
    "etaPmpOx", "etaPmp1Fu", "etaPmp2Fu", "etaTrb",
    "torqTrb", "torqPmpOx", "torqPmp1Fu", "torqPmp2Fu",
    "mfChCool", "flagFillFu", "flagFillOx",
)

NY = len(Y_ORDER)
NAUX = len(AUX_ORDER)

I_xCVf1 = 0
I_xCVf2 = 1
I_xCVf3 = 2
I_xCVf4 = 3
I_xCVo1 = 4

I_uCVf1 = 5
I_uCVf2 = 6
I_uCVf3 = 7
I_uCVf4 = 8
I_uCVo1 = 9

I_masbf = 10
I_masbo = 11

I_pf1 = 12
I_pf4 = 13
I_pf5 = 14
I_pf6 = 15
I_pf7 = 16

I_poB = 17
I_po1 = 18
I_po3 = 19
I_po4 = 20

I_pGg = 21
I_pGv = 22
I_pCh = 23

I_mfTx1 = 24
I_mf2xCh = 25
I_mf3x4 = 26
I_mf4x6 = 27
I_mf6xIc = 28
I_mf4x5 = 29
I_mf5xIg = 30
I_mfBx7 = 31
I_mf7x6 = 32
I_mf7x5 = 33

I_moTx1 = 34
I_mo2x3 = 35
I_mo3xGg = 36
I_mo2xGg = 37
I_mo3x4 = 38
I_mo4xIc = 39
I_mo4xIg = 40
I_moBx4 = 41

I_volVf1xCh = 42
I_volVo1xGg = 43
I_omega = 44

A_mGg = 0
A_mGv = 1
A_mCh = 2
A_etaPmpOx = 3
A_etaPmp1Fu = 4
A_etaPmp2Fu = 5
A_etaTrb = 6
A_torqTrb = 7
A_torqPmpOx = 8
A_torqPmp1Fu = 9
A_torqPmp2Fu = 10
A_mfChCool = 11
A_flagFillFu = 12
A_flagFillOx = 13


class Params:
    def __init__(self):
        # initial conditions
        self.xCVf1_0 = 0.0
        self.xCVf2_0 = 0.0
        self.xCVf3_0 = 0.0
        self.xCVf4_0 = 0.0
        self.xCVo1_0 = 0.0

        self.uCVf1_0 = 0.0
        self.uCVf2_0 = 0.0
        self.uCVf3_0 = 0.0
        self.uCVf4_0 = 0.0
        self.uCVo1_0 = 0.0

        self.masbf_0 = 0.0
        self.masbo_0 = 0.0

        self.pf1_0 = 0.0
        self.pf4_0 = 0.0
        self.pf5_0 = 0.0
        self.pf6_0 = 0.0
        self.pf7_0 = 0.0

        self.poB_0 = 0.0
        self.po1_0 = 0.0
        self.po3_0 = 0.0
        self.po4_0 = 0.0

        self.pGg_0 = 0.0
        self.pGv_0 = 0.0
        self.pCh_0 = 0.0

        self.mfTx1_0 = 0.0
        self.mf2xCh_0 = 0.0
        self.mf3x4_0 = 0.0
        self.mf4x6_0 = 0.0
        self.mf6xIc_0 = 0.0
        self.mf4x5_0 = 0.0
        self.mf5xIg_0 = 0.0
        self.mfBx7_0 = 0.0
        self.mf7x6_0 = 0.0
        self.mf7x5_0 = 0.0

        self.moTx1_0 = 0.0
        self.mo2x3_0 = 0.0
        self.mo3xGg_0 = 0.0
        self.mo2xGg_0 = 0.0
        self.mo3x4_0 = 0.0
        self.mo4xIc_0 = 0.0
        self.mo4xIg_0 = 0.0
        self.moBx4_0 = 0.0

        self.volVf1xCh_0 = 0.0
        self.volVo1xGg_0 = 0.0
        self.omega_0 = 1.0

        # shaft / torques
        self.JTrbPmp = 1.0
        self.g0 = 1.0

        # simple valve dynamics coefficients
        self.k_xcv = 0.0
        self.k_ucv = 0.0

    @classmethod
    def from_excel(cls, path: str):
        # тимчасово: щоб система вже запускалась.
        # Потім підключимо реальне читання Excel.
        return cls()

    def as_tuple(self):
        return np.array([
            self.JTrbPmp,
            self.g0,
            self.k_xcv,
            self.k_ucv,
        ], dtype=np.float64)


def initial_y(p: Params) -> np.ndarray:
    return np.array([
        p.xCVf1_0, p.xCVf2_0, p.xCVf3_0, p.xCVf4_0, p.xCVo1_0,
        p.uCVf1_0, p.uCVf2_0, p.uCVf3_0, p.uCVf4_0, p.uCVo1_0,
        p.masbf_0, p.masbo_0,
        p.pf1_0, p.pf4_0, p.pf5_0, p.pf6_0, p.pf7_0,
        p.poB_0, p.po1_0, p.po3_0, p.po4_0,
        p.pGg_0, p.pGv_0, p.pCh_0,
        p.mfTx1_0, p.mf2xCh_0, p.mf3x4_0, p.mf4x6_0, p.mf6xIc_0,
        p.mf4x5_0, p.mf5xIg_0, p.mfBx7_0, p.mf7x6_0, p.mf7x5_0,
        p.moTx1_0, p.mo2x3_0, p.mo3xGg_0, p.mo2xGg_0, p.mo3x4_0,
        p.mo4xIc_0, p.mo4xIg_0, p.moBx4_0,
        p.volVf1xCh_0, p.volVo1xGg_0,
        p.omega_0,
    ], dtype=np.float64)


@njit(cache=False)
def clamp_y_inplace(y: np.ndarray):
    # клапани: хід >= 0
    if y[I_xCVf1] < 0.0: y[I_xCVf1] = 0.0
    if y[I_xCVf2] < 0.0: y[I_xCVf2] = 0.0
    if y[I_xCVf3] < 0.0: y[I_xCVf3] = 0.0
    if y[I_xCVf4] < 0.0: y[I_xCVf4] = 0.0
    if y[I_xCVo1] < 0.0: y[I_xCVo1] = 0.0

    # маси, тиски, об'єми
    if y[I_masbf] < 0.0: y[I_masbf] = 0.0
    if y[I_masbo] < 0.0: y[I_masbo] = 0.0

    if y[I_pf1] < 0.0: y[I_pf1] = 0.0
    if y[I_pf4] < 0.0: y[I_pf4] = 0.0
    if y[I_pf5] < 0.0: y[I_pf5] = 0.0
    if y[I_pf6] < 0.0: y[I_pf6] = 0.0
    if y[I_pf7] < 0.0: y[I_pf7] = 0.0

    if y[I_poB] < 0.0: y[I_poB] = 0.0
    if y[I_po1] < 0.0: y[I_po1] = 0.0
    if y[I_po3] < 0.0: y[I_po3] = 0.0
    if y[I_po4] < 0.0: y[I_po4] = 0.0

    if y[I_pGg] < 0.0: y[I_pGg] = 0.0
    if y[I_pGv] < 0.0: y[I_pGv] = 0.0
    if y[I_pCh] < 0.0: y[I_pCh] = 0.0

    if y[I_volVf1xCh] < 0.0: y[I_volVf1xCh] = 0.0
    if y[I_volVo1xGg] < 0.0: y[I_volVo1xGg] = 0.0

    if y[I_omega] < 1.0e-12:
        y[I_omega] = 1.0e-12


@njit(cache=False)
def rhs(t: float, y: np.ndarray, p_arr: np.ndarray, dy: np.ndarray, aux: np.ndarray):
    clamp_y_inplace(y)

    JTrbPmp = p_arr[0]
    g0 = p_arr[1]
    k_xcv = p_arr[2]
    k_ucv = p_arr[3]

    # unpack
    xCVf1 = y[I_xCVf1]
    xCVf2 = y[I_xCVf2]
    xCVf3 = y[I_xCVf3]
    xCVf4 = y[I_xCVf4]
    xCVo1 = y[I_xCVo1]

    uCVf1 = y[I_uCVf1]
    uCVf2 = y[I_uCVf2]
    uCVf3 = y[I_uCVf3]
    uCVf4 = y[I_uCVf4]
    uCVo1 = y[I_uCVo1]

    omega = y[I_omega]

    # default
    for i in range(NY):
        dy[i] = 0.0
    for i in range(NAUX):
        aux[i] = 0.0

    # -------------------------------------------------
    # 1. Клапани: перший реальний блок правих частин
    # У notebook точно є:
    # DxCV* = uCV*
    # А DuCV* потім уже підставимо з повної моделі клапанів.
    # -------------------------------------------------
    DxCVf1 = uCVf1
    DxCVf2 = uCVf2
    DxCVf3 = uCVf3
    DxCVf4 = uCVf4
    DxCVo1 = uCVo1

    # поки спрощена форма, щоб система вже була жива
    DuCVf1 = -k_xcv * xCVf1 - k_ucv * uCVf1
    DuCVf2 = -k_xcv * xCVf2 - k_ucv * uCVf2
    DuCVf3 = -k_xcv * xCVf3 - k_ucv * uCVf3
    DuCVf4 = -k_xcv * xCVf4 - k_ucv * uCVf4
    DuCVo1 = -k_xcv * xCVo1 - k_ucv * uCVo1

    # -------------------------------------------------
    # 2. Інші блоки поки нульові
    # -------------------------------------------------
    Domega = 0.0

    # write dy
    dy[I_xCVf1] = DxCVf1
    dy[I_xCVf2] = DxCVf2
    dy[I_xCVf3] = DxCVf3
    dy[I_xCVf4] = DxCVf4
    dy[I_xCVo1] = DxCVo1

    dy[I_uCVf1] = DuCVf1
    dy[I_uCVf2] = DuCVf2
    dy[I_uCVf3] = DuCVf3
    dy[I_uCVf4] = DuCVf4
    dy[I_uCVo1] = DuCVo1

    dy[I_omega] = Domega


__all__ = [
    "Params",
    "Y_ORDER",
    "AUX_ORDER",
    "NY",
    "NAUX",
    "initial_y",
    "clamp_y_inplace",
    "rhs",
]
