# core/systems/sys_wolmath.py
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

Y_INDEX = {name: i for i, name in enumerate(Y_ORDER)}
AUX_INDEX = {name: i for i, name in enumerate(AUX_ORDER)}

class Params:
    def __init__(self):
        # тут поки руками або з excel
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

        # далі всі інші *_0
        self.masbf_0 = 0.0
        self.masbo_0 = 0.0
        self.pf1_0 = 0.0
        self.pf4_0 = 0.0
        ...
        self.volVf1xCh_0 = 0.0
        self.volVo1xGg_0 = 0.0
        self.omega_0 = 1.0  # краще не 0, щоб уникати ділення

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

@njit(cache=True)
def rhs(t: float, y: np.ndarray, p_arr: np.ndarray, dy: np.ndarray, aux: np.ndarray):
    # unpack
    xCVf1 = y[Y_INDEX["xCVf1"]]
    xCVf2 = y[Y_INDEX["xCVf2"]]
    ...
    omega = y[Y_INDEX["omega"]]

    # default
    for i in range(dy.size):
        dy[i] = 0.0
    for i in range(aux.size):
        aux[i] = 0.0

    # 1. клапани
    DxCVf1 = uCVf1
    DxCVf2 = uCVf2
    DxCVf3 = uCVf3
    DxCVf4 = uCVf4
    DxCVo1 = uCVo1

    DuCVf1 = 0.0
    DuCVf2 = 0.0
    DuCVf3 = 0.0
    DuCVf4 = 0.0
    DuCVo1 = 0.0

    # 2. газогенератор / турбіна / камера
    mGg = 0.0
    mGv = 0.0
    mCh = 0.0

    # 3. насоси / моменти
    torqPmpOx = 0.0
    torqPmp1Fu = 0.0
    torqPmp2Fu = 0.0
    torqTrb = 0.0

    # 4. решта похідних
    Dmasbf = 0.0
    Dmasbo = 0.0
    Dpf1 = 0.0
    ...
    DvolVf1xCh = 0.0
    DvolVo1xGg = 0.0
    Domega = 0.0

    # write dy
    dy[Y_INDEX["xCVf1"]] = DxCVf1
    dy[Y_INDEX["xCVf2"]] = DxCVf2
    ...
    dy[Y_INDEX["omega"]] = Domega

    aux[AUX_INDEX["mGg"]] = mGg
    aux[AUX_INDEX["mGv"]] = mGv
    aux[AUX_INDEX["mCh"]] = mCh
    aux[AUX_INDEX["torqTrb"]] = torqTrb
