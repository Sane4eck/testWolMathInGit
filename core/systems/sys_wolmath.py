# core/systems/sys_wolmath.py
import numpy as np
from numba import njit
from dataclasses import dataclass, fields, astuple

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


@dataclass(frozen=True)
class Params:
    # initial conditions
    xCVf1_0: float = 0.0
    xCVf2_0: float = 0.0
    xCVf3_0: float = 0.0
    xCVf4_0: float = 0.0
    xCVo1_0: float = 0.0

    uCVf1_0: float = 0.0
    uCVf2_0: float = 0.0
    uCVf3_0: float = 0.0
    uCVf4_0: float = 0.0
    uCVo1_0: float = 0.0

    masbf_0: float = 0.0
    masbo_0: float = 0.0

    pf1_0: float = 0.0
    pf4_0: float = 0.0
    pf5_0: float = 0.0
    pf6_0: float = 0.0
    pf7_0: float = 0.0

    poB_0: float = 0.0
    po1_0: float = 0.0
    po3_0: float = 0.0
    po4_0: float = 0.0

    pGg_0: float = 0.0
    pGv_0: float = 0.0
    pCh_0: float = 0.0

    mfTx1_0: float = 0.0
    mf2xCh_0: float = 0.0
    mf3x4_0: float = 0.0
    mf4x6_0: float = 0.0
    mf6xIc_0: float = 0.0
    mf4x5_0: float = 0.0
    mf5xIg_0: float = 0.0
    mfBx7_0: float = 0.0
    mf7x6_0: float = 0.0
    mf7x5_0: float = 0.0

    moTx1_0: float = 0.0
    mo2x3_0: float = 0.0
    mo3xGg_0: float = 0.0
    mo2xGg_0: float = 0.0
    mo3x4_0: float = 0.0
    mo4xIc_0: float = 0.0
    mo4xIg_0: float = 0.0
    moBx4_0: float = 0.0

    volVf1xCh_0: float = 0.0
    volVo1xGg_0: float = 0.0
    omega_0: float = 1.0

    # shaft / valve dynamics
    JTrbPmp: float = 1.0
    g0: float = 1.0
    k_xcv: float = 0.0
    k_ucv: float = 0.0

    # temporary algebraic refs for the shaft block
    pf2_ref: float = 0.0
    pf3_ref: float = 0.0
    po2_ref: float = 0.0
    pTrbIn_ref: float = 0.0

    mfChCool_ref: float = 0.0
    mo2xTrbOut_ref: float = 0.0
    mGv_ref: float = 0.0
    flagBurnGg_ref: float = 0.0

    # nominal densities
    rhoPmp1FuNom: float = 1.0
    rhoPmp2FuNom: float = 1.0
    rhoPmpOxNom: float = 1.0

    # nominal efficiencies
    etaPmp1FuNom: float = 1.0
    etaPmp2FuNom: float = 1.0
    etaPmpOxNom: float = 1.0

    # nominal leakage
    mLeakFu1Nom: float = 0.0
    mLeakFu2Nom: float = 0.0
    mLeakOxNom: float = 0.0

    # pump eta curves
    pmp1_eta_c0: float = 1.0
    pmp1_eta_c1: float = 1.0
    pmp1_eta_c2: float = 2.0

    pmp2_eta_c0: float = 1.0
    pmp2_eta_c1: float = 1.0
    pmp2_eta_c2: float = 2.0

    pmpOx_eta_c0: float = 1.0
    pmpOx_eta_c1: float = 1.0
    pmpOx_eta_c2: float = 2.0

    # turbine
    trb_eta_c0: float = 1.0
    trb_eta_c1: float = 0.0
    kGg: float = 1.4
    RGg: float = 287.0
    TGg: float = 300.0
    DTrb: float = 1.0
    coefTorqTrb: float = 1.0

    def as_tuple(self):
        return np.asarray(astuple(self), dtype=np.float64)

    @classmethod
    def from_excel(cls, path: str):
        # На цьому етапі лишаємо заглушку.
        # Далі сюди підключимо реальне читання Excel.
        return cls()


PARAM_ORDER = tuple(f.name for f in fields(Params))


def _declare_indices():
    for i, name in enumerate(Y_ORDER):
        globals()[f"I_{name}"] = i
    for i, name in enumerate(AUX_ORDER):
        globals()[f"A_{name}"] = i
    for i, name in enumerate(PARAM_ORDER):
        globals()[f"P_{name}"] = i


_declare_indices()
del _declare_indices


def initial_y(p: Params | None = None) -> np.ndarray:
    if p is None:
        p = Params()

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
def clamp_y_inplace(y):
    limit_m = 1.0e8
    limit_p = 1.0e9
    limit_vol = 1.0e9

    # valve lift
    if y[I_xCVf1] < 0.0: y[I_xCVf1] = 0.0
    if y[I_xCVf2] < 0.0: y[I_xCVf2] = 0.0
    if y[I_xCVf3] < 0.0: y[I_xCVf3] = 0.0
    if y[I_xCVf4] < 0.0: y[I_xCVf4] = 0.0
    if y[I_xCVo1] < 0.0: y[I_xCVo1] = 0.0

    # masses
    if y[I_masbf] < 0.0: y[I_masbf] = 0.0
    if y[I_masbo] < 0.0: y[I_masbo] = 0.0

    # pressures
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

    # flow limits
    if y[I_mfTx1] > limit_m: y[I_mfTx1] = limit_m
    if y[I_mfTx1] < -limit_m: y[I_mfTx1] = -limit_m
    if y[I_mf2xCh] > limit_m: y[I_mf2xCh] = limit_m
    if y[I_mf2xCh] < -limit_m: y[I_mf2xCh] = -limit_m
    if y[I_mf3x4] > limit_m: y[I_mf3x4] = limit_m
    if y[I_mf3x4] < -limit_m: y[I_mf3x4] = -limit_m
    if y[I_mf4x6] > limit_m: y[I_mf4x6] = limit_m
    if y[I_mf4x6] < -limit_m: y[I_mf4x6] = -limit_m
    if y[I_mf6xIc] > limit_m: y[I_mf6xIc] = limit_m
    if y[I_mf6xIc] < -limit_m: y[I_mf6xIc] = -limit_m
    if y[I_mf4x5] > limit_m: y[I_mf4x5] = limit_m
    if y[I_mf4x5] < -limit_m: y[I_mf4x5] = -limit_m
    if y[I_mf5xIg] > limit_m: y[I_mf5xIg] = limit_m
    if y[I_mf5xIg] < -limit_m: y[I_mf5xIg] = -limit_m
    if y[I_mfBx7] > limit_m: y[I_mfBx7] = limit_m
    if y[I_mfBx7] < -limit_m: y[I_mfBx7] = -limit_m
    if y[I_mf7x6] > limit_m: y[I_mf7x6] = limit_m
    if y[I_mf7x6] < -limit_m: y[I_mf7x6] = -limit_m
    if y[I_mf7x5] > limit_m: y[I_mf7x5] = limit_m
    if y[I_mf7x5] < -limit_m: y[I_mf7x5] = -limit_m

    if y[I_moTx1] > limit_m: y[I_moTx1] = limit_m
    if y[I_moTx1] < -limit_m: y[I_moTx1] = -limit_m
    if y[I_mo2x3] > limit_m: y[I_mo2x3] = limit_m
    if y[I_mo2x3] < -limit_m: y[I_mo2x3] = -limit_m
    if y[I_mo3xGg] > limit_m: y[I_mo3xGg] = limit_m
    if y[I_mo3xGg] < -limit_m: y[I_mo3xGg] = -limit_m
    if y[I_mo2xGg] > limit_m: y[I_mo2xGg] = limit_m
    if y[I_mo2xGg] < -limit_m: y[I_mo2xGg] = -limit_m
    if y[I_mo3x4] > limit_m: y[I_mo3x4] = limit_m
    if y[I_mo3x4] < -limit_m: y[I_mo3x4] = -limit_m
    if y[I_mo4xIc] > limit_m: y[I_mo4xIc] = limit_m
    if y[I_mo4xIc] < -limit_m: y[I_mo4xIc] = -limit_m
    if y[I_mo4xIg] > limit_m: y[I_mo4xIg] = limit_m
    if y[I_mo4xIg] < -limit_m: y[I_mo4xIg] = -limit_m
    if y[I_moBx4] > limit_m: y[I_moBx4] = limit_m
    if y[I_moBx4] < -limit_m: y[I_moBx4] = -limit_m

    # volumes
    if y[I_volVf1xCh] < 0.0: y[I_volVf1xCh] = 0.0
    if y[I_volVo1xGg] < 0.0: y[I_volVo1xGg] = 0.0
    if y[I_volVf1xCh] > limit_vol: y[I_volVf1xCh] = limit_vol
    if y[I_volVo1xGg] > limit_vol: y[I_volVo1xGg] = limit_vol

    # shaft speed
    if y[I_omega] < 1.0e-12: y[I_omega] = 1.0e-12


@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

    # params
    JTrbPmp = p[P_JTrbPmp]
    g0 = p[P_g0]
    k_xcv = p[P_k_xcv]
    k_ucv = p[P_k_ucv]

    pf2 = p[P_pf2_ref]
    pf3 = p[P_pf3_ref]
    po2 = p[P_po2_ref]
    pTrbIn = p[P_pTrbIn_ref]

    mfChCool = p[P_mfChCool_ref]
    mo2xTrbOut = p[P_mo2xTrbOut_ref]
    mGv_ref = p[P_mGv_ref]
    flagBurnGg = p[P_flagBurnGg_ref]

    rhoPmp1FuNom = p[P_rhoPmp1FuNom]
    rhoPmp2FuNom = p[P_rhoPmp2FuNom]
    rhoPmpOxNom = p[P_rhoPmpOxNom]

    etaPmp1FuNom = p[P_etaPmp1FuNom]
    etaPmp2FuNom = p[P_etaPmp2FuNom]
    etaPmpOxNom = p[P_etaPmpOxNom]

    mLeakFu1Nom = p[P_mLeakFu1Nom]
    mLeakFu2Nom = p[P_mLeakFu2Nom]
    mLeakOxNom = p[P_mLeakOxNom]

    pmp1_eta_c0 = p[P_pmp1_eta_c0]
    pmp1_eta_c1 = p[P_pmp1_eta_c1]
    pmp1_eta_c2 = p[P_pmp1_eta_c2]

    pmp2_eta_c0 = p[P_pmp2_eta_c0]
    pmp2_eta_c1 = p[P_pmp2_eta_c1]
    pmp2_eta_c2 = p[P_pmp2_eta_c2]

    pmpOx_eta_c0 = p[P_pmpOx_eta_c0]
    pmpOx_eta_c1 = p[P_pmpOx_eta_c1]
    pmpOx_eta_c2 = p[P_pmpOx_eta_c2]

    trb_eta_c0 = p[P_trb_eta_c0]
    trb_eta_c1 = p[P_trb_eta_c1]
    kGg = p[P_kGg]
    RGg = p[P_RGg]
    TGg = p[P_TGg]
    DTrb = p[P_DTrb]
    coefTorqTrb = p[P_coefTorqTrb]

    # unpack y
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

    mf2xCh = y[I_mf2xCh]
    mf3x4 = y[I_mf3x4]
    mo2x3 = y[I_mo2x3]
    pGv = y[I_pGv]
    omega = y[I_omega]

    # reset outputs
    for i in range(NY):
        dy[i] = 0.0
    for i in range(NAUX):
        aux[i] = 0.0

    # -------------------------------------------------
    # 1. valve kinematics
    # -------------------------------------------------
    DxCVf1 = uCVf1
    DxCVf2 = uCVf2
    DxCVf3 = uCVf3
    DxCVf4 = uCVf4
    DxCVo1 = uCVo1

    # temporary simplified valve dynamics
    DuCVf1 = -k_xcv * xCVf1 - k_ucv * uCVf1
    DuCVf2 = -k_xcv * xCVf2 - k_ucv * uCVf2
    DuCVf3 = -k_xcv * xCVf3 - k_ucv * uCVf3
    DuCVf4 = -k_xcv * xCVf4 - k_ucv * uCVf4
    DuCVo1 = -k_xcv * xCVo1 - k_ucv * uCVo1

    # -------------------------------------------------
    # 2. shaft / pumps / turbine
    # -------------------------------------------------
    eps = 1.0e-12
    om = omega if omega > eps else eps

    # fuel pump 1
    mfLeakP1 = mLeakFu1Nom * om
    etaPmp1Fu = base_pump_eta_m(
        mf2xCh + mfChCool + mf3x4,
        rhoPmp1FuNom,
        om,
        pmp1_eta_c0, pmp1_eta_c1, pmp1_eta_c2,
    )
    if etaPmp1Fu > etaPmp1FuNom * 0.01:
        mPmp1FuPow = mf2xCh + mfChCool + mf3x4
    else:
        mPmp1FuPow = mfLeakP1
        etaPmp1Fu = 1.0

    if etaPmp1Fu > 1.0e-4 and rhoPmp1FuNom > eps:
        torqPmp1Fu = pf2 * mPmp1FuPow / (om * rhoPmp1FuNom * etaPmp1Fu)
    else:
        torqPmp1Fu = 0.0

    # fuel pump 2
    mfLeakP2 = mLeakFu2Nom * om
    etaPmp2Fu = base_pump_eta_m(
        mf3x4,
        rhoPmp2FuNom,
        om,
        pmp2_eta_c0, pmp2_eta_c1, pmp2_eta_c2,
    )
    if etaPmp2Fu > etaPmp2FuNom * 0.1:
        mPmp2FuPow = mf3x4
    else:
        mPmp2FuPow = mfLeakP2
        etaPmp2Fu = 1.0

    if etaPmp2Fu > 1.0e-4 and rhoPmp2FuNom > eps:
        torqPmp2Fu = pf3 * mPmp2FuPow / (om * rhoPmp2FuNom * etaPmp2Fu)
    else:
        torqPmp2Fu = 0.0

    # oxidizer pump
    moLeakP1 = mLeakOxNom * om
    etaPmpOx = base_pump_eta_m(
        mo2x3 + mo2xTrbOut,
        rhoPmpOxNom,
        om,
        pmpOx_eta_c0, pmpOx_eta_c1, pmpOx_eta_c2,
    )
    if etaPmpOx > etaPmpOxNom * 0.01:
        mPmpOxPow = mo2x3 + mo2xTrbOut
    else:
        mPmpOxPow = moLeakP1
        etaPmpOx = 1.0

    if etaPmpOx > 1.0e-4 and rhoPmpOxNom > eps:
        torqPmpOx = po2 * mPmpOxPow / (om * rhoPmpOxNom * etaPmpOx)
    else:
        torqPmpOx = 0.0

    # turbine
    mGv = mGv_ref
    torqTrb = 0.0
    etaTrb = 0.0
    LadTrb = 0.0

    if flagBurnGg > 0.5 and pTrbIn > eps and pTrbIn >= pGv and kGg > 1.0:
        PITrb = pGv / pTrbIn
        LadTrb = kGg / (kGg - 1.0) * RGg * TGg * (1.0 - PITrb ** ((kGg - 1.0) / kGg))
        if LadTrb < 0.0:
            LadTrb = 0.0

        uTrb = np.pi * DTrb * om / 60.0
        cadTrb = np.sqrt(2.0 * g0 * LadTrb) if LadTrb > 0.0 else 0.0

        if cadTrb > 1.0e-6:
            etaTrb = base_turb_eta(uTrb / cadTrb, trb_eta_c0, trb_eta_c1)
            if etaTrb < 0.0:
                etaTrb = 0.0

        torqTrb = mGv * LadTrb * etaTrb * coefTorqTrb / om

    if JTrbPmp > eps:
        Domega = (torqTrb - torqPmpOx - torqPmp1Fu - torqPmp2Fu) * g0 / JTrbPmp
    else:
        Domega = 0.0

    # -------------------------------------------------
    # 3. not yet transferred blocks
    # -------------------------------------------------
    # masses, pressures, flow rates, chamber/gas-generator filling
    # stay zero on this stage

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

    # write aux
    aux[A_mGv] = mGv
    aux[A_etaPmpOx] = etaPmpOx
    aux[A_etaPmp1Fu] = etaPmp1Fu
    aux[A_etaPmp2Fu] = etaPmp2Fu
    aux[A_etaTrb] = etaTrb
    aux[A_torqTrb] = torqTrb
    aux[A_torqPmpOx] = torqPmpOx
    aux[A_torqPmp1Fu] = torqPmp1Fu
    aux[A_torqPmp2Fu] = torqPmp2Fu
    aux[A_mfChCool] = mfChCool


__all__ = [
    "Y_ORDER",
    "AUX_ORDER",
    "NY",
    "NAUX",
    "PARAM_ORDER",
    "Params",
    "initial_y",
    "clamp_y_inplace",
    "rhs",
]
