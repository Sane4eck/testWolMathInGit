# core/systems/sys_wolmath.py
import numpy as np
from numba import njit
from dataclasses import dataclass, fields, astuple

from core.physics import (
    ggaz,
    linear_law,
    f_valve,
    base_pump_eta_m,
    base_turb_eta,
    base_pump_h,
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
    "pf2_est", "pf3_est", "po2_est", "pTrbIn_est",
    "etaPmp1Fu", "etaPmp2Fu", "etaPmpOx", "etaTrb",
    "torqPmp1Fu", "torqPmp2Fu", "torqPmpOx", "torqTrb",
    "mGg", "mGv", "mCh",
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

    pf1_0: float = 1.0e5
    pf4_0: float = 1.0e5
    pf5_0: float = 1.0e5
    pf6_0: float = 1.0e5
    pf7_0: float = 1.0e5

    poB_0: float = 1.0e5
    po1_0: float = 1.0e5
    po3_0: float = 1.0e5
    po4_0: float = 1.0e5

    pGg_0: float = 1.0e5
    pGv_0: float = 1.0e5
    pCh_0: float = 1.0e5

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

    # globals
    g0: float = 1.0
    omega_to_rpm: float = 1.0
    pEnv: float = 1.0e5
    TEnv: float = 300.0
    JTrbPmp: float = 1.0

    # simple valve dynamics placeholders
    k_xcv: float = 0.0
    c_xcv: float = 0.0

    # fuel pump 1
    rhoPmp1FuNom: float = 800.0
    etaPmp1FuNom: float = 1.0
    pmp1_eta_c0: float = 1.0
    pmp1_eta_c1: float = 1.0
    pmp1_eta_c2: float = 2.0
    pmp1_h_c0: float = 0.0
    pmp1_h_c1: float = 0.0
    pmp1_h_c2: float = 0.0
    coefHfPmp1: float = 1.0

    # fuel pump 2
    rhoPmp2FuNom: float = 800.0
    etaPmp2FuNom: float = 1.0
    pmp2_eta_c0: float = 1.0
    pmp2_eta_c1: float = 1.0
    pmp2_eta_c2: float = 2.0
    pmp2_h_c0: float = 0.0
    pmp2_h_c1: float = 0.0
    pmp2_h_c2: float = 0.0
    coefHfPmp2: float = 1.0

    # oxidizer pump
    rhoPmpOxNom: float = 1100.0
    etaPmpOxNom: float = 1.0
    pmpOx_eta_c0: float = 1.0
    pmpOx_eta_c1: float = 1.0
    pmpOx_eta_c2: float = 2.0
    pmpOx_h_c0: float = 0.0
    pmpOx_h_c1: float = 0.0
    pmpOx_h_c2: float = 0.0
    coefHoPmp: float = 1.0

    # turbine / GG placeholders
    flagBurnGg: float = 0.0
    RGg_nom: float = 287.0
    TGg_nom: float = 300.0
    kGg_nom: float = 1.4
    DTrb: float = 1.0
    coefTorqTrb: float = 1.0
    muGg: float = 1.0
    fGg1: float = 1.0e-10
    muGv: float = 1.0
    fGv1: float = 1.0e-10

    def as_tuple(self):
        return astuple(self)

    @classmethod
    def from_excel(cls, path: str):
        # Поки лишаємо stub.
        # На наступному кроці сюди підключимо реальне читання Excel і таблиць.
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
    limit_vol = 1.0e9

    # openings
    if y[I_xCVf1] < 0.0: y[I_xCVf1] = 0.0
    if y[I_xCVf2] < 0.0: y[I_xCVf2] = 0.0
    if y[I_xCVf3] < 0.0: y[I_xCVf3] = 0.0
    if y[I_xCVf4] < 0.0: y[I_xCVf4] = 0.0
    if y[I_xCVo1] < 0.0: y[I_xCVo1] = 0.0

    # masses / pressures
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

    # flows
    for idx in (
        I_mfTx1, I_mf2xCh, I_mf3x4, I_mf4x6, I_mf6xIc,
        I_mf4x5, I_mf5xIg, I_mfBx7, I_mf7x6, I_mf7x5,
        I_moTx1, I_mo2x3, I_mo3xGg, I_mo2xGg, I_mo3x4,
        I_mo4xIc, I_mo4xIg, I_moBx4,
    ):
        if y[idx] > limit_m: y[idx] = limit_m
        if y[idx] < -limit_m: y[idx] = -limit_m

    # volumes
    if y[I_volVf1xCh] < 0.0: y[I_volVf1xCh] = 0.0
    if y[I_volVo1xGg] < 0.0: y[I_volVo1xGg] = 0.0
    if y[I_volVf1xCh] > limit_vol: y[I_volVf1xCh] = limit_vol
    if y[I_volVo1xGg] > limit_vol: y[I_volVo1xGg] = limit_vol

    # shaft speed
    if y[I_omega] < 1.0e-12:
        y[I_omega] = 1.0e-12


@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

    # params
    g0 = p[P_g0]
    omega_to_rpm = p[P_omega_to_rpm]
    JTrbPmp = p[P_JTrbPmp]

    k_xcv = p[P_k_xcv]
    c_xcv = p[P_c_xcv]

    rhoPmp1FuNom = p[P_rhoPmp1FuNom]
    rhoPmp2FuNom = p[P_rhoPmp2FuNom]
    rhoPmpOxNom = p[P_rhoPmpOxNom]

    etaPmp1FuNom = p[P_etaPmp1FuNom]
    etaPmp2FuNom = p[P_etaPmp2FuNom]
    etaPmpOxNom = p[P_etaPmpOxNom]

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

    pf1 = y[I_pf1]
    po1 = y[I_po1]
    pGg = y[I_pGg]
    pGv = y[I_pGv]

    mf2xCh = y[I_mf2xCh]
    mf3x4 = y[I_mf3x4]
    mo2x3 = y[I_mo2x3]
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

    # temporary valve dynamics
    DuCVf1 = -k_xcv * xCVf1 - c_xcv * uCVf1
    DuCVf2 = -k_xcv * xCVf2 - c_xcv * uCVf2
    DuCVf3 = -k_xcv * xCVf3 - c_xcv * uCVf3
    DuCVf4 = -k_xcv * xCVf4 - c_xcv * uCVf4
    DuCVo1 = -k_xcv * xCVo1 - c_xcv * uCVo1

    # -------------------------------------------------
    # 2. shaft / pumps / turbine
    # -------------------------------------------------
    eps = 1.0e-12
    rpm = omega * omega_to_rpm
    if rpm < eps:
        rpm = eps

    # simple pump head estimates
    HfPmp1 = base_pump_h(
        mf2xCh + mf3x4,
        rhoPmp1FuNom,
        rpm,
        p[P_pmp1_h_c0], p[P_pmp1_h_c1], p[P_pmp1_h_c2],
    )
    pf2_est = pf1 + HfPmp1 * p[P_coefHfPmp1]

    HfPmp2 = base_pump_h(
        mf3x4,
        rhoPmp2FuNom,
        rpm,
        p[P_pmp2_h_c0], p[P_pmp2_h_c1], p[P_pmp2_h_c2],
    )
    pf3_est = pf2_est + HfPmp2 * p[P_coefHfPmp2]

    HoPmp = base_pump_h(
        mo2x3,
        rhoPmpOxNom,
        rpm,
        p[P_pmpOx_h_c0], p[P_pmpOx_h_c1], p[P_pmpOx_h_c2],
    )
    po2_est = po1 + HoPmp * p[P_coefHoPmp]

    etaPmp1Fu = base_pump_eta_m(
        mf2xCh + mf3x4,
        rhoPmp1FuNom,
        rpm,
        p[P_pmp1_eta_c0], p[P_pmp1_eta_c1], p[P_pmp1_eta_c2],
    )
    if etaPmp1Fu < etaPmp1FuNom * 0.01:
        etaPmp1Fu = 1.0

    etaPmp2Fu = base_pump_eta_m(
        mf3x4,
        rhoPmp2FuNom,
        rpm,
        p[P_pmp2_eta_c0], p[P_pmp2_eta_c1], p[P_pmp2_eta_c2],
    )
    if etaPmp2Fu < etaPmp2FuNom * 0.01:
        etaPmp2Fu = 1.0

    etaPmpOx = base_pump_eta_m(
        mo2x3,
        rhoPmpOxNom,
        rpm,
        p[P_pmpOx_eta_c0], p[P_pmpOx_eta_c1], p[P_pmpOx_eta_c2],
    )
    if etaPmpOx < etaPmpOxNom * 0.01:
        etaPmpOx = 1.0

    torqPmp1Fu = 0.0
    if etaPmp1Fu > 1.0e-4 and rhoPmp1FuNom > eps:
        torqPmp1Fu = pf2_est * (mf2xCh + mf3x4) / (omega * rhoPmp1FuNom * etaPmp1Fu)

    torqPmp2Fu = 0.0
    if etaPmp2Fu > 1.0e-4 and rhoPmp2FuNom > eps:
        torqPmp2Fu = pf3_est * mf3x4 / (omega * rhoPmp2FuNom * etaPmp2Fu)

    torqPmpOx = 0.0
    if etaPmpOx > 1.0e-4 and rhoPmpOxNom > eps:
        torqPmpOx = po2_est * mo2x3 / (omega * rhoPmpOxNom * etaPmpOx)

    # turbine side
    pTrbIn_est = pGg
    mGg = 0.0
    mGv = 0.0
    mCh = 0.0
    etaTrb = 0.0
    torqTrb = 0.0

    if p[P_flagBurnGg] > 0.5:
        mGg = ggaz(
            pGg, pGv,
            p[P_muGg], p[P_fGg1],
            p[P_TGg_nom], p[P_kGg_nom], p[P_RGg_nom],
        )
        mGv = ggaz(
            pGv, p[I_pCh] if I_pCh < NY else pGv,
            p[P_muGv], p[P_fGv1],
            p[P_TGg_nom], p[P_kGg_nom], p[P_RGg_nom],
        )

        if pTrbIn_est > eps and pTrbIn_est >= pGv and p[P_kGg_nom] > 1.0:
            PITrb = pGv / pTrbIn_est
            LadTrb = (
                p[P_kGg_nom] / (p[P_kGg_nom] - 1.0)
                * p[P_RGg_nom] * p[P_TGg_nom]
                * (1.0 - PITrb ** ((p[P_kGg_nom] - 1.0) / p[P_kGg_nom]))
            )
            if LadTrb < 0.0:
                LadTrb = 0.0

            uTrb = np.pi * p[P_DTrb] * rpm / 60.0
            cadTrb = np.sqrt(2.0 * g0 * LadTrb) if LadTrb > 0.0 else 0.0

            if cadTrb > 1.0e-6:
                etaTrb = base_turb_eta(uTrb / cadTrb, p[P_trb_eta_c0], p[P_trb_eta_c1])
                if etaTrb < 0.0:
                    etaTrb = 0.0

            torqTrb = mGv * LadTrb * etaTrb * p[P_coefTorqTrb] / omega

    Domega = 0.0
    if JTrbPmp > eps:
        Domega = (torqTrb - torqPmpOx - torqPmp1Fu - torqPmp2Fu) * g0 / JTrbPmp

    # -------------------------------------------------
    # 3. remaining blocks are still to be transferred 1:1 from notebook
    # -------------------------------------------------

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
    aux[A_pf2_est] = pf2_est
    aux[A_pf3_est] = pf3_est
    aux[A_po2_est] = po2_est
    aux[A_pTrbIn_est] = pTrbIn_est

    aux[A_etaPmp1Fu] = etaPmp1Fu
    aux[A_etaPmp2Fu] = etaPmp2Fu
    aux[A_etaPmpOx] = etaPmpOx
    aux[A_etaTrb] = etaTrb

    aux[A_torqPmp1Fu] = torqPmp1Fu
    aux[A_torqPmp2Fu] = torqPmp2Fu
    aux[A_torqPmpOx] = torqPmpOx
    aux[A_torqTrb] = torqTrb

    aux[A_mGg] = mGg
    aux[A_mGv] = mGv
    aux[A_mCh] = mCh


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
