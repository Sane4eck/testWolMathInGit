# core/systems/sys_wolmath.py
import numpy as np
from numba import njit
from dataclasses import dataclass, fields, astuple

from core.physics import ggaz, f_valve, base_pump_eta_m, base_turb_eta, base_pump_h

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
    "pf2", "pf3", "po2", "pTrbIn",
    "mfChCool", "mf2xChFill", "mf2xChJet",
    "mo2xTrbOut", "mo3xTrbIn", "mo2xGgFill", "mo2xGgJet",
    "HfPmp1", "HfPmp2", "HoPmp",
    "mGg", "mGv", "mCh",
    "etaPmpOx", "etaPmp1Fu", "etaPmp2Fu", "etaTrb",
    "torqTrb", "torqPmpOx", "torqPmp1Fu", "torqPmp2Fu",
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

    # global
    pEnv: float = 1.0e5
    pfT: float = 1.0e5
    pfB: float = 1.0e5
    poT: float = 1.0e5
    rhoFu: float = 800.0
    rhoOx: float = 1100.0
    TEnv: float = 300.0
    g0: float = 1.0
    omega_to_rpm: float = 1.0

    # shaft
    JTrbPmp: float = 1.0

    # fuel line + pump heads
    rfTx1: float = 0.0
    ifTx1: float = 1.0
    Cfp1cav: float = 1.0
    mfChCoolNom: float = 0.0
    mf2xChNom: float = 1.0

    pmp1_h_c0: float = 0.0
    pmp1_h_c1: float = 0.0
    pmp1_h_c2: float = 0.0
    coefHfPmp1: float = 1.0

    pmp2_h_c0: float = 0.0
    pmp2_h_c1: float = 0.0
    pmp2_h_c2: float = 0.0
    coefHfPmp2: float = 1.0

    pmpOx_h_c0: float = 0.0
    pmpOx_h_c1: float = 0.0
    pmpOx_h_c2: float = 0.0
    coefHoPmp: float = 1.0

    # fuel branches
    rVf1Nom: float = 1.0
    rf2xVf1: float = 0.0
    rfVf1xCh: float = 0.0
    if2xCVf1: float = 1.0
    ifCVf1xCh: float = 1.0
    volVf1xChNom: float = 1.0

    rf3x4: float = 0.0
    if3x4: float = 1.0
    rf4xCVf2: float = 0.0
    rfCVf2x6: float = 0.0
    if4x6: float = 1.0
    rf6xVf3: float = 0.0
    if6xIc: float = 1.0
    rf7x6: float = 0.0
    if7x6: float = 1.0
    rf4xCVf1: float = 0.0
    rfCVf1x5: float = 0.0
    if4x5: float = 1.0
    rf5xVf2: float = 0.0
    if5xIg: float = 1.0
    rfBxCVf3: float = 0.0
    rfCVf3x7: float = 0.0
    ifBx7: float = 1.0
    rf7xCVf4: float = 0.0
    rfCVf4x5b: float = 0.0
    if7x5: float = 1.0
    Cf4: float = 1.0
    Cf5: float = 1.0
    Cf6: float = 1.0
    Cf7: float = 1.0

    # oxygen branches
    roTx1: float = 0.0
    ioTx1: float = 1.0
    mo2xTrbOutNom: float = 0.0
    mo2x3Nom: float = 1.0
    roBxVo2: float = 0.0
    roVo2x4: float = 0.0
    ioBx4: float = 1.0
    ro4xVo3: float = 0.0
    roVo3xIg: float = 0.0
    io4xIg: float = 1.0
    ro4xVo4: float = 0.0
    roVo4xIc: float = 0.0
    io4xIc: float = 1.0

    volVo1xGgNom: float = 1.0
    rVo1Nom: float = 1.0
    ro2xVo1: float = 0.0
    roVo1x3: float = 0.0
    ro3xGg: float = 0.0
    io2xVo1: float = 1.0
    ioVo1x3: float = 1.0
    io3xGg: float = 1.0

    ro3xCVo1: float = 0.0
    roCVo1x4: float = 0.0
    io3x4: float = 1.0
    io3xCVo1: float = 0.0
    ioCVo1x4: float = 0.0
    Co3: float = 1.0
    Co4: float = 1.0
    mo3xTrbInNom: float = 0.0
    mo3xGgNom: float = 1.0

    # schedules
    aVf1_0: float = 1.0e-10
    aVf1_1: float = 1.0e-10
    tVf11: float = 0.0
    tVf12: float = 0.0
    dtVf11: float = 0.0
    dtVf12: float = 0.0

    aVf2_0: float = 1.0e-10
    aVf2_1: float = 1.0e-10
    tVf21: float = 0.0
    tVf22: float = 0.0
    dtVf21: float = 0.0
    dtVf22: float = 0.0

    aVf3_0: float = 1.0e-10
    aVf3_1: float = 1.0e-10
    tVf31: float = 0.0
    tVf32: float = 0.0
    dtVf31: float = 0.0
    dtVf32: float = 0.0

    aVo1_0: float = 1.0e-10
    aVo1_1: float = 1.0e-10
    tVo11: float = 0.0
    tVo12: float = 0.0
    dtVo11: float = 0.0
    dtVo12: float = 0.0

    aVo2_0: float = 1.0e-10
    aVo2_1: float = 1.0e-10
    tVo21: float = 0.0
    tVo22: float = 0.0
    dtVo21: float = 0.0
    dtVo22: float = 0.0

    aVo3_0: float = 1.0e-10
    aVo3_1: float = 1.0e-10
    tVo31: float = 0.0
    tVo32: float = 0.0
    dtVo31: float = 0.0
    dtVo32: float = 0.0

    aVo4_0: float = 1.0e-10
    aVo4_1: float = 1.0e-10
    tVo41: float = 0.0
    tVo42: float = 0.0
    dtVo41: float = 0.0
    dtVo42: float = 0.0

    # CV01 geometry / dynamics
    xFlapMaxCV01: float = 1.0e-3
    fGapBodyFlapCV01: float = 1.0e-10
    fFlapExterCV01: float = 1.0e-6
    diamSeatCV01: float = 1.0e-3
    silaSpring0CV01: float = 0.0
    zSpringCV01: float = 0.0
    myuThrotCV01: float = 0.8
    fSeatCV01: float = 1.0e-6
    silaFrictStaticCV01: float = 0.0
    silaFrictKineticCV01: float = 0.0
    coefPropFrictViscousCV01: float = 0.0
    masMovingPartsCV01: float = 1.0
    coefSilaFlowCV01: float = 1.0

    coefRCVf1: float = 1.0
    coefRCVf2: float = 1.0
    coefRCVf3: float = 1.0
    coefRCVf4: float = 1.0
    coefRCVo1: float = 1.0

    # GG / chamber / turbine
    flagBurnGg: float = 0.0
    flagBurnGgSTOP: float = 0.0
    flagBurnCh: float = 0.0
    flagBurnChSTOP: float = 0.0
    dztaTrb: float = 1.0
    muGg: float = 1.0
    fGg1: float = 1.0e-10
    muGv: float = 1.0
    fGv1: float = 1.0e-10
    VGg: float = 1.0
    VGv: float = 1.0
    RGg_nom: float = 287.0
    TGg_nom: float = 300.0
    kGg_nom: float = 1.4
    muThCh: float = 1.0
    aThChcorect: float = 1.0e-10
    VCh: float = 1.0
    RCh_nom: float = 287.0
    TCh_nom: float = 300.0
    kCh_nom: float = 1.4
    DTrb: float = 1.0
    coefTorqTrb: float = 1.0

    # efficiencies / leaks
    rhoPmp1FuNom: float = 800.0
    rhoPmp2FuNom: float = 800.0
    rhoPmpOxNom: float = 1100.0
    etaPmp1FuNom: float = 1.0
    etaPmp2FuNom: float = 1.0
    etaPmpOxNom: float = 1.0
    mLeakFu1Nom: float = 0.0
    mLeakFu2Nom: float = 0.0
    mLeakOxNom: float = 0.0
    rpmNom: float = 1.0

    pmp1_eta_c0: float = 1.0
    pmp1_eta_c1: float = 1.0
    pmp1_eta_c2: float = 2.0
    pmp2_eta_c0: float = 1.0
    pmp2_eta_c1: float = 1.0
    pmp2_eta_c2: float = 2.0
    pmpOx_eta_c0: float = 1.0
    pmpOx_eta_c1: float = 1.0
    pmpOx_eta_c2: float = 2.0
    trb_eta_c0: float = 1.0
    trb_eta_c1: float = 0.0

    def as_tuple(self):
        return astuple(self)

    @classmethod
    def from_excel(cls, path: str):
        # тут поки без Excel-парсингу
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


@njit(cache=False)
def ftren(sila, rtr, f3, c3, dy):
    if -1.0e-5 < dy <= 1.0e-5:
        return rtr if sila >= 0.0 else -rtr
    return (f3 if dy >= 0.0 else -f3) + c3 * dy


@njit(cache=False)
def func_check_valve(
    p1, p2, u, x,
    xFlapMax, fGapBodyFlap, fFlapExter, diamSeat,
    silaSpring0, zSpring, myuThrot, fSeat,
    silaFrictStatic, silaFrictKinetic, coefPropFrictViscous,
    masMovingParts, coefSilaFlow
):
    x2 = 0.0
    if np.pi * diamSeat > 1.0e-30:
        x2 = fGapBodyFlap / (np.pi * diamSeat)
    x1 = 0.0
    myuThrot1 = 0.8
    myuThrot2 = 0.8

    den_px = myuThrot1 * myuThrot1 * (np.pi * diamSeat * x) ** 2 + myuThrot2 * myuThrot2 * fGapBodyFlap ** 2
    if den_px <= 1.0e-30:
        px = p2
    else:
        px = (
            myuThrot1 * myuThrot1 * (np.pi * diamSeat * x) ** 2 * p1
            + myuThrot2 * myuThrot2 * fGapBodyFlap ** 2 * p2
        ) / den_px

    fGap = np.pi * diamSeat * x
    if x <= x1:
        fSeatNew = fSeat
    elif x <= x2:
        if x2 > x1:
            fSeatNew = fSeat + ((fFlapExter - fSeat) / (x2 - x1)) * (x - x1)
        else:
            fSeatNew = fFlapExter
    else:
        fSeatNew = fFlapExter

    fSeatNew = fFlapExter
    fThrot = fGap if fGap < fGapBodyFlap else fGapBodyFlap
    silaSpring = silaSpring0 + zSpring * x
    silaFlow = 2.0 * myuThrot * fGap * (px - p2) * np.cos(69.0 * np.pi / 180.0) * coefSilaFlow
    sila = p1 * fSeat - p2 * fFlapExter + px * (fFlapExter - fSeatNew) - silaSpring
    silaFrict = ftren(sila, silaFrictStatic, silaFrictKinetic, coefPropFrictViscous, 0.0)
    d2x = (sila - silaFrict) / masMovingParts
    du = d2x
    dx = u
    if fThrot <= 0.0:
        fThrot = 1.0e-10
    return dx, du, fThrot


@njit(cache=False)
def func_check_valve_calc_resis(x, diamSeat, fGapBodyFlap, coefCorectivResis):
    dx = 0.0
    if x <= dx:
        fGap = 0.0
    else:
        fGap = np.pi * diamSeat * (x - dx)
    fThrot = fGap if fGap < fGapBodyFlap else fGapBodyFlap
    if fThrot <= 0.0:
        fThrot = 1.0e-10
    return coefCorectivResis / (fThrot * fThrot)


@njit(cache=False)
def area_from_schedule(t, a0, a1, t1, t2, dt1, dt2):
    a = f_valve(a0, a1, t1, t2, dt1, dt2, t)
    if a <= 0.0:
        a = 1.0e-10
    return a


@njit(cache=False)
def orifice_resistance_from_area(a):
    aa = a if a > 1.0e-10 else 1.0e-10
    return 1.0 / (2.0 * aa * aa)


@njit(cache=False)
def func_filling_rhs(
    val_is_open, vol, vol_nom, resisValNom, fValMax,
    resisPipe1, resisPipe2, inertPipe1, inertPipe2,
    p1, p2, m, rho, fVal
):
    ratioVol = 0.0
    if vol_nom > 1.0e-30:
        ratioVol = vol / vol_nom
    if ratioVol < 0.0:
        ratioVol = 0.0
    if ratioVol > 1.0:
        ratioVol = 1.0

    if fVal <= 0.0:
        fVal = 1.0e-10

    resisVal = resisValNom * (fValMax / fVal) ** 2
    FL1 = ratioVol ** 0.1
    resisPipe = resisPipe1 + resisVal + FL1 * resisPipe2 + 1.0e-20
    inertPipe = inertPipe1 + FL1 * inertPipe2 + 1.0e-20
    resisPipeNom = resisPipe1 + resisValNom + FL1 * resisPipe2

    mFill = m
    if val_is_open > 0.5:
        if resisPipe < 100.0 * resisPipeNom:
            Dm = (p1 - p2 - (resisPipe / rho) * abs(mFill) * mFill) / inertPipe
        else:
            Dm = 0.0
            term = (p1 - p2) / resisPipe * rho
            if term > 0.0:
                mFill = np.sqrt(term)
            else:
                mFill = 0.0
    else:
        Dm = 0.0
        if mFill < 0.0:
            mFill = 0.0

    if ratioVol >= 1.0:
        Dvol = 0.0
        mJet = mFill
    else:
        FL2 = 1.0 - 0.98 * ratioVol
        if FL2 < 0.0:
            FL2 = 0.0
        Dvol = (mFill * FL2) / rho
        mJet = mFill * (1.0 - FL2)

    return Dm, Dvol, mFill, mJet


def initial_y(p=None):
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

    if y[I_xCVf1] < 0.0: y[I_xCVf1] = 0.0
    if y[I_xCVf2] < 0.0: y[I_xCVf2] = 0.0
    if y[I_xCVf3] < 0.0: y[I_xCVf3] = 0.0
    if y[I_xCVf4] < 0.0: y[I_xCVf4] = 0.0
    if y[I_xCVo1] < 0.0: y[I_xCVo1] = 0.0

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

    for idx in (
        I_mfTx1, I_mf2xCh, I_mf3x4, I_mf4x6, I_mf6xIc, I_mf4x5, I_mf5xIg, I_mfBx7, I_mf7x6, I_mf7x5,
        I_moTx1, I_mo2x3, I_mo3xGg, I_mo2xGg, I_mo3x4, I_mo4xIc, I_mo4xIg, I_moBx4
    ):
        if y[idx] > limit_m: y[idx] = limit_m
        if y[idx] < -limit_m: y[idx] = -limit_m

    if y[I_volVf1xCh] < 0.0: y[I_volVf1xCh] = 0.0
    if y[I_volVo1xGg] < 0.0: y[I_volVo1xGg] = 0.0
    if y[I_volVf1xCh] > limit_vol: y[I_volVf1xCh] = limit_vol
    if y[I_volVo1xGg] > limit_vol: y[I_volVo1xGg] = limit_vol

    if y[I_omega] < 1.0e-12:
        y[I_omega] = 1.0e-12


@njit(cache=False)
def rhs(t, y, p, dy, aux):
    clamp_y_inplace(y)

    pEnv = p[P_pEnv]
    pfT = p[P_pfT]
    pfB = p[P_pfB]
    poT = p[P_poT]
    rhoFu = p[P_rhoFu]
    rhoOx = p[P_rhoOx]
    TEnv = p[P_TEnv]
    g0 = p[P_g0]
    omega_to_rpm = p[P_omega_to_rpm]
    JTrbPmp = p[P_JTrbPmp]

    xCVf1 = y[I_xCVf1]; xCVf2 = y[I_xCVf2]; xCVf3 = y[I_xCVf3]; xCVf4 = y[I_xCVf4]; xCVo1 = y[I_xCVo1]
    uCVf1 = y[I_uCVf1]; uCVf2 = y[I_uCVf2]; uCVf3 = y[I_uCVf3]; uCVf4 = y[I_uCVf4]; uCVo1 = y[I_uCVo1]
    pf1 = y[I_pf1]; pf4 = y[I_pf4]; pf5 = y[I_pf5]; pf6 = y[I_pf6]; pf7 = y[I_pf7]
    poB = y[I_poB]; po1 = y[I_po1]; po3 = y[I_po3]; po4 = y[I_po4]
    pGg = y[I_pGg]; pGv = y[I_pGv]; pCh = y[I_pCh]
    mfTx1 = y[I_mfTx1]; mf2xCh = y[I_mf2xCh]; mf3x4 = y[I_mf3x4]; mf4x6 = y[I_mf4x6]; mf6xIc = y[I_mf6xIc]
    mf4x5 = y[I_mf4x5]; mf5xIg = y[I_mf5xIg]; mfBx7 = y[I_mfBx7]; mf7x6 = y[I_mf7x6]; mf7x5 = y[I_mf7x5]
    moTx1 = y[I_moTx1]; mo2x3 = y[I_mo2x3]; mo3xGg = y[I_mo3xGg]; mo2xGg = y[I_mo2xGg]; mo3x4 = y[I_mo3x4]
    mo4xIc = y[I_mo4xIc]; mo4xIg = y[I_mo4xIg]; moBx4 = y[I_moBx4]
    volVf1xCh = y[I_volVf1xCh]; volVo1xGg = y[I_volVo1xGg]; omega = y[I_omega]

    for i in range(NY):
        dy[i] = 0.0
    for i in range(NAUX):
        aux[i] = 0.0

    aVf1 = area_from_schedule(t, p[P_aVf1_0], p[P_aVf1_1], p[P_tVf11], p[P_tVf12], p[P_dtVf11], p[P_dtVf12])
    aVf2 = area_from_schedule(t, p[P_aVf2_0], p[P_aVf2_1], p[P_tVf21], p[P_tVf22], p[P_dtVf21], p[P_dtVf22])
    aVf3 = area_from_schedule(t, p[P_aVf3_0], p[P_aVf3_1], p[P_tVf31], p[P_tVf32], p[P_dtVf31], p[P_dtVf32])
    aVo1 = area_from_schedule(t, p[P_aVo1_0], p[P_aVo1_1], p[P_tVo11], p[P_tVo12], p[P_dtVo11], p[P_dtVo12])
    aVo2 = area_from_schedule(t, p[P_aVo2_0], p[P_aVo2_1], p[P_tVo21], p[P_tVo22], p[P_dtVo21], p[P_dtVo22])
    aVo3 = area_from_schedule(t, p[P_aVo3_0], p[P_aVo3_1], p[P_tVo31], p[P_tVo32], p[P_dtVo31], p[P_dtVo32])
    aVo4 = area_from_schedule(t, p[P_aVo4_0], p[P_aVo4_1], p[P_tVo41], p[P_tVo42], p[P_dtVo41], p[P_dtVo42])

    rVf2 = orifice_resistance_from_area(aVf2)
    rVf3 = orifice_resistance_from_area(aVf3)
    rVo2 = orifice_resistance_from_area(aVo2)
    rVo3 = orifice_resistance_from_area(aVo3)
    rVo4 = orifice_resistance_from_area(aVo4)

    rpm = omega * omega_to_rpm
    if rpm < 1.0e-12:
        rpm = 1.0e-12

    # tank balances
    Dmasbf = -mfBx7
    Dmasbo = -moBx4
    DpoB = 0.0

    # fuel line
    DmfTx1 = (pfT - pf1 - (p[P_rfTx1] / rhoFu) * abs(mfTx1) * mfTx1) / p[P_ifTx1]
    mfChCool = p[P_mfChCoolNom] * (mf2xCh / p[P_mf2xChNom]) if abs(p[P_mf2xChNom]) > 1.0e-30 else 0.0
    Dpf1 = (mfTx1 - mf2xCh - mfChCool - mf3x4) / p[P_Cfp1cav]

    HfPmp1 = base_pump_h(mf2xCh + mfChCool + mf3x4, rhoFu, rpm, p[P_pmp1_h_c0], p[P_pmp1_h_c1], p[P_pmp1_h_c2])
    pf2 = pf1 + HfPmp1 * p[P_coefHfPmp1]

    Dmf2xCh, DvolVf1xCh, mf2xChFill, mf2xChJet = func_filling_rhs(
        1.0 if (p[P_tVf11] < t <= (p[P_tVf12] + p[P_dtVf12])) else 0.0,
        volVf1xCh, p[P_volVf1xChNom], p[P_rVf1Nom], aVf1,
        p[P_rf2xVf1], p[P_rfVf1xCh], p[P_if2xCVf1], p[P_ifCVf1xCh],
        pf2, pCh, mf2xCh, rhoFu, aVf1
    )

    HfPmp2 = base_pump_h(mf3x4, rhoFu, rpm, p[P_pmp2_h_c0], p[P_pmp2_h_c1], p[P_pmp2_h_c2])
    pf3 = pf2 + HfPmp2 * p[P_coefHfPmp2]

    Dmf3x4 = (pf3 - pf4 - (p[P_rf3x4] / rhoFu) * abs(mf3x4) * mf3x4) / p[P_if3x4]

    rCVf2 = func_check_valve_calc_resis(xCVf2, p[P_diamSeatCV01], p[P_fGapBodyFlapCV01], p[P_coefRCVf2])
    rf4x6 = p[P_rf4xCVf2] + rCVf2 + p[P_rfCVf2x6]
    Dmf4x6 = (pf4 - pf6 - (rf4x6 / rhoFu) * abs(mf4x6) * mf4x6) / p[P_if4x6]

    rf6xIc = p[P_rf6xVf3] + rVf3
    Dmf6xIc = (pf6 - pCh - (rf6xIc / rhoFu) * abs(mf6xIc) * mf6xIc) / p[P_if6xIc]

    Dmf7x6 = (pf7 - pf6 - (p[P_rf7x6] / rhoFu) * abs(mf7x6) * mf7x6) / p[P_if7x6]

    rCVf1 = func_check_valve_calc_resis(xCVf1, p[P_diamSeatCV01], p[P_fGapBodyFlapCV01], p[P_coefRCVf1])
    rf4x5 = p[P_rf4xCVf1] + rCVf1 + p[P_rfCVf1x5]
    Dmf4x5 = (pf4 - pf5 - (rf4x5 / rhoFu) * abs(mf4x5) * mf4x5) / p[P_if4x5]

    rf5xIg = p[P_rf5xVf2] + rVf2
    Dmf5xIg = (pf5 - pGg - (rf5xIg / rhoFu) * abs(mf5xIg) * mf5xIg) / p[P_if5xIg]

    rCVf3 = func_check_valve_calc_resis(xCVf3, p[P_diamSeatCV01], p[P_fGapBodyFlapCV01], p[P_coefRCVf3])
    rfBx7 = p[P_rfBxCVf3] + rCVf3 + p[P_rfCVf3x7]
    DmfBx7 = (pfB - pf7 - (rfBx7 / rhoFu) * abs(mfBx7) * mfBx7) / p[P_ifBx7]

    rf7x5b = p[P_rf7xCVf4] + p[P_rfCVf4x5b]
    Dmf7x5 = (pf7 - pf5 - (rf7x5b / rhoFu) * abs(mf7x5) * mf7x5) / p[P_if7x5]

    Dpf4 = (mf3x4 - mf4x6 - mf4x5) / p[P_Cf4]
    Dpf6 = (mf4x6 + mf7x6 - mf6xIc) / p[P_Cf6]
    Dpf5 = (mf4x5 + mf7x5 - mf5xIg) / p[P_Cf5]
    Dpf7 = (mfBx7 - mf7x6 - mf7x5) / p[P_Cf7]

    # fuel CV dynamics
    pCVf2In = pf4 - (p[P_rf4xCVf2] / rhoFu) * abs(mf4x6) * mf4x6
    pCVf2Out = pf6 + (p[P_rfCVf2x6] / rhoFu) * abs(mf4x6) * mf4x6
    DxCVf2, DuCVf2, _ = func_check_valve(
        pCVf2In, pCVf2Out, uCVf2, xCVf2,
        p[P_xFlapMaxCV01], p[P_fGapBodyFlapCV01], p[P_fFlapExterCV01], p[P_diamSeatCV01],
        p[P_silaSpring0CV01], p[P_zSpringCV01], p[P_myuThrotCV01], p[P_fSeatCV01],
        p[P_silaFrictStaticCV01], p[P_silaFrictKineticCV01], p[P_coefPropFrictViscousCV01],
        p[P_masMovingPartsCV01], p[P_coefSilaFlowCV01]
    )

    pCVf1In = pf4 - (p[P_rf4xCVf1] / rhoFu) * abs(mf4x5) * mf4x5
    pCVf1Out = pf5 + (p[P_rfCVf1x5] / rhoFu) * abs(mf4x5) * mf4x5
    DxCVf1, DuCVf1, _ = func_check_valve(
        pCVf1In, pCVf1Out, uCVf1, xCVf1,
        p[P_xFlapMaxCV01], p[P_fGapBodyFlapCV01], p[P_fFlapExterCV01], p[P_diamSeatCV01],
        p[P_silaSpring0CV01], p[P_zSpringCV01], p[P_myuThrotCV01], p[P_fSeatCV01],
        p[P_silaFrictStaticCV01], p[P_silaFrictKineticCV01], p[P_coefPropFrictViscousCV01],
        p[P_masMovingPartsCV01], p[P_coefSilaFlowCV01]
    )

    pCVf3In = pfB - (p[P_rfBxCVf3] / rhoFu) * abs(mfBx7) * mfBx7
    pCVf3Out = pf7 + (p[P_rfCVf3x7] / rhoFu) * abs(mfBx7) * mfBx7
    DxCVf3, DuCVf3, _ = func_check_valve(
        pCVf3In, pCVf3Out, uCVf3, xCVf3,
        p[P_xFlapMaxCV01], p[P_fGapBodyFlapCV01], p[P_fFlapExterCV01], p[P_diamSeatCV01],
        p[P_silaSpring0CV01], p[P_zSpringCV01], p[P_myuThrotCV01], p[P_fSeatCV01],
        p[P_silaFrictStaticCV01], p[P_silaFrictKineticCV01], p[P_coefPropFrictViscousCV01],
        p[P_masMovingPartsCV01], p[P_coefSilaFlowCV01]
    )

    pCVf4In = pf7 - (p[P_rf7xCVf4] / rhoFu) * abs(mf7x5) * mf7x5
    pCVf4Out = pf5 + (p[P_rfCVf4x5b] / rhoFu) * abs(mf7x5) * mf7x5
    DxCVf4, DuCVf4, _ = func_check_valve(
        pCVf4In, pCVf4Out, uCVf4, xCVf4,
        p[P_xFlapMaxCV01], p[P_fGapBodyFlapCV01], p[P_fFlapExterCV01], p[P_diamSeatCV01],
        p[P_silaSpring0CV01], p[P_zSpringCV01], p[P_myuThrotCV01], p[P_fSeatCV01],
        p[P_silaFrictStaticCV01], p[P_silaFrictKineticCV01], p[P_coefPropFrictViscousCV01],
        p[P_masMovingPartsCV01], p[P_coefSilaFlowCV01]
    )

    # oxygen line
    DmoTx1 = (poT - po1 - (p[P_roTx1] / rhoOx) * abs(moTx1) * moTx1) / p[P_ioTx1]
    mo2xTrbOut = p[P_mo2xTrbOutNom] * (mo2x3 / p[P_mo2x3Nom]) if abs(p[P_mo2x3Nom]) > 1.0e-30 else 0.0
    Dpo1 = (moTx1 - mo2x3 - mo2xTrbOut) / p[P_Cfp1cav]

    HoPmp = base_pump_h(mo2x3 + mo2xTrbOut, rhoOx, rpm, p[P_pmpOx_h_c0], p[P_pmpOx_h_c1], p[P_pmpOx_h_c2])
    po2 = po1 + HoPmp * p[P_coefHoPmp]

    roBx4 = p[P_roBxVo2] + rVo2 + p[P_roVo2x4]
    DmoBx4 = (poB - po4 - (roBx4 / rhoOx) * abs(moBx4) * moBx4) / p[P_ioBx4]

    ro4xIg = p[P_ro4xVo3] + rVo3 + p[P_roVo3xIg]
    Dmo4xIg = (po4 - pGg - (ro4xIg / rhoOx) * abs(mo4xIg) * mo4xIg) / p[P_io4xIg]

    ro4xIc = p[P_ro4xVo4] + rVo4 + p[P_roVo4xIc]
    Dmo4xIc = (po4 - pCh - (ro4xIc / rhoOx) * abs(mo4xIc) * mo4xIc) / p[P_io4xIc]

    DxCVo1 = uCVo1
    DuCVo1 = 0.0

    if volVo1xGg < p[P_volVo1xGgNom]:
        Dmo2xGg, DvolVo1xGg, mo2xGgFill, mo2xGgJet = func_filling_rhs(
            1.0 if (p[P_tVo11] < t <= (p[P_tVo12] + p[P_dtVo12])) else 0.0,
            volVo1xGg, p[P_volVo1xGgNom], p[P_rVo1Nom], aVo1,
            p[P_ro2xVo1], p[P_roVo1x3] + p[P_ro3xGg],
            p[P_io2xVo1], p[P_ioVo1x3] + p[P_io3xGg],
            po2, pGg, mo2xGg, rhoOx, aVo1
        )
        Dmo2x3 = Dmo2xGg
        Dmo3xGg = Dmo2xGg
        Dmo3x4 = 0.0
        Dpo3 = 0.0
        Dpo4 = (moBx4 - mo4xIg - mo4xIc) / p[P_Co4]
        mo2x3_eff = mo2xGgFill
        mo3xTrbIn = 0.0
    else:
        ro2x3 = p[P_ro2xVo1] + orifice_resistance_from_area(aVo1) + p[P_roVo1x3]
        Dmo2x3 = (po2 - po3 - (ro2x3 / rhoOx) * abs(mo2x3) * mo2x3) / p[P_io2xVo1]
        Dmo3xGg = (po3 - pGg - (p[P_ro3xGg] / rhoOx) * abs(mo3xGg) * mo3xGg) / p[P_io3xGg]
        rCVo1 = func_check_valve_calc_resis(xCVo1, p[P_diamSeatCV01], p[P_fGapBodyFlapCV01], p[P_coefRCVo1])
        ro3x4 = p[P_ro3xCVo1] + rCVo1 + p[P_roCVo1x4]
        Dmo3x4 = (po3 - po4 - (ro3x4 / rhoOx) * abs(mo3x4) * mo3x4) / p[P_io3x4]
        Dpo3 = (mo2x3 - mo3xGg - mo3x4) / p[P_Co3]
        Dpo4 = (moBx4 + mo3x4 - mo4xIg - mo4xIc) / p[P_Co4]
        Dmo2xGg = Dmo3xGg
        mo2xGgFill = mo3xGg
        mo2xGgJet = mo3xGg
        mo2x3_eff = mo2x3
        mo3xTrbIn = p[P_mo3xTrbInNom] * (mo3xGg / p[P_mo3xGgNom]) if abs(p[P_mo3xGgNom]) > 1.0e-30 else 0.0

        pCVo1In = po3 - (p[P_ro3xCVo1] / rhoOx) * abs(mo3x4) * mo3x4
        pCVo1Out = po4 + (p[P_roCVo1x4] / rhoOx) * abs(mo3x4) * mo3x4
        DxCVo1, DuCVo1, _ = func_check_valve(
            pCVo1In, pCVo1Out, uCVo1, xCVo1,
            p[P_xFlapMaxCV01], p[P_fGapBodyFlapCV01], p[P_fFlapExterCV01], p[P_diamSeatCV01],
            p[P_silaSpring0CV01], p[P_zSpringCV01], p[P_myuThrotCV01], p[P_fSeatCV01],
            p[P_silaFrictStaticCV01], p[P_silaFrictKineticCV01], p[P_coefPropFrictViscousCV01],
            p[P_masMovingPartsCV01], p[P_coefSilaFlowCV01]
        )

    # GG / chamber / TNA
    moGg = max(mo2xGgJet + mo4xIg, 0.0)
    mfGg = max(mf5xIg, 1.0e-10)
    moCh = max(moGg + mo4xIc + mo2xTrbOut + mo3xTrbIn, 0.0)
    mfCh = max(mf5xIg + mf6xIc + mfChCool + mf2xChJet, 1.0e-10)

    if p[P_flagBurnGg] > 0.5 and p[P_flagBurnGgSTOP] < 0.5:
        RGg = p[P_RGg_nom]
        TGg = p[P_TGg_nom]
        kGg = p[P_kGg_nom]
        TGv = max(TEnv, TGg * (pGv / pGg) ** ((kGg - 1.0) / kGg)) if pGg > 1.0e-12 else TEnv
        RGv = RGg
        kGv = kGg

        term = pGg * pGg - kGg * RGg * TGg * (moGg + mfGg + mo3xTrbIn) ** 2
        if term < 1.0:
            term = 1.0
        pTrbIn = p[P_dztaTrb] * p[P_dztaTrb] * np.sqrt(term)

        mGg = ggaz(pGg, pGv, p[P_muGg], p[P_fGg1], TGg, kGg, RGg)
        mGv = ggaz(pGv, pCh, p[P_muGv], p[P_fGv1], TGv, kGv, RGv)

        DpGg = ((RGg * TGg) / p[P_VGg]) * (moGg + mfGg + mo3xTrbIn - mGg)
        DpGv = ((RGv * TGv) / p[P_VGv]) * (mGg + mo2xTrbOut - mGv)
    else:
        RGg = 287.0
        TGg = TEnv
        kGg = 1.4
        pTrbIn = pGg
        mGg = 0.0
        mGv = 0.0
        DpGg = 0.0
        DpGv = 0.0

    mfLeakP1 = p[P_mLeakFu1Nom] * (rpm / p[P_rpmNom]) if p[P_rpmNom] > 1.0e-12 else 0.0
    etaPmp1Fu = base_pump_eta_m(mf2xCh + mfChCool + mf3x4, p[P_rhoPmp1FuNom], rpm, p[P_pmp1_eta_c0], p[P_pmp1_eta_c1], p[P_pmp1_eta_c2])
    if etaPmp1Fu > p[P_etaPmp1FuNom] * 0.01:
        mPmp1FuPow = mf2xCh + mfChCool + mf3x4
    else:
        mPmp1FuPow = mfLeakP1
        etaPmp1Fu = 1.0
    torqPmp1Fu = pf2 * mPmp1FuPow / (omega * p[P_rhoPmp1FuNom] * etaPmp1Fu) if etaPmp1Fu > 1.0e-4 else 0.0

    mfLeakP2 = p[P_mLeakFu2Nom] * (rpm / p[P_rpmNom]) if p[P_rpmNom] > 1.0e-12 else 0.0
    etaPmp2Fu = base_pump_eta_m(mf3x4, p[P_rhoPmp2FuNom], rpm, p[P_pmp2_eta_c0], p[P_pmp2_eta_c1], p[P_pmp2_eta_c2])
    if etaPmp2Fu > p[P_etaPmp2FuNom] * 0.1:
        mPmp2FuPow = mf3x4
    else:
        mPmp2FuPow = mfLeakP2
        etaPmp2Fu = 1.0
    torqPmp2Fu = pf3 * mPmp2FuPow / (omega * p[P_rhoPmp2FuNom] * etaPmp2Fu) if etaPmp2Fu > 1.0e-4 else 0.0

    moLeakP1 = p[P_mLeakOxNom] * (rpm / p[P_rpmNom]) if p[P_rpmNom] > 1.0e-12 else 0.0
    etaPmpOx = base_pump_eta_m(mo2x3_eff + mo2xTrbOut, p[P_rhoPmpOxNom], rpm, p[P_pmpOx_eta_c0], p[P_pmpOx_eta_c1], p[P_pmpOx_eta_c2])
    if etaPmpOx > p[P_etaPmpOxNom] * 0.01:
        mPmpOxPow = mo2x3_eff + mo2xTrbOut
    else:
        mPmpOxPow = moLeakP1
        etaPmpOx = 1.0
    torqPmpOx = po2 * mPmpOxPow / (omega * p[P_rhoPmpOxNom] * etaPmpOx) if etaPmpOx > 1.0e-4 else 0.0

    torqTrb = 0.0
    etaTrb = 0.0
    if p[P_flagBurnGg] > 0.5 and pTrbIn >= pGv and pTrbIn > 1.0e-12:
        PITrb = pGv / pTrbIn
        LadTrb = kGg / (kGg - 1.0) * RGg * TGg * (1.0 - PITrb ** ((kGg - 1.0) / kGg))
        uTrb = np.pi * p[P_DTrb] * rpm / 60.0
        cadTrb = np.sqrt(2.0 * g0 * LadTrb) if LadTrb > 0.0 else 0.0
        if cadTrb > 1.0e-6:
            etaTrb = base_turb_eta(uTrb / cadTrb, p[P_trb_eta_c0], p[P_trb_eta_c1])
            if etaTrb < 0.0:
                etaTrb = 0.0
        torqTrb = (mGv * LadTrb * etaTrb * p[P_coefTorqTrb]) / omega

    Domega = (torqTrb - torqPmpOx - torqPmp1Fu - torqPmp2Fu) * g0 / JTrbPmp if JTrbPmp > 1.0e-12 else 0.0

    if p[P_flagBurnCh] > 0.5 and p[P_flagBurnChSTOP] < 0.5:
        RCh = p[P_RCh_nom]
        TCh = p[P_TCh_nom]
        kCh = p[P_kCh_nom]
        mCh = ggaz(pCh, pEnv, p[P_muThCh], p[P_aThChcorect], TCh, kCh, RCh)
        DpCh = ((RCh * TCh) / p[P_VCh]) * (mGv + mo4xIc + mf6xIc + mf2xCh + mfChCool - mCh)
    else:
        mCh = 0.0
        DpCh = 0.0

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

    dy[I_masbf] = Dmasbf
    dy[I_masbo] = Dmasbo

    dy[I_pf1] = Dpf1
    dy[I_pf4] = Dpf4
    dy[I_pf5] = Dpf5
    dy[I_pf6] = Dpf6
    dy[I_pf7] = Dpf7

    dy[I_poB] = DpoB
    dy[I_po1] = Dpo1
    dy[I_po3] = Dpo3
    dy[I_po4] = Dpo4

    dy[I_pGg] = DpGg
    dy[I_pGv] = DpGv
    dy[I_pCh] = DpCh

    dy[I_mfTx1] = DmfTx1
    dy[I_mf2xCh] = Dmf2xCh
    dy[I_mf3x4] = Dmf3x4
    dy[I_mf4x6] = Dmf4x6
    dy[I_mf6xIc] = Dmf6xIc
    dy[I_mf4x5] = Dmf4x5
    dy[I_mf5xIg] = Dmf5xIg
    dy[I_mfBx7] = DmfBx7
    dy[I_mf7x6] = Dmf7x6
    dy[I_mf7x5] = Dmf7x5

    dy[I_moTx1] = DmoTx1
    dy[I_mo2x3] = Dmo2x3
    dy[I_mo3xGg] = Dmo3xGg
    dy[I_mo2xGg] = Dmo2xGg
    dy[I_mo3x4] = Dmo3x4
    dy[I_mo4xIc] = Dmo4xIc
    dy[I_mo4xIg] = Dmo4xIg
    dy[I_moBx4] = DmoBx4

    dy[I_volVf1xCh] = DvolVf1xCh
    dy[I_volVo1xGg] = DvolVo1xGg
    dy[I_omega] = Domega

    aux[A_pf2] = pf2
    aux[A_pf3] = pf3
    aux[A_po2] = po2
    aux[A_pTrbIn] = pTrbIn
    aux[A_mfChCool] = mfChCool
    aux[A_mf2xChFill] = mf2xChFill
    aux[A_mf2xChJet] = mf2xChJet
    aux[A_mo2xTrbOut] = mo2xTrbOut
    aux[A_mo3xTrbIn] = mo3xTrbIn
    aux[A_mo2xGgFill] = mo2xGgFill
    aux[A_mo2xGgJet] = mo2xGgJet
    aux[A_HfPmp1] = HfPmp1
    aux[A_HfPmp2] = HfPmp2
    aux[A_HoPmp] = HoPmp
    aux[A_mGg] = mGg
    aux[A_mGv] = mGv
    aux[A_mCh] = mCh
    aux[A_etaPmpOx] = etaPmpOx
    aux[A_etaPmp1Fu] = etaPmp1Fu
    aux[A_etaPmp2Fu] = etaPmp2Fu
    aux[A_etaTrb] = etaTrb
    aux[A_torqTrb] = torqTrb
    aux[A_torqPmpOx] = torqPmpOx
    aux[A_torqPmp1Fu] = torqPmp1Fu
    aux[A_torqPmp2Fu] = torqPmp2Fu


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
