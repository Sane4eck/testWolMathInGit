from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from io_utils import load_general_params, load_pressure_vapor_tables, load_valve_cyclogram, si_or_sgs
from rk4 import rk4_step

EPS = 1e-20
OMEGA_COEF = 2.0 * np.pi / 60.0


STATE_NAMES = [
    "xCVf1", "xCVf2", "xCVf3", "xCVf4", "xCVo1",
    "uCVf1", "uCVf2", "uCVf3", "uCVf4", "uCVo1",
    "masbf", "masbo",
    "pf1", "pf4", "pf5", "pf6", "pf7", "poB", "po1", "po3", "po4", "pGg", "pGv", "pCh",
    "mfTx1", "mf2xCh", "mf3x4", "mf4x6", "mf6xIc", "mf4x5", "mf5xIg", "mfBx7", "mf7x6", "mf7x5",
    "moTx1", "mo2x3", "mo3xGg", "mo2xGg", "mo3x4", "mo4xIc", "mo4xIg", "moBx4",
    "volVf1xCh", "volVo1xGg",
    "omega",
]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}


@dataclass(slots=True)
class ValveGeometry:
    f_gap_body_flap: float
    f_seat: float
    mas_moving_parts: float
    f_flap_exter: float


@dataclass(slots=True)
class RuntimeFlags:
    flag_fill_vf1xch: int = 0
    flag_val_open_vf1: int = 0
    flag_fill_vo1xgg: int = 0
    flag_val_open_vo1: int = 0
    flag_tau_gg: int = 0
    flag_tau_gg_ox: int = 0
    flag_tau_gg_fu: int = 0
    flag_burn_gg: int = 0
    flag_burn_gg_stop: int = 0
    flag_ad_tau_ch: int = 0
    flag_tau_ch_fu: int = 0
    flag_tau_ch_fu_gen: int = 0
    flag_burn_ch: int = 0
    flag_burn_ch_stop: int = 0
    t_burn_gg: float = 100.0
    t_burn_ch: float = 100.0
    time_ad_tau_gg: float = 100.0
    time_ad_tau_ch: float = 100.0


@dataclass(slots=True)
class SimulationResult:
    time: np.ndarray
    y: np.ndarray
    state_names: list[str]
    history: dict[str, np.ndarray]


@dataclass(slots=True)
class LaunchModel:
    excel_path: str | Path
    unit_system: str = "SI"
    p: dict[str, float] = field(init=False)
    units: object = field(init=False)
    valves: dict[str, list[float]] = field(init=False)
    interp_pressure_vapor_ox: interp1d = field(init=False)
    interp_pressure_vapor_fu: interp1d = field(init=False)
    flags: RuntimeFlags = field(default_factory=RuntimeFlags)

    def __post_init__(self) -> None:
        self.excel_path = Path(self.excel_path)
        self.units = si_or_sgs(self.unit_system)
        self.p = load_general_params(self.excel_path, self.units)
        self.valves = load_valve_cyclogram(self.excel_path, self.units)
        self.interp_pressure_vapor_ox, self.interp_pressure_vapor_fu = load_pressure_vapor_tables(self.excel_path, self.units)
        self._install_valve_constants()
        self._install_defaults()

    def _install_defaults(self) -> None:
        self.p.setdefault("pEnv", 101325.0 * self.units.delta_p)
        self.p.setdefault("TEnv", 293.15)
        self.p.setdefault("REnv", 287.0 * self.units.delta_r)
        self.p.setdefault("countPrint", 1000.0)
        self.p.setdefault("endTime", 1.0)
        self.p.setdefault("dt", 1e-7)
        self.p.setdefault("coefTorqTrb", 1.0)
        self.p.setdefault("JTrbPmp", 1.0)
        self.p.setdefault("xFlapMax", 1e-3)
        self.p.setdefault("coefSilaFlow", 1.0)
        self.p.setdefault("silaFrictStaticCV01", 0.0)
        self.p.setdefault("silaFrictKineticCV01", 0.0)
        self.p.setdefault("coefPropFrictViscousCV01", 0.0)
        self.p.setdefault("myuThrotCV01", 0.8)
        self.p.setdefault("silaSpring0CV01", 0.0)
        self.p.setdefault("zSpringCV01", 0.0)
        self.p.setdefault("timeDelayIgnGg", 0.0)
        self.p.setdefault("timeDelayIgnCh", 0.0)
        self.p.setdefault("mGgOxMin", 0.0)
        self.p.setdefault("mGgFuMin", 0.0)
        self.p.setdefault("mChFuGenMin", 0.0)
        self.p.setdefault("mChIngFuMin", 0.0)
        self.p.setdefault("dztaTrb", 1.0)
        self.p.setdefault("muGg", 1.0)
        self.p.setdefault("muGv", 1.0)
        self.p.setdefault("fGg1", 1.0)
        self.p.setdefault("fGv1", 1.0)
        self.p.setdefault("VGg", 1.0)
        self.p.setdefault("VGv", 1.0)
        self.p.setdefault("VCh", 1.0)
        self.p.setdefault("DTrb", 1.0)
        self.p.setdefault("muThCh", 1.0)
        self.p.setdefault("aThChcorect", 1.0)
        self.p.setdefault("rpmNom", 1.0)
        self.p.setdefault("rhoPmp1FuNom", 1.0)
        self.p.setdefault("rhoPmp2FuNom", 1.0)
        self.p.setdefault("rhoPmpOxNom", 1.0)
        self.p.setdefault("etaPmp1FuNom", 1.0)
        self.p.setdefault("etaPmp2FuNom", 1.0)
        self.p.setdefault("etaPmpOxNom", 1.0)
        self.p.setdefault("coefHfPmp1", 1.0)
        self.p.setdefault("coefHfPmp2", 1.0)
        self.p.setdefault("coefHoPmp", 1.0)
        self.p.setdefault("mo2xTrbOutNom", 0.0)
        self.p.setdefault("mo2x3Nom", 1.0)
        self.p.setdefault("mo3xTrbInNom", 0.0)
        self.p.setdefault("mo3xGgNom", 1.0)
        self.p.setdefault("mfChCoolNom", 0.0)
        self.p.setdefault("mf2xChNom", 1.0)
        self.p.setdefault("pfT", self.p["pEnv"])
        self.p.setdefault("poT", self.p["pEnv"])
        self.p.setdefault("pfB", self.p["pEnv"])
        self.p.setdefault("poBNom", self.p["pEnv"])
        self.p.setdefault("pfBNom", self.p["pEnv"])
        self.p.setdefault("ToBNom", self.p["TEnv"])
        self.p.setdefault("ToB", self.p["TEnv"])
        self.p.setdefault("ToB0", self.p["TEnv"])
        self.p.setdefault("rhofB", 1.0)
        self.p.setdefault("rhooB", 1.0)
        self.p.setdefault("volfBNom", 1.0)
        self.p.setdefault("voloBNom", 1.0)
        self.p.setdefault("volVf1xChNom", 1.0)
        self.p.setdefault("volVo1xGgNom", 1.0)
        self.p.setdefault("kGgNom", 1.4)
        self.p.setdefault("speedSoundJETA", 1.0)
        self.p.setdefault("areaPipeDY4", 1.0)
        self.p.setdefault("areaPipeDY6", 1.0)
        self.p.setdefault("epsf7", 0.0)
        self.p.setdefault("epsf5", 0.0)
        self.p.setdefault("epsf4", 0.0)
        self.p.setdefault("epsf6", 0.0)
        self.p.setdefault("pMax", self.p["pEnv"] * 100.0)

    def _install_valve_constants(self) -> None:
        for name, vals in self.valves.items():
            self.p[f"t{name}1"] = vals[0]
            self.p[f"dt{name}1"] = vals[1]
            self.p[f"t{name}2"] = vals[2]
            self.p[f"dt{name}2"] = vals[3]
            self.p[f"a{name}Max"] = vals[4]
            self.p[f"a{name}Min"] = vals[5]
            self.p[f"mu{name}"] = vals[6]

    @staticmethod
    def _linear_law(t: float, val_zero: float, val_nom: float, t1: float, t2: float) -> float:
        if t <= t1:
            return val_zero
        if t <= t2:
            return val_zero + (val_nom - val_zero) / (t2 - t1) * (t - t1)
        return val_nom

    @staticmethod
    def _fkl(f1: float, f2: float, t1: float, t2: float, dt1: float, dt2: float, t: float) -> float:
        if t < t1 or t >= t2 + dt2:
            return f1
        if t1 <= t < t1 + dt1:
            return f1 + (f2 - f1) * (t - t1) / dt1
        if t1 + dt1 <= t < t2:
            return f2
        return f2 - (f2 - f1) * (t - t2) / dt2

    def int_a_vo1(self, t: float) -> float:
        return self._fkl(self.p["aVo1Min"], self.p["aVo1Max"], self.p["tVo11"], self.p["tVo12"], self.p["dtVo11"], self.p["dtVo12"], t)

    def int_a_vo2(self, t: float) -> float:
        return self._fkl(self.p["aVo2Min"], self.p["aVo2Max"], self.p["tVo21"], self.p["tVo22"], self.p["dtVo21"], self.p["dtVo22"], t)

    def int_a_vo3(self, t: float) -> float:
        return self._fkl(self.p["aVo3Min"], self.p["aVo3Max"], self.p["tVo31"], self.p["tVo32"], self.p["dtVo31"], self.p["dtVo32"], t)

    def int_a_vo4(self, t: float) -> float:
        return self._fkl(self.p["aVo4Min"], self.p["aVo4Max"], self.p["tVo41"], self.p["tVo42"], self.p["dtVo41"], self.p["dtVo42"], t)

    def int_a_vf1(self, t: float) -> float:
        return self._fkl(self.p["aVf1Min"], self.p["aVf1Max"], self.p["tVf11"], self.p["tVf12"], self.p["dtVf11"], self.p["dtVf12"], t)

    def int_a_vf2(self, t: float) -> float:
        return self._fkl(self.p["aVf2Min"], self.p["aVf2Max"], self.p["tVf21"], self.p["tVf22"], self.p["dtVf21"], self.p["dtVf22"], t)

    def int_a_vf3(self, t: float) -> float:
        return self._fkl(self.p["aVf3Min"], self.p["aVf3Max"], self.p["tVf31"], self.p["tVf32"], self.p["dtVf31"], self.p["dtVf32"], t)

    @staticmethod
    def func_base_pump_eta(m_pump_out: float, rho: float, rpm: float, coef: np.ndarray) -> float:
        qotn = m_pump_out / (rho * (10.0 ** 2) ** -3 * rpm)
        den = coef[1] ** 2 - qotn * (2.0 * coef[1] - coef[2])
        return coef[0] * ((coef[2] - qotn) * qotn / den)

    @staticmethod
    def func_base_turb_eta(u_to_c: float, coef: np.ndarray) -> float:
        return coef[0] * u_to_c + coef[1] * u_to_c ** 2

    @staticmethod
    def func_base_pump_h(m_pump_out: float, rho: float, rpm: float, coef: np.ndarray) -> float:
        return coef[0] * rho * rpm ** 2 + coef[1] * rpm * m_pump_out + (coef[2] * m_pump_out ** 2) / rho

    def ggaz(self, p1: float, p2: float, mu: float, area: float, t1: float, k: float, r: float) -> float:
        g0 = self.units.g0 if p1 <= 1e4 else 1.0
        if p1 <= p2:
            return 0.0
        pi = p2 / p1
        pi_crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
        if pi < pi_crit:
            pi = pi_crit
        term = (g0 * (2.0 * k) / (r * t1 * (k - 1.0))) * (pi ** (2.0 / k) - pi ** ((k + 1.0) / k))
        return p1 * mu * area * np.sqrt(max(term, 0.0))

    @staticmethod
    def ftren(sila: float, rtr: float, f3: float, c3: float, dy: float) -> float:
        if -1e-5 < dy <= 1e-5:
            return rtr if sila >= 0.0 else -rtr
        return (f3 if dy >= 0.0 else -f3) + c3 * dy

    @staticmethod
    def func_check_valve_calc_resis_ot_x(x: float, diam_seat: float, f_gap_body_flap: float, coef_corrective_resis: float) -> float:
        dx = 0.0
        f_gap = 0.0 if x <= dx else np.pi * diam_seat * (x - dx)
        f_throt = min(f_gap, f_gap_body_flap)
        f_throt = max(f_throt, 1e-10)
        return coef_corrective_resis / (f_throt ** 2)

    def func_check_valve(
        self,
        p1: float,
        p2: float,
        u: float,
        x: float,
        geom: ValveGeometry,
        diam_seat: float,
        sila_spring0: float,
        z_spring: float,
        myu_throt: float,
        sila_frict_static: float,
        sila_frict_kinetic: float,
        coef_prop_frict_viscous: float,
        coef_sila_flow: float,
    ) -> tuple[float, float, float]:
        _x2 = geom.f_gap_body_flap / (np.pi * diam_seat)
        myu_throt1 = 0.8
        myu_throt2 = 0.8
        f_gap = np.pi * diam_seat * x
        den = myu_throt1 ** 2 * (np.pi * diam_seat * x) ** 2 + myu_throt2 ** 2 * geom.f_gap_body_flap ** 2
        if den <= 0.0:
            px = p2
        else:
            px = (
                myu_throt1 ** 2 * (np.pi * diam_seat * x) ** 2 * p1
                + myu_throt2 ** 2 * geom.f_gap_body_flap ** 2 * p2
            ) / den
        f_throt = min(f_gap, geom.f_gap_body_flap)
        sila_spring = sila_spring0 + z_spring * x
        sila_flow = 2.0 * myu_throt * f_gap * (px - p2) * np.cos(69.0 * np.pi / 180.0) * coef_sila_flow
        sila = p1 * geom.f_seat - p2 * geom.f_flap_exter + px * (geom.f_flap_exter - geom.f_seat) - sila_spring
        sila_frict = self.ftren(sila, sila_frict_static, sila_frict_kinetic, coef_prop_frict_viscous, 0.0)
        d2x = (sila - sila_frict) / geom.mas_moving_parts
        dx = u
        du = d2x
        f_throt = max(f_throt, 1e-10)
        return dx, du, f_throt

    @staticmethod
    def def_check_val(diam_seat: float, diam_body_inter: float, diam_flap_exter: float, diam_connector_in: float, mas_flap: float, mas_spring: float, diam_stem_flap: float) -> ValveGeometry:
        f_seat = 0.25 * np.pi * diam_seat ** 2
        f_connector_in = 0.25 * np.pi * diam_connector_in ** 2
        _ = f_connector_in
        mas_moving_parts = mas_flap + mas_spring / 3.0
        f_body_inter = 0.25 * np.pi * diam_body_inter ** 2
        f_flap_exter = 0.25 * np.pi * diam_flap_exter ** 2
        f_gap_body_flap = f_body_inter - f_flap_exter
        _f_stem_flap = 0.25 * np.pi * diam_stem_flap ** 2
        return ValveGeometry(f_gap_body_flap, f_seat, mas_moving_parts, f_flap_exter)

    @staticmethod
    def func_conditions_xdx_val(x_valv: float, dx_valv: float, x_flap_max: float) -> tuple[float, float]:
        x_out, dx_out = x_valv, dx_valv
        if x_out >= x_flap_max:
            x_out = x_flap_max
            if dx_out > 0.0:
                dx_out = 0.0
        if x_out <= 0.0:
            x_out = 0.0
            if dx_out < 0.0:
                dx_out = 0.0
        return x_out, dx_out

    @staticmethod
    def func_filling(
        flag_fill: int,
        flag_val_open: int,
        condition_time_val_open: bool,
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
        func_interp_fval: Callable[[float], float],
        t: float,
        power: float = 4.0,
    ) -> tuple[int, int, float, float, float, float]:
        m_fill = m
        m_jet = 0.0
        dm = 0.0
        dvol = 0.0
        if condition_time_val_open:
            ratio_vol = vol / vol_nom if vol_nom != 0.0 else 0.0
            if np.iscomplex(ratio_vol):
                ratio_vol = 0.0
            if ratio_vol < 0.0:
                ratio_vol = 0.0
                dvol = 0.0
            if flag_fill == 0 and ratio_vol >= 1.0:
                ratio_vol = 1.0
                flag_fill = 1
            f_val = float(func_interp_fval(t))
            if f_val <= 0.0:
                f_val = 1e-10
            resis_val = resis_val_nom * (f_val_max / f_val) ** 2
            fl1 = ratio_vol ** 0.1
            resis_pipe = resis_pipe1 + resis_val + fl1 * resis_pipe2 + EPS
            inert_pipe = inert_pipe1 + fl1 * inert_pipe2 + EPS
            if flag_val_open == 1:
                dm = (p1 - p2 - resis_pipe / rho * abs(m_fill) * m_fill) / inert_pipe
            else:
                dm = 0.0
                base = (p1 - p2) / resis_pipe * rho
                m_fill = np.sqrt(max(base, 0.0))
                if resis_pipe < 100.0 * (resis_pipe1 + resis_val_nom + fl1 * resis_pipe2):
                    flag_val_open = 1
                else:
                    flag_val_open = 0
            if flag_fill == 1:
                dvol = 0.0
                m_jet = m_fill
            else:
                fl2 = 1.0 - 0.98 * ratio_vol ** power
                dvol = (m_fill * fl2) / rho
                m_jet = m_fill * (1.0 - fl2)
        else:
            if m_fill <= 0.0:
                m_fill = 0.0
            m_jet = m_fill
        if np.iscomplex(dm):
            dm = 0.0
        if np.iscomplex(dvol):
            dvol = 0.0
        return flag_fill, flag_val_open, float(dm), float(dvol), float(m_fill), float(m_jet)

    def _lookup(self, name: str, default: float = 0.0) -> float:
        return float(self.p.get(name, default))

    def _func_h_pmp1_fu(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsHPmpFu1", [0.0, 0.0, 0.0]), dtype=float)
        return self.func_base_pump_h(mass_flow, rho, rpm, coef)

    def _func_h_pmp2_fu(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsHPmpFu2", [0.0, 0.0, 0.0]), dtype=float)
        return self.func_base_pump_h(mass_flow, rho, rpm, coef)

    def _func_h_pmp_ox(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsHoPmp", [0.0, 0.0, 0.0]), dtype=float)
        return self.func_base_pump_h(mass_flow, rho, rpm, coef)

    def _func_eta_pmp_fu1(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsEtaPmpFu1", [1.0, 1.0, 1.0]), dtype=float)
        return self.func_base_pump_eta(mass_flow, rho, rpm, coef)

    def _func_eta_pmp_fu2(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsEtaPmpFu2", [1.0, 1.0, 1.0]), dtype=float)
        return self.func_base_pump_eta(mass_flow, rho, rpm, coef)

    def _func_eta_pmp_ox(self, mass_flow: float, rho: float, rpm: float) -> float:
        coef = np.asarray(self.p.get("coefsEtaPmpOx", [1.0, 1.0, 1.0]), dtype=float)
        return self.func_base_pump_eta(mass_flow, rho, rpm, coef)

    def _func_eta_trb(self, u_to_c: float) -> float:
        coef = np.asarray(self.p.get("coefsEtaTrb", [0.0, 0.0]), dtype=float)
        return self.func_base_turb_eta(u_to_c, coef)

    def initial_state(self) -> np.ndarray:
        p_env = self.p["pEnv"]
        omega0 = 1.0 * OMEGA_COEF
        values = {
            "xCVf1": 0.0, "xCVf2": 0.0, "xCVf3": 0.0, "xCVf4": 0.0, "xCVo1": 0.0,
            "uCVf1": 0.0, "uCVf2": 0.0, "uCVf3": 0.0, "uCVf4": 0.0, "uCVo1": 0.0,
            "masbf": 0.0, "masbo": 0.0,
            "pf1": self.p.get("pfT", p_env), "pf4": self.p.get("pfT", p_env), "pf5": self.p.get("pfB", p_env),
            "pf6": self.p.get("pfB", p_env), "pf7": self.p.get("pfB", p_env),
            "poB": self.p.get("poBNom", p_env), "po1": self.p.get("poT", p_env), "po3": self.p.get("poT", p_env),
            "po4": self.p.get("poB", p_env), "pGg": p_env, "pGv": p_env, "pCh": p_env,
            "mfTx1": 0.0, "mf2xCh": 0.0, "mf3x4": 0.0, "mf4x6": 0.0, "mf6xIc": 0.0,
            "mf4x5": 0.0, "mf5xIg": 0.0, "mfBx7": 0.0, "mf7x6": 0.0, "mf7x5": 0.0,
            "moTx1": 0.0, "mo2x3": 0.0, "mo3xGg": 0.0, "mo2xGg": 0.0, "mo3x4": 0.0,
            "mo4xIc": 0.0, "mo4xIg": 0.0, "moBx4": 0.0,
            "volVf1xCh": 0.0, "volVo1xGg": 0.0,
            "omega": omega0,
        }
        return np.array([values[name] for name in STATE_NAMES], dtype=float)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        s = {name: float(y[idx]) for name, idx in STATE_INDEX.items()}
        d = {name: 0.0 for name in STATE_NAMES}
        p = self.p

        p_env = p["pEnv"]
        p_max = p["pMax"]

        for name in ["po1", "po3", "po4"]:
            s[name] = min(max(s[name], p_env), p_max)
        for name in ["pf1", "pf4", "pf5", "pf6", "pf7"]:
            s[name] = min(max(s[name], p_env), p_max)
        for name in ["pGg", "pGv", "pCh"]:
            s[name] = min(max(s[name], p_env), p_max)

        s["volVf1xCh"] = min(max(s["volVf1xCh"], 0.0), p["volVf1xChNom"])
        s["volVo1xGg"] = min(max(s["volVo1xGg"], 0.0), p["volVo1xGgNom"])
        s["omega"] = max(s["omega"], 1.0)
        rpm = s["omega"] / OMEGA_COEF

        rho_fu = p.get("rhoFu", 1.0)
        rho_ox = p.get("rhoOx", 1.0)
        koB = p.get("koB", 1.4)
        roB = p.get("RoB", p.get("REnv", 287.0))

        d["masbf"] = -s["mfBx7"]
        d["masbo"] = -s["moBx4"]
        volfB = p.get("volfBNom", 1.0) - abs(s["masbf"]) / p.get("rhofB", 1.0)
        _ = volfB
        poB = p.get("poBNom", p_env)
        toB = p.get("ToB0", p.get("TEnv", 293.15)) * (poB / p.get("poBNom", poB)) ** ((koB - 1.0) / koB)
        _ = toB
        d["poB"] = 0.0

        d["mfTx1"] = (p.get("pfT", p_env) - s["pf1"] - p.get("rfTx1", 0.0) / rho_fu * abs(s["mfTx1"]) * s["mfTx1"]) / max(p.get("ifTx1", 1.0), EPS)
        mf_ch_cool = p.get("mfChCoolNom", 0.0) * (s["mf2xCh"] / max(p.get("mf2xChNom", 1.0), EPS))
        d["pf1"] = (s["mfTx1"] - s["mf2xCh"] - mf_ch_cool - s["mf3x4"]) / max(p.get("Cfp1cav", 1.0), EPS)

        hf_pmp1 = self._func_h_pmp1_fu(s["mf2xCh"] + mf_ch_cool + s["mf3x4"], rho_fu, rpm)
        pf2 = s["pf1"] + hf_pmp1 * p.get("coefHfPmp1", 1.0)
        self.flags.flag_fill_vf1xch, self.flags.flag_val_open_vf1, d["mf2xCh"], d["volVf1xCh"], mf2xch_fill, mf2xch_jet = self.func_filling(
            self.flags.flag_fill_vf1xch,
            self.flags.flag_val_open_vf1,
            p.get("tVf11", 0.0) < t <= (p.get("tVf12", 0.0) + p.get("dtVf12", 0.0)),
            s["volVf1xCh"],
            p["volVf1xChNom"],
            p.get("rVf1Nom", 1.0),
            p.get("aVf1Max", 1.0),
            p.get("rf2xVf1", 0.0),
            p.get("rfVf1xCh", 0.0),
            p.get("if2xCVf1", 1.0),
            p.get("ifCVf1xCh", 1.0),
            pf2,
            s["pCh"],
            s["mf2xCh"],
            rho_fu,
            self.int_a_vf1,
            t,
            power=4.0,
        )

        hf_pmp2 = self._func_h_pmp2_fu(s["mf3x4"], rho_fu, rpm)
        pf3 = pf2 + hf_pmp2 * p.get("coefHfPmp2", 1.0)
        d["mf3x4"] = (pf3 - s["pf4"] - p.get("rf3x4", 0.0) / rho_fu * abs(s["mf3x4"]) * s["mf3x4"]) / max(p.get("if3x4", 1.0), EPS)

        r_cvf2 = self.func_check_valve_calc_resis_ot_x(s["xCVf2"], p.get("diamSeatCV01", 1.0), p.get("fGapBodyFlapCV01", 1.0), p.get("coefRCVf2", 1.0))
        rf4x6 = p.get("rf4xCVf2", 0.0) + r_cvf2 + p.get("rfCVf2x6", 0.0)
        d["mf4x6"] = (s["pf4"] - s["pf6"] - rf4x6 / rho_fu * abs(s["mf4x6"]) * s["mf4x6"]) / max(p.get("if4x6", 1.0), EPS)

        r_vf3 = 1.0 / max(2.0 * self.int_a_vf3(t) ** 2, EPS)
        rf6xic = p.get("rf6xVf3", 0.0) + r_vf3
        d["mf6xIc"] = (s["pf6"] - s["pCh"] - rf6xic / rho_fu * abs(s["mf6xIc"]) * s["mf6xIc"]) / max(p.get("if6xIc", 1.0), EPS)

        d["mf7x6"] = (s["pf7"] - s["pf6"] - p.get("rf7x6", 0.0) / rho_fu * abs(s["mf7x6"]) * s["mf7x6"]) / max(p.get("if7x6", 1.0), EPS)

        r_cvf1 = self.func_check_valve_calc_resis_ot_x(s["xCVf1"], p.get("diamSeatCV01", 1.0), p.get("fGapBodyFlapCV01", 1.0), p.get("coefRCVf1", 1.0))
        rf4x5 = p.get("rf4xCVf1", 0.0) + r_cvf1 + p.get("rfCVf1x5", 0.0)
        d["mf4x5"] = (s["pf4"] - s["pf5"] - rf4x5 / rho_fu * abs(s["mf4x5"]) * s["mf4x5"]) / max(p.get("if4x5", 1.0), EPS)

        r_vf2 = 1.0 / max(2.0 * self.int_a_vf2(t) ** 2, EPS)
        rf5xig = p.get("rf5xVf2", 0.0) + r_vf2
        d["mf5xIg"] = (s["pf5"] - s["pGg"] - rf5xig / rho_fu * abs(s["mf5xIg"]) * s["mf5xIg"]) / max(p.get("if5xIg", 1.0), EPS)

        r_cvf3 = self.func_check_valve_calc_resis_ot_x(s["xCVf3"], p.get("diamSeatCV01", 1.0), p.get("fGapBodyFlapCV01", 1.0), p.get("coefRCVf3", 1.0))
        rfbx7 = p.get("rfBxCVf3", 0.0) + r_cvf3 + p.get("rfCVf3x7", 0.0)
        d["mfBx7"] = (p.get("pfB", p_env) - s["pf7"] - rfbx7 / rho_fu * abs(s["mfBx7"]) * s["mfBx7"]) / max(p.get("ifBx7", 1.0), EPS)

        d["pf4"] = (s["mf3x4"] - s["mf4x6"] - s["mf4x5"]) / max(p.get("Cf4", 1.0), EPS)
        d["pf6"] = (s["mf4x6"] + s["mf7x6"] - s["mf6xIc"]) / max(p.get("Cf6", 1.0), EPS)
        d["pf5"] = (s["mf4x5"] + s["mf7x5"] - s["mf5xIg"]) / max(p.get("Cf5", 1.0), EPS)
        d["pf7"] = (s["mfBx7"] - s["mf7x6"] - s["mf7x5"]) / max(p.get("Cf7", 1.0), EPS)

        p_cvf2_in = s["pf4"] - p.get("rf4xCVf2", 0.0) / rho_fu * abs(s["mf4x6"]) * s["mf4x6"] - (p.get("if4xCVf2", 0.0) if s["mf4x6"] > 0.0 else 0.0) * d["mf4x6"]
        p_cvf2_out = s["pf6"] + p.get("rfCVf2x6", 0.0) / rho_fu * abs(s["mf4x6"]) * s["mf4x6"] + (p.get("ifCVf2x6", 0.0) if s["mf4x6"] > 0.0 else 0.0) * d["mf4x6"]

        geom = ValveGeometry(p.get("fGapBodyFlapCV01", 1.0), p.get("fSeatCV01", 1.0), p.get("masMovingPartsCV01", 1.0), p.get("fFlapExterCV01", 1.0))
        d["xCVf2"], d["uCVf2"], _a_cvf2 = self.func_check_valve(
            p_cvf2_in, p_cvf2_out, s["uCVf2"], s["xCVf2"], geom,
            p.get("diamSeatCV01", 1.0), p.get("silaSpring0CV01", 0.0), p.get("zSpringCV01", 0.0),
            p.get("myuThrotCV01", 0.8), p.get("silaFrictStaticCV01", 0.0), p.get("silaFrictKineticCV01", 0.0),
            p.get("coefPropFrictViscousCV01", 0.0), p.get("coefSilaFlow", 1.0),
        )
        d["xCVf1"], d["uCVf1"], _a_cvf1 = self.func_check_valve(
            s["pf4"], s["pf5"], s["uCVf1"], s["xCVf1"], geom,
            p.get("diamSeatCV01", 1.0), p.get("silaSpring0CV01", 0.0), p.get("zSpringCV01", 0.0),
            p.get("myuThrotCV01", 0.8), p.get("silaFrictStaticCV01", 0.0), p.get("silaFrictKineticCV01", 0.0),
            p.get("coefPropFrictViscousCV01", 0.0), p.get("coefSilaFlow", 1.0),
        )
        d["xCVf3"], d["uCVf3"], _a_cvf3 = self.func_check_valve(
            p.get("pfB", p_env), s["pf7"], s["uCVf3"], s["xCVf3"], geom,
            p.get("diamSeatCV01", 1.0), p.get("silaSpring0CV01", 0.0), p.get("zSpringCV01", 0.0),
            p.get("myuThrotCV01", 0.8), p.get("silaFrictStaticCV01", 0.0), p.get("silaFrictKineticCV01", 0.0),
            p.get("coefPropFrictViscousCV01", 0.0), p.get("coefSilaFlow", 1.0),
        )
        d["xCVf4"], d["uCVf4"], _a_cvf4 = self.func_check_valve(
            s["pf7"], s["pf5"], s["uCVf4"], s["xCVf4"], geom,
            p.get("diamSeatCV01", 1.0), p.get("silaSpring0CV01", 0.0), p.get("zSpringCV01", 0.0),
            p.get("myuThrotCV01", 0.8), p.get("silaFrictStaticCV01", 0.0), p.get("silaFrictKineticCV01", 0.0),
            p.get("coefPropFrictViscousCV01", 0.0), p.get("coefSilaFlow", 1.0),
        )

        d["moTx1"] = (p.get("poT", p_env) - s["po1"] - p.get("roTx1", 0.0) / rho_ox * abs(s["moTx1"]) * s["moTx1"]) / max(p.get("ioTx1", 1.0), EPS)
        mo2xtrb_out = p.get("mo2xTrbOutNom", 0.0) * (s["mo2x3"] / max(p.get("mo2x3Nom", 1.0), EPS))
        d["po1"] = (s["moTx1"] - s["mo2x3"] - mo2xtrb_out) / max(p.get("Cfp1cav", 1.0), EPS)
        ho_pmp = self._func_h_pmp_ox(s["mo2x3"] + mo2xtrb_out, rho_ox, rpm)
        po2 = s["po1"] + ho_pmp * p.get("coefHoPmp", 1.0)

        r_vo2 = 1.0 / max(2.0 * self.int_a_vo2(t) ** 2, EPS)
        robx4 = p.get("roBxVo2", 0.0) + r_vo2 + p.get("roVo2x4", 0.0)
        d["moBx4"] = (p.get("poB", p_env) - s["po4"] - robx4 / rho_ox * abs(s["moBx4"]) * s["moBx4"]) / max(p.get("ioBx4", 1.0), EPS)

        r_vo3 = 1.0 / max(2.0 * self.int_a_vo3(t) ** 2, EPS)
        ro4xig = p.get("ro4xVo3", 0.0) + r_vo3 + p.get("roVo3xIg", 0.0)
        d["mo4xIg"] = (s["po4"] - s["pGg"] - ro4xig / rho_ox * abs(s["mo4xIg"]) * s["mo4xIg"]) / max(p.get("io4xIg", 1.0), EPS)

        r_vo4 = 1.0 / max(2.0 * self.int_a_vo4(t) ** 2, EPS)
        ro4xic = p.get("ro4xVo4", 0.0) + r_vo4 + p.get("roVo4xIc", 0.0)
        d["mo4xIc"] = (s["po4"] - s["pCh"] - ro4xic / rho_ox * abs(s["mo4xIc"]) * s["mo4xIc"]) / max(p.get("io4xIc", 1.0), EPS)

        if self.flags.flag_fill_vo1xgg == 0:
            self.flags.flag_fill_vo1xgg, self.flags.flag_val_open_vo1, d["mo2xGg"], d["volVo1xGg"], mo2xgg_fill, mo2xgg_jet = self.func_filling(
                self.flags.flag_fill_vo1xgg,
                self.flags.flag_val_open_vo1,
                p.get("tVo11", 0.0) < t <= (p.get("tVo12", 0.0) + p.get("dtVo12", 0.0)),
                s["volVo1xGg"],
                p["volVo1xGgNom"],
                p.get("rVo1Nom", 1.0),
                p.get("aVo1Max", 1.0),
                p.get("ro2xVo1", 0.0),
                p.get("roVo1x3", 0.0) + p.get("ro3xGg", 0.0),
                p.get("io2xVo1", 1.0),
                p.get("ioVo1x3", 1.0) + p.get("io3xGg", 1.0),
                po2,
                s["pGg"],
                s["mo2xGg"],
                rho_ox,
                self.int_a_vo1,
                t,
                power=8.0,
            )
            d["mo2x3"] = d["mo2xGg"]
            d["mo3xGg"] = d["mo2xGg"]
            d["mo3x4"] = 0.0
            d["po3"] = 0.0
            d["po4"] = (s["moBx4"] - s["mo4xIg"] - s["mo4xIc"]) / max(p.get("Co4", 1.0), EPS)
            mo3xtrb_in = 0.0
        else:
            r_vo1 = 1.0 / max(2.0 * self.int_a_vo1(t) ** 2, EPS)
            ro2x3 = p.get("ro2xVo1", 0.0) + r_vo1 + p.get("roVo1x3", 0.0)
            d["mo2x3"] = (po2 - s["po3"] - ro2x3 / rho_ox * abs(s["mo2x3"]) * s["mo2x3"]) / max(p.get("io2x3", 1.0), EPS)
            d["mo3xGg"] = (s["po3"] - s["pGg"] - p.get("ro3xGg", 0.0) / rho_ox * abs(s["mo3xGg"]) * s["mo3xGg"]) / max(p.get("io3xGg", 1.0), EPS)
            r_cvo1 = self.func_check_valve_calc_resis_ot_x(s["xCVo1"], p.get("diamSeatCV01", 1.0), p.get("fGapBodyFlapCV01", 1.0), p.get("coefRCVo1", 1.0))
            ro3x4 = p.get("ro3xCVo1", 0.0) + r_cvo1 + p.get("roCVo1x4", 0.0)
            d["mo3x4"] = (s["po3"] - s["po4"] - ro3x4 / rho_ox * abs(s["mo3x4"]) * s["mo3x4"]) / max(p.get("io3x4", 1.0), EPS)
            d["po3"] = (s["mo2x3"] - s["mo3xGg"] - s["mo3x4"]) / max(p.get("Co3", 1.0), EPS)
            d["po4"] = (s["moBx4"] + s["mo3x4"] - s["mo4xIg"] - s["mo4xIc"]) / max(p.get("Co4", 1.0), EPS)
            d["mo2xGg"] = d["mo3xGg"]
            mo3xtrb_in = p.get("mo3xTrbInNom", 0.0) * (s["mo3xGg"] / max(p.get("mo3xGgNom", 1.0), EPS))
            d["xCVo1"], d["uCVo1"], _a_cvo1 = self.func_check_valve(
                s["po3"], s["po4"], s["uCVo1"], s["xCVo1"], geom,
                p.get("diamSeatCV01", 1.0), p.get("silaSpring0CV01", 0.0), p.get("zSpringCV01", 0.0),
                p.get("myuThrotCV01", 0.8), p.get("silaFrictStaticCV01", 0.0), p.get("silaFrictKineticCV01", 0.0),
                p.get("coefPropFrictViscousCV01", 0.0), p.get("coefSilaFlow", 1.0),
            )

        mo_gg = max(s["mo2xGg"] + s["mo4xIg"], 0.0)
        mf_gg = max(s["mf5xIg"], 1e-10)
        mo_ch = max(mo_gg + s["mo4xIc"] + mo2xtrb_out + mo3xtrb_in, 0.0)
        mf_ch = max(s["mf5xIg"] + s["mf6xIc"] + mf_ch_cool + mf2xch_jet, 1e-10)

        rg_g = p.get("REnv", 287.0)
        tg_g = p.get("TEnv", 293.15)
        kg_g = 1.4
        rg_v = rg_g
        kg_v = kg_g
        tg_v = tg_g
        dp_gg = 0.0
        dp_gv = 0.0
        m_gg = 0.0
        m_gv = 0.0

        if self.flags.flag_burn_gg == 1 and self.flags.flag_burn_gg_stop == 0:
            km_gg = max(min(mo_gg / mf_gg, 499.0), 1e-4)
            rg_g = p.get("interpROxygenJetaGg", lambda pressure, km: p.get("REnv", 287.0))(s["pGg"], km_gg) if callable(p.get("interpROxygenJetaGg")) else p.get("REnv", 287.0)
            tg_g = max((p.get("interpTOxygenJetaGg", lambda pressure, km: p.get("TEnv", 293.15))(s["pGg"], km_gg) if callable(p.get("interpTOxygenJetaGg")) else p.get("TEnv", 293.15)), p.get("TEnv", 293.15))
            kg_g = p.get("interpKOxygenJetaGg", lambda pressure, km: 1.4)(s["pGg"], km_gg) if callable(p.get("interpKOxygenJetaGg")) else 1.4
            if t <= self.flags.t_burn_gg + 0.15:
                tg_g = max(p.get("TEnv", 293.15), tg_g / 3.0)
            rg_v = rg_g
            kg_v = kg_g
            tg_v = max(p.get("TEnv", 293.15), tg_g * (max(s["pGv"], EPS) / max(s["pGg"], EPS)) ** ((kg_g - 1.0) / kg_g))
            p_trb_in = p.get("dztaTrb", 1.0) ** 2 * np.sqrt(max(s["pGg"] ** 2 - kg_g * rg_g * tg_g * (mo_gg + mf_gg + mo3xtrb_in) ** 2, 1.0))
            m_gg = self.ggaz(s["pGg"], s["pGv"], p.get("muGg", 1.0), p.get("fGg1", 1.0), tg_g, kg_g, rg_g)
            m_gv = self.ggaz(s["pGv"], s["pCh"], p.get("muGv", 1.0), p.get("fGv1", 1.0), tg_v, kg_v, rg_v)
            dp_gg = (rg_g * tg_g / max(p.get("VGg", 1.0), EPS)) * (mo_gg + mf_gg + mo3xtrb_in - m_gg)
            dp_gv = (rg_v * tg_v / max(p.get("VGv", 1.0), EPS)) * (m_gg + mo2xtrb_out - m_gv)
        else:
            p_trb_in = p_env

        d["pGg"] = dp_gg
        d["pGv"] = dp_gv

        mf_leak_p1 = p.get("mLeakFu1Nom", 0.0) * (rpm / max(p.get("rpmNom", 1.0), EPS))
        eta_pmp1_fu = self._func_eta_pmp_fu1(s["mf2xCh"] + mf_ch_cool + s["mf3x4"], p.get("rhoPmp1FuNom", rho_fu), rpm)
        if eta_pmp1_fu > p.get("etaPmp1FuNom", 1.0) * 0.01:
            m_pmp1_fu_pow = s["mf2xCh"] + mf_ch_cool + s["mf3x4"]
        else:
            m_pmp1_fu_pow = mf_leak_p1
            eta_pmp1_fu = 1.0
        torq_pmp1_fu = 0.0 if eta_pmp1_fu <= 1e-4 else pf2 * m_pmp1_fu_pow / (max(s["omega"], EPS) * p.get("rhoPmp1FuNom", rho_fu) * eta_pmp1_fu)

        mf_leak_p2 = p.get("mLeakFu2Nom", 0.0) * (rpm / max(p.get("rpmNom", 1.0), EPS))
        eta_pmp2_fu = self._func_eta_pmp_fu2(s["mf3x4"], p.get("rhoPmp2FuNom", rho_fu), rpm)
        if eta_pmp2_fu > p.get("etaPmp2FuNom", 1.0) * 0.1:
            m_pmp2_fu_pow = s["mf3x4"]
        else:
            m_pmp2_fu_pow = mf_leak_p2
            eta_pmp2_fu = 1.0
        torq_pmp2_fu = 0.0 if eta_pmp2_fu <= 1e-4 else pf3 * m_pmp2_fu_pow / (max(s["omega"], EPS) * p.get("rhoPmp2FuNom", rho_fu) * eta_pmp2_fu)

        mo_leak_p1 = p.get("mLeakOxNom", 0.0) * (rpm / max(p.get("rpmNom", 1.0), EPS))
        eta_pmp_ox = self._func_eta_pmp_ox(s["mo2x3"] + mo2xtrb_out, p.get("rhoPmpOxNom", rho_ox), rpm)
        if eta_pmp_ox > p.get("etaPmpOxNom", 1.0) * 0.01:
            m_pmp_ox_pow = s["mo2x3"] + mo2xtrb_out
        else:
            m_pmp_ox_pow = mo_leak_p1
            eta_pmp_ox = 1.0
        torq_pmp_ox = 0.0 if eta_pmp_ox <= 1e-4 else po2 * m_pmp_ox_pow / (max(s["omega"], EPS) * p.get("rhoPmpOxNom", rho_ox) * eta_pmp_ox)

        torq_trb = 0.0
        if self.flags.flag_burn_gg == 1 and p_trb_in >= s["pGv"]:
            pi_trb = s["pGv"] / max(p_trb_in, EPS)
            lad_trb = kg_g / (kg_g - 1.0) * rg_g * tg_g * (1.0 - pi_trb ** ((kg_g - 1.0) / kg_g))
            u_trb = np.pi * p.get("DTrb", 1.0) * rpm / 60.0
            cad_trb = np.sqrt(max(2.0 * self.units.g0 * lad_trb, 0.0))
            eta_trb = 0.0
            if cad_trb > 1e-6:
                eta_trb = self._func_eta_trb(u_trb / cad_trb)
                if eta_trb < 0.0:
                    eta_trb = 0.0
            torq_trb = 0.0 if s["omega"] <= EPS else m_gv * lad_trb * eta_trb * p.get("coefTorqTrb", 1.0) / s["omega"]

        d["omega"] = (torq_trb - torq_pmp_ox - torq_pmp1_fu - torq_pmp2_fu) * self.units.g0 / max(p.get("JTrbPmp", 1.0), EPS)

        if self.flags.flag_burn_ch == 1 and self.flags.flag_burn_ch_stop == 0:
            km_ch = max(min(mo_ch / mf_ch, 99.0), 1e-4)
            r_ch = p.get("interpROxygenJetaCh", lambda pressure, km: p.get("REnv", 287.0))(s["pCh"], km_ch) if callable(p.get("interpROxygenJetaCh")) else p.get("REnv", 287.0)
            t_ch = p.get("interpTOxygenJetaCh", lambda pressure, km: p.get("TEnv", 293.15))(s["pCh"], km_ch) if callable(p.get("interpTOxygenJetaCh")) else p.get("TEnv", 293.15)
            k_ch = p.get("interpKOxygenJetaCh", lambda pressure, km: 1.4)(s["pCh"], km_ch) if callable(p.get("interpKOxygenJetaCh")) else 1.4
            m_ch = self.ggaz(s["pCh"], p_env, p.get("muThCh", 1.0), p.get("aThChcorect", 1.0), t_ch, k_ch, r_ch)
            d["pCh"] = (r_ch * t_ch / max(p.get("VCh", 1.0), EPS)) * (m_gv + s["mo4xIc"] + s["mf6xIc"] + s["mf2xCh"] + mf_ch_cool - m_ch)
        else:
            d["pCh"] = 0.0

        return np.array([d[name] for name in STATE_NAMES], dtype=float)

    def simulate(self, end_time: float | None = None, dt: float | None = None) -> SimulationResult:
        dt = self.p["dt"] if dt is None else dt
        end_time = self.p["endTime"] if end_time is None else end_time
        n = int(end_time / dt)
        step_print = max(int(end_time / dt / max(self.p.get("countPrint", 1000.0), 1.0)), 1)

        y = self.initial_state()
        time_hist: list[float] = []
        y_hist: list[np.ndarray] = []

        t = 0.0
        for j in range(1, n + 1):
            if t >= 1000.0:
                break
            if (j - 1) % step_print == 0:
                time_hist.append(t)
                y_hist.append(y.copy())
            y = rk4_step(self.rhs, y, t, dt)
            for idx, name in enumerate(["xCVf1", "xCVf2", "xCVf3", "xCVf4", "xCVo1"]):
                pos, vel = self.func_conditions_xdx_val(y[STATE_INDEX[name]], y[STATE_INDEX[name.replace("x", "u", 1)]], self.p.get("xFlapMax", 1e-3))
                y[STATE_INDEX[name]] = pos
                y[STATE_INDEX[name.replace("x", "u", 1)]] = vel
            t += dt

        time_arr = np.asarray(time_hist, dtype=float)
        y_arr = np.vstack(y_hist) if y_hist else np.empty((0, len(STATE_NAMES)))
        history = {name: y_arr[:, i] if len(y_arr) else np.array([]) for i, name in enumerate(STATE_NAMES)}
        history["1_time"] = time_arr
        return SimulationResult(time=time_arr, y=y_arr, state_names=STATE_NAMES.copy(), history=history)
