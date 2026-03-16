import numpy as np

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

def initial_y(params):
    return np.array([
        params.xCVf1_0, params.xCVf2_0, params.xCVf3_0, params.xCVf4_0, params.xCVo1_0,
        params.uCVf1_0, params.uCVf2_0, params.uCVf3_0, params.uCVf4_0, params.uCVo1_0,
        params.masbf_0, params.masbo_0,
        params.pf1_0, params.pf4_0, params.pf5_0, params.pf6_0, params.pf7_0,
        params.poB_0, params.po1_0, params.po3_0, params.po4_0,
        params.pGg_0, params.pGv_0, params.pCh_0,
        params.mfTx1_0, params.mf2xCh_0, params.mf3x4_0, params.mf4x6_0, params.mf6xIc_0,
        params.mf4x5_0, params.mf5xIg_0, params.mfBx7_0, params.mf7x6_0, params.mf7x5_0,
        params.moTx1_0, params.mo2x3_0, params.mo3xGg_0, params.mo2xGg_0, params.mo3x4_0,
        params.mo4xIc_0, params.mo4xIg_0, params.moBx4_0,
        params.volVf1xCh_0, params.volVo1xGg_0,
        params.omega_0,
    ], dtype=np.float64)
