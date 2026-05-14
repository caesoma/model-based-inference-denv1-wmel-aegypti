"""
Load raw qPCR CT values and convert to normalised integer titers.

DENV and Wolbachia (TM513) levels are expressed relative to the host housekeeping
gene RPS17, and then rescaled so the minimum non-zero value equals 1.
"""

import numpy as np
import pandas as pd

DATA_PATH = "raw_data/S1 Data.csv"

DENV_COLS  = ["CT1_DENV",  "CT2_DENV",  "CT3_DENV",  "CT4_DENV",  "CT5_DENV",  "CT6_DENV"]
WMEL_COLS  = ["CT1_TM513", "CT2_TM514", "CT3_TM515", "CT4_TM516"]
RPS_COLS   = ["CT1_RPS17", "CT2_RPS17", "CT3_RPS17", "CT4_RPS17"]

DOSE_ORDER = ["10^4", "10^5", "10^6", "10^7", "10^8"]


def _mean_ct(row, cols):
    vals = pd.to_numeric(row[cols], errors="coerce")
    return vals.mean()


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df.rename(columns={"In—culo": "dose", "Inóculo": "dose"}, errors="ignore", inplace=True)
    # unify any remaining encoding variant
    for c in df.columns:
        if "culo" in c:
            df.rename(columns={c: "dose"}, inplace=True)
    df["dose"] = df["dose"].astype(str).str.strip()
    df["Pop"]  = df["Pop"].astype(str).str.strip()
    return df


def _ct_to_rel(mean_ct_target, mean_ct_rps):
    """ΔCT-based relative quantity: 2^(-(CT_target - CT_rps))."""
    if np.isnan(mean_ct_target) or np.isnan(mean_ct_rps):
        return 0.0
    return 2.0 ** (-(mean_ct_target - mean_ct_rps))


def compute_titers(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        ct_denv = _mean_ct(row, DENV_COLS)
        ct_wmel = _mean_ct(row, WMEL_COLS)
        ct_rps  = _mean_ct(row, RPS_COLS)

        denv_rel = _ct_to_rel(ct_denv, ct_rps)
        wmel_rel = _ct_to_rel(ct_wmel, ct_rps)

        records.append({
            "sample": row["amostra"],
            "group":  row["Pop"],
            "dpi":    row["dpi"],
            "dose":   row["dose"],
            "denv_rel": denv_rel,
            "wmel_rel": wmel_rel,
        })

    out = pd.DataFrame(records)
    # drop controls and blank rows
    out = out[out["dose"].isin(DOSE_ORDER)].copy()
    out["dpi"] = pd.to_numeric(out["dpi"], errors="coerce")
    out = out.dropna(subset=["dpi"])
    out["dpi"] = out["dpi"].astype(int)
    return out


def scale_to_unity(series: pd.Series) -> pd.Series:
    """Set smallest non-zero value to 1, rescale all others proportionally, round."""
    nonzero = series[series > 0]
    if nonzero.empty:
        return series.copy()
    min_val = nonzero.min()
    scaled = series / min_val
    return scaled.round().astype(int)


def preprocess(path: str = DATA_PATH) -> pd.DataFrame:
    raw = load_raw(path)
    titers = compute_titers(raw)
    titers["denv_titer"] = scale_to_unity(titers["denv_rel"])

    # wMelTET are confirmed Wolbachia-negative controls; set their wMel titer to 0
    # and scale only using the wMelBR distribution.
    is_tet = titers["group"].str.upper().str.contains("TET")
    wmel_br_scaled = scale_to_unity(titers.loc[~is_tet, "wmel_rel"])
    titers["wmel_titer"] = 0
    titers.loc[~is_tet, "wmel_titer"] = wmel_br_scaled

    titers["dose_exp"] = titers["dose"].str.extract(r"\^(\d+)").astype(int)
    return titers[["sample", "group", "dpi", "dose", "dose_exp",
                   "denv_titer", "wmel_titer"]]
