"""
Pearson correlations and OLS between log10 DENV-1 and log10 wMel titers.

Whole-data correlation reproduces Fig 3A (R≈0.251, p≈0.051).
Stratified results reproduce Table 1.
Only wMelBR mosquitoes have Wolbachia (the wMelTET group has none by design).
"""

import numpy as np
import pandas as pd
from scipy import stats


def _log10_pair(df: pd.DataFrame):
    """Return log10(denv), log10(wmel) for rows where both are > 0."""
    sub = df[(df["denv_titer"] > 0) & (df["wmel_titer"] > 0)].copy()
    return np.log10(sub["denv_titer"].values), np.log10(sub["wmel_titer"].values)


def correlation_row(label, denv_log, wmel_log):
    if len(denv_log) < 4:
        return None
    slope, intercept, r, p, se = stats.linregress(denv_log, wmel_log)
    return {
        "label":     label,
        "n":         len(denv_log),
        "r":         round(r, 3),
        "R2":        round(r**2, 3),
        "slope":     round(slope, 3),
        "intercept": round(intercept, 3),
        "p_value":   round(p, 4),
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute whole-dataset + per-(dose, time) correlations for wMelBR mosquitoes.
    Only conditions with > 3 non-zero detectable pairs are included.
    """
    br = df[df["group"].str.upper().str.contains("MEL|BR|WMEL")].copy()

    rows = []

    # whole data
    d, w = _log10_pair(br)
    row = correlation_row("all", d, w)
    if row:
        rows.append(row)

    # per dose × time
    for (dose_exp, dpi), grp in br.groupby(["dose_exp", "dpi"]):
        d, w = _log10_pair(grp)
        label = f"10^{dose_exp} | {dpi}dpi"
        row = correlation_row(label, d, w)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)
