"""
Generalised linear model: log(denv_titer) ~ group * dose * time
on all mosquitoes with detectable (>0) DENV-1 titers.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_glm(df: pd.DataFrame) -> tuple:
    """Returns (result, anova_table)."""
    sub = df[df["denv_titer"] > 0].copy()
    sub["log_titer"] = np.log(sub["denv_titer"].astype(float))
    sub["dose"]  = sub["dose_exp"].astype(str)
    sub["time"]  = sub["dpi"].astype(str)
    sub["group"] = sub["group"].str.upper().str.replace("WMEL", "wMEL")

    formula = "log_titer ~ C(group) * C(dose) * C(time)"
    result = smf.ols(formula, data=sub).fit()
    anova = smf.ols(formula, data=sub).fit()

    from statsmodels.stats.anova import anova_lm
    anova_table = anova_lm(result, typ=2)
    return result, anova_table
