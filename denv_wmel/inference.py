"""
Bayesian inference of host-pathogen ODE parameters via PyMC v5 + Metropolis.

Parameters shared between groups: r, k, delta_p, alpha, gamma
Parameters that differ: lambda, delta_R, P0_high, dosefold

Likelihood: Poisson on integer-rounded titers.
Conditions with ≤ 6 non-zero titers are excluded (per paper).

The ODE is wrapped as a non-differentiable pytensor op; Metropolis-Hastings
is used instead of NUTS.  Starting values are set near the paper's simulation
parameters (Fig 4 caption) to ensure adequate acceptance rates.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import pymc as pm
import arviz as az

from .model import rhs

DOSE_ORDER = [4, 5, 6, 7, 8]   # exponents; index 0 = lowest (10^4)
TIMEPOINTS = [3, 7, 14]
GROUPS     = ["TET", "wMEL"]

# Starting values close to the paper's forward-simulation parameters (Fig 4)
# These give a reasonable acceptance rate for Metropolis.
START_SHARED = dict(r=1.2, k=1e-5, delta_p=4.0, alpha=8.0, gamma=4.0)
START_TET    = dict(lam_TET=100.0, delta_R_TET=4.0, P0_TET=15.0, dosefold_TET=3.0)
START_WMEL   = dict(lam_wMEL=200.0, delta_R_wMEL=9.0, P0_wMEL=10.0, dosefold_wMEL=3.0)
START_ALL    = {**START_SHARED, **START_TET, **START_WMEL}


def _prep_obs(df: pd.DataFrame) -> dict:
    """
    Returns dict: group → {(dose_exp, dpi): array of positive integer titers}.
    Conditions with ≤ 6 non-zero titers excluded (paper threshold).
    """
    obs = {}
    for group in GROUPS:
        key = "TET" if group == "TET" else "MEL"
        g = df[df["group"].str.upper().str.contains(key)]
        obs[group] = {}
        for dose_exp in DOSE_ORDER:
            for dpi in TIMEPOINTS:
                sub = g[(g["dose_exp"] == dose_exp) & (g["dpi"] == dpi)]
                pos = sub[sub["denv_titer"] > 0]["denv_titer"].values
                # paper: conditions with six or fewer non-zero titers excluded
                if len(pos) > 6:
                    obs[group][(dose_exp, dpi)] = pos.astype(int)
    return obs


# ---------------------------------------------------------------------------
# Non-differentiable ODE op — computes all (dose × time) predictions at once
# for a given set of group parameters.
# ---------------------------------------------------------------------------

@as_op(
    itypes=[pt.dscalar] * 9,   # r, delta_p, k, alpha, lam, gamma, delta_R, P0_high, dosefold
    otypes=[pt.dvector],        # flat vector: len(DOSE_ORDER) × len(TIMEPOINTS)
)
def _ode_predictions(r, delta_p, k, alpha, lam, gamma, delta_R, P0_high, dosefold):
    preds = []
    for dose_exp in DOSE_ORDER:
        idx = 8 - dose_exp                                  # 0 = highest dose
        P0  = float(P0_high) / (float(dosefold) ** idx)
        R0  = float(alpha) / max(float(gamma), 1e-9)
        args = (float(r), float(delta_p), float(k),
                float(alpha), float(lam), float(gamma), float(delta_R))
        for dpi in TIMEPOINTS:
            t = np.linspace(0.0, float(dpi), max(200, int(dpi * 50)))
            try:
                traj = odeint(rhs, [P0, R0], t, args=args, rtol=1e-6, atol=1e-8)
                val  = float(traj[-1, 0])
            except Exception:
                val  = 1e-3
            preds.append(max(val, 1e-3))
    return np.array(preds, dtype=np.float64)


def _prediction_index(dose_exp: int, dpi: int) -> int:
    return DOSE_ORDER.index(dose_exp) * len(TIMEPOINTS) + TIMEPOINTS.index(dpi)


def build_and_sample(df: pd.DataFrame,
                     draws: int = 2000,
                     tune: int = 2000,
                     chains: int = 4,
                     random_seed: int = 42) -> az.InferenceData:
    obs = _prep_obs(df)

    # gamma-prior means: use paper-informed start values
    r_mean  = START_SHARED["r"]
    P0_mean = START_TET["P0_TET"]

    def gamma_params(mean: float, cv: float = 0.5):
        """Returns (alpha, beta) for a Gamma with given mean and coefficient of variation."""
        sd = mean * cv
        return mean**2 / sd**2, mean / sd**2

    with pm.Model() as model:
        # --- shared parameters ---
        r_a, r_b  = gamma_params(r_mean, cv=0.5)
        r         = pm.Gamma("r",       alpha=r_a, beta=r_b)
        k         = pm.Uniform("k",       lower=1e-8, upper=1e-3)
        delta_p   = pm.Uniform("delta_p", lower=0.0,  upper=20.0)
        alpha     = pm.Uniform("alpha",   lower=0.0,  upper=20.0)
        gamma     = pm.Uniform("gamma",   lower=0.01, upper=20.0)

        # --- group-specific parameters ---
        lam_TET   = pm.Uniform("lam_TET",      lower=0.0, upper=500.0)
        lam_wMEL  = pm.Uniform("lam_wMEL",     lower=0.0, upper=500.0)
        dR_TET    = pm.Uniform("delta_R_TET",  lower=0.0, upper=20.0)
        dR_wMEL   = pm.Uniform("delta_R_wMEL", lower=0.0, upper=20.0)

        P0a, P0b  = gamma_params(P0_mean, cv=0.5)
        P0_TET    = pm.Gamma("P0_TET",  alpha=P0a, beta=P0b)
        P0_wMEL   = pm.Gamma("P0_wMEL", alpha=P0a, beta=P0b)

        # gamma prior centered on 3; tenfold dilution known per dose step
        df_TET    = pm.Gamma("dosefold_TET",  alpha=9.0, beta=3.0)
        df_wMEL   = pm.Gamma("dosefold_wMEL", alpha=9.0, beta=3.0)

        group_params = {
            "TET":  (lam_TET,  dR_TET,  P0_TET,  df_TET),
            "wMEL": (lam_wMEL, dR_wMEL, P0_wMEL, df_wMEL),
        }

        # compute all dose×time ODE predictions once per group
        for group, cond_dict in obs.items():
            if not cond_dict:
                continue
            lam_g, dR_g, P0_g, df_g = group_params[group]

            all_preds = _ode_predictions(
                r.astype("float64"),
                delta_p.astype("float64"),
                k.astype("float64"),
                alpha.astype("float64"),
                lam_g.astype("float64"),
                gamma.astype("float64"),
                dR_g.astype("float64"),
                P0_g.astype("float64"),
                df_g.astype("float64"),
            )

            for (dose_exp, dpi), y_obs in cond_dict.items():
                idx = _prediction_index(dose_exp, dpi)
                mu  = pm.Deterministic(f"mu_{group}_{dose_exp}_{dpi}", all_preds[idx])
                pm.Poisson(
                    f"obs_{group}_{dose_exp}_{dpi}",
                    mu=pm.math.maximum(mu, 1e-3),
                    observed=y_obs,
                )

        # Metropolis-Hastings (ODE op has no gradient)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                step=pm.Metropolis(),
                initvals=START_ALL,
                random_seed=random_seed,
                progressbar=True,
            )

    return idata
