import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.model import solve, equilibria, rhs, initial_conditions
from denv_wmel.simulation import PAPER_PARAMS

PARAMS = PAPER_PARAMS.copy()


def test_ode_runs():
    t = np.linspace(0, 10, 100)
    traj = solve(PARAMS, P0=15.0, t=t)
    assert traj.shape == (100, 2)
    assert np.all(np.isfinite(traj))


def test_pathogen_free_eq():
    (P_free, R_free), _, _ = equilibria(PARAMS)
    assert P_free == 0.0
    assert abs(R_free - PARAMS["alpha"] / PARAMS["gamma"]) < 1e-10


def test_systemic_eq_positive():
    (_, _), (P_sys, R_sys), _ = equilibria(PARAMS)
    assert P_sys > 0
    assert R_sys > 0


def test_rhs_zero_state():
    state = [0.0, PARAMS["alpha"] / PARAMS["gamma"]]
    dP, dR = rhs(state, 0, **PARAMS)
    assert dP == 0.0   # no pathogens → no growth
    assert abs(dR) < 1e-9   # at constitutive equilibrium


def test_initial_conditions_decrease():
    P0s = initial_conditions(15.0, 2.0, 5)
    assert P0s[0] == 15.0
    assert all(P0s[i] > P0s[i+1] for i in range(len(P0s)-1))


def test_correlation_whole_dataset():
    """Whole-dataset correlation: R ≈ 0.251, p ≈ 0.051 (Table paper)."""
    from denv_wmel.preprocessing import preprocess
    from denv_wmel.correlations import compute_correlations
    df = preprocess()
    corr = compute_correlations(df)
    row = corr[corr["label"] == "all"].iloc[0]
    assert abs(row["r"] - 0.251) < 0.03, f"r = {row['r']}"
    assert row["p_value"] < 0.10, f"p = {row['p_value']}"
