"""
Forward simulation (Fig 4): stochastic ensemble + deterministic trajectory
at the paper's reported parameter set.
"""

import numpy as np
from .model import solve, gillespie, initial_conditions

# parameter set from Fig 4 caption
PAPER_PARAMS = dict(
    r=3.5, delta_p=0.042, k=1e-5,
    alpha=4.8, lam=0.12, gamma=0.032,
    delta_R=0.045,
)
P0_HIGH  = 15.0
DOSEFOLD = 2.0
N_DOSES  = 5
TMAX     = 21.0   # covers all experimental time points (3, 7, 14 dpi) with margin
N_TRAJ   = 50     # stochastic realisations per dose
SAMPLE_TIMES = np.array([3.0, 7.0, 14.0])


def run_deterministic():
    t = np.linspace(0, TMAX, 500)
    P0_list = initial_conditions(P0_HIGH, DOSEFOLD, N_DOSES)
    return t, [solve(PAPER_PARAMS, P0, t) for P0 in P0_list]


def run_stochastic(n_traj=N_TRAJ):
    P0_list = initial_conditions(P0_HIGH, DOSEFOLD, N_DOSES)
    results = []
    for i, P0 in enumerate(P0_list):
        trajs = []
        for j in range(n_traj):
            ts, Ps, _ = gillespie(PAPER_PARAMS, P0, TMAX, seed=i * 1000 + j)
            trajs.append((ts, Ps))
        results.append(trajs)
    return results


def sample_pseudo_data(stochastic_results, sample_times=SAMPLE_TIMES):
    """
    For each dose × trajectory, sample P at the given time points by finding
    the nearest state in the SSA trajectory.
    """
    records = []
    for dose_idx, trajs in enumerate(stochastic_results):
        for traj_idx, (ts, Ps) in enumerate(trajs):
            for t_s in sample_times:
                idx = np.searchsorted(ts, t_s)
                idx = min(idx, len(Ps) - 1)
                records.append({
                    "dose_idx":  dose_idx,
                    "traj_idx":  traj_idx,
                    "time":      t_s,
                    "P":         Ps[idx],
                })
    import pandas as pd
    return pd.DataFrame(records)
