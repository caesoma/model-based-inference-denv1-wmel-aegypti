"""
Host-pathogen ODE model (Eq. 1 from the paper):

  dP/dt = r*P - delta_p*P*R - k*P^2
  dR/dt = alpha + lambda*P - gamma*R - delta_R*P*R

Steady-state solutions (p. 9) and forward integrators (deterministic + stochastic).
"""

import numpy as np
from scipy.integrate import odeint


def rhs(state, t, r, delta_p, k, alpha, lam, gamma, delta_R):
    P, R = state
    P = max(P, 0.0)
    R = max(R, 0.0)
    dP = r * P - delta_p * P * R - k * P**2
    dR = alpha + lam * P - gamma * R - delta_R * P * R
    return [dP, dR]


def solve(params: dict, P0: float, t: np.ndarray) -> np.ndarray:
    """Deterministic ODE solution; returns array of shape (len(t), 2)."""
    R0 = params["alpha"] / params["gamma"]   # constitutive equilibrium
    state0 = [P0, R0]
    args = (params["r"], params["delta_p"], params["k"],
            params["alpha"], params["lam"], params["gamma"], params["delta_R"])
    return odeint(rhs, state0, t, args=args, rtol=1e-8, atol=1e-10)


def gillespie(params: dict, P0: float, tmax: float, seed: int = 0) -> tuple:
    """
    Exact SSA (Gillespie) for the discrete stochastic version.
    Returns (times, P_trajectory, R_trajectory).
    """
    rng = np.random.default_rng(seed)
    r, dp, k  = params["r"], params["delta_p"], params["k"]
    al, lm    = params["alpha"], params["lam"]
    gm, dR    = params["gamma"], params["delta_R"]

    P = int(round(P0))
    R = int(round(params["alpha"] / params["gamma"]))
    t = 0.0

    ts, Ps, Rs = [t], [P], [R]

    while t < tmax:
        # reaction rates
        r1 = r  * P            # P birth
        r2 = dp * P * R        # P death (host response)
        r3 = k  * P * P        # P density death
        r4 = al                # R constitutive production
        r5 = lm * P            # R induced production
        r6 = gm * R            # R decay
        r7 = dR * P * R        # R wear

        total = r1 + r2 + r3 + r4 + r5 + r6 + r7
        if total == 0:
            break

        dt = rng.exponential(1.0 / total)
        t += dt

        u = rng.uniform() * total
        cumulative = 0.0
        for rate, dP_val, dR_val in [
            (r1,  1,  0),
            (r2, -1,  0),
            (r3, -1,  0),
            (r4,  0,  1),
            (r5,  0,  1),
            (r6,  0, -1),
            (r7,  0, -1),
        ]:
            cumulative += rate
            if u < cumulative:
                P = max(0, P + dP_val)
                R = max(0, R + dR_val)
                break

        ts.append(t)
        Ps.append(P)
        Rs.append(R)

    return np.array(ts), np.array(Ps), np.array(Rs)


def initial_conditions(P0_high: float, dosefold: float, n_doses: int = 5):
    """
    Initial P for each dose (index 0 = highest dose):
      P0(i) = P0_high / dosefold^i
    """
    return [P0_high / dosefold**i for i in range(n_doses)]


def equilibria(params: dict):
    """
    Returns (P_free, R_free), (P_systemic, R_systemic), (P_hat, R_hat)
    as in the paper (p. 9).  Complex roots → NaN.
    """
    r, dp, k  = params["r"], params["delta_p"], params["k"]
    al        = params["alpha"]
    lm        = params["lam"]
    gm, dR    = params["gamma"], params["delta_R"]

    # pathogen-free
    P_free = 0.0
    R_free = al / gm

    disc = (lm * dp + gm * k - r * dR)**2 - 4 * k * dR * (al * dp - gm * r)
    if disc < 0:
        return (P_free, R_free), (np.nan, np.nan), (np.nan, np.nan)

    sq = np.sqrt(disc)
    # systemic (stable establishment)
    P_sys = (r * dR - lm * dp - gm * k + sq) / (2 * k * dR)
    R_sys = (r * dR + lm * dp + gm * k - sq) / (2 * dp * dR)

    # unstable (threshold)
    P_hat = (r * dR - lm * dp - gm * k - sq) / (2 * k * dR)
    R_hat = (r * dp + lm * dp + gm * k + sq) / (2 * dp * dR)

    return (P_free, R_free), (P_sys, R_sys), (P_hat, R_hat)
