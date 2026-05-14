"""
Reproduce Figures 1–5 from Souto-Maior et al. 2018.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DOSE_ORDER  = [4, 5, 6, 7, 8]
TIMEPOINTS  = [3, 7, 14]

# color map matching the paper: each dose gets a color; TET = vivid, BR = light
DOSE_COLORS_TET = {4: "tab:cyan",  5: "black",    6: "tab:orange",
                   7: "tab:blue",  8: "tab:red"}
DOSE_COLORS_BR  = {4: "#b2dfdb",  5: "#b0bec5",  6: "#ffe0b2",
                   7: "#90caf9",  8: "#a5d6a7"}

JITTER = 0.15


def _jitter(arr, rng=None, scale=JITTER):
    if rng is None:
        rng = np.random.default_rng(0)
    return arr + rng.uniform(-scale, scale, size=len(arr))


def _safe_log10(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.where(arr > 0, np.log10(arr.astype(float)), np.nan)
    return out


# -------------------------------------------------------------------------
# Fig 1  – DENV-1 viral titers
# -------------------------------------------------------------------------
def fig1(df: pd.DataFrame, out: str = "figures/fig1.pdf"):
    tet = df[df["group"].str.upper().str.contains("TET")]
    br  = df[~df["group"].str.upper().str.contains("TET")]

    doses = DOSE_ORDER
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey=True, sharex=True)
    axes = axes.flatten()
    rng = np.random.default_rng(1)

    for ax_i, dose_exp in enumerate(doses):
        ax = axes[ax_i]
        for dpi in TIMEPOINTS:
            t_sub = tet[(tet["dose_exp"] == dose_exp) & (tet["dpi"] == dpi)]
            b_sub = br[(br["dose_exp"]  == dose_exp) & (br["dpi"]  == dpi)]

            y_t = t_sub["denv_titer"].values
            y_b = b_sub["denv_titer"].values

            # plot zeros at a special y level
            y_lim_low = -0.5
            y_t_plot  = np.where(y_t > 0, _safe_log10(y_t), y_lim_low)
            y_b_plot  = np.where(y_b > 0, _safe_log10(y_b), y_lim_low)

            x_t = _jitter(np.full(len(y_t), dpi), rng)
            x_b = _jitter(np.full(len(y_b), dpi), rng)

            ax.scatter(x_b, y_b_plot, c=DOSE_COLORS_BR[dose_exp], s=30, alpha=0.7, zorder=2)
            ax.scatter(x_t, y_t_plot, c=DOSE_COLORS_TET[dose_exp], s=30, alpha=0.9, zorder=3)

        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"$10^{dose_exp}$ TCID$_{{50}}$", fontsize=9)
        ax.set_xticks(TIMEPOINTS)

    axes[-1].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("time (days)")
    for i in range(0, len(axes), 2):
        axes[i].set_ylabel("viral titers (log$_{10}$)")

    fig.suptitle("DENV-1 viral titers", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# -------------------------------------------------------------------------
# Fig 2  – Wolbachia titers (wMelBR only)
# -------------------------------------------------------------------------
def fig2(df: pd.DataFrame, out: str = "figures/fig2.pdf"):
    br = df[~df["group"].str.upper().str.contains("TET")]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey=True, sharex=True)
    axes = axes.flatten()
    rng = np.random.default_rng(2)

    for ax_i, dose_exp in enumerate(DOSE_ORDER):
        ax = axes[ax_i]
        for dpi in TIMEPOINTS:
            sub = br[(br["dose_exp"] == dose_exp) & (br["dpi"] == dpi)]
            y = sub["wmel_titer"].values
            y_plot = np.where(y > 0, _safe_log10(y), np.nan)
            x = _jitter(np.full(len(y), dpi), rng)
            ax.scatter(x, y_plot, c="green", s=30, alpha=0.8)

        ax.set_title(f"$10^{dose_exp}$ TCID$_{{50}}$", fontsize=9)
        ax.set_xticks(TIMEPOINTS)

    axes[-1].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("time (days)")
    for i in range(0, len(axes), 2):
        axes[i].set_ylabel("Wolbachia titers (log$_{10}$)")

    fig.suptitle("Wolbachia titers (wMelBR)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# -------------------------------------------------------------------------
# Fig 3  – DENV-1 vs Wolbachia correlation
# -------------------------------------------------------------------------
def fig3(df: pd.DataFrame, corr_table: pd.DataFrame,
         out: str = "figures/fig3.pdf"):
    br = df[~df["group"].str.upper().str.contains("TET")]
    pos = br[(br["denv_titer"] > 0) & (br["wmel_titer"] > 0)]

    x_all = np.log10(pos["denv_titer"].astype(float))
    y_all = np.log10(pos["wmel_titer"].astype(float))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A – all data
    ax_a.scatter(x_all, y_all, c="black", s=25, alpha=0.7)
    row = corr_table[corr_table["label"] == "all"].iloc[0]
    xfit = np.linspace(x_all.min(), x_all.max(), 100)
    yfit = row["slope"] * xfit + row["intercept"]
    ax_a.plot(xfit, yfit, "k-", lw=1.5)
    ax_a.set_title(f"correlation: {row['r']:.3f}, slope: {row['slope']:.2f}\n"
                   f"p-value = {row['p_value']:.3f}", fontsize=9)
    ax_a.set_xlabel("DENV-1 viral titers")
    ax_a.set_ylabel("wMel titers")

    # Panel B – per dose
    dose_colors = ["tab:cyan", "tab:orange", "tab:blue", "tab:red"]
    for ci, dose_exp in enumerate([5, 6, 7, 8]):
        sub = pos[pos["dose_exp"] == dose_exp]
        if sub.empty:
            continue
        xd = np.log10(sub["denv_titer"].astype(float))
        yd = np.log10(sub["wmel_titer"].astype(float))
        ax_b.scatter(xd, yd, c=dose_colors[ci], s=25, alpha=0.8,
                     label=f"$10^{dose_exp}$")
        if len(xd) >= 4:
            slope, intercept, *_ = stats.linregress(xd, yd)
            xs = np.linspace(xd.min(), xd.max(), 50)
            ax_b.plot(xs, slope * xs + intercept, color=dose_colors[ci], lw=1.2)

    ax_b.legend(title="dose", fontsize=8)
    ax_b.set_xlabel("DENV-1 viral titers")
    ax_b.set_ylabel("wMel titers")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# -------------------------------------------------------------------------
# Fig 4  – Forward simulation
# -------------------------------------------------------------------------
def fig4(det_results, stoch_results, pseudo_data,
         out: str = "figures/fig4.pdf"):
    t_det, traj_det = det_results
    colors = ["tab:cyan", "black", "tab:orange", "tab:blue", "tab:red"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A – stochastic
    for dose_idx, trajs in enumerate(stoch_results):
        for ts, Ps in trajs:
            Parr = np.array(Ps, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y = np.where(Parr > 0, np.log10(Parr), np.nan)
            ax_a.plot(ts, y, color=colors[dose_idx], lw=0.5, alpha=0.3)

    pd_ = pseudo_data
    for dose_idx in range(len(stoch_results)):
        sub = pd_[pd_["dose_idx"] == dose_idx]
        y = np.where(sub["P"] > 0, np.log10(sub["P"].astype(float) + 1e-3), -0.5)
        ax_a.scatter(sub["time"], y, c=colors[dose_idx], s=20, zorder=5)

    ax_a.set_xlabel("time")
    ax_a.set_ylabel("viral titers (log$_{10}$)")
    ax_a.set_title("A  stochastic")

    # Panel B – deterministic
    for dose_idx, traj in enumerate(traj_det):
        P = traj[:, 0]
        y = np.where(P > 0, np.log10(P + 1e-3), -0.5)
        ax_b.plot(t_det, y, color=colors[dose_idx], lw=2)

    pd_ = pseudo_data
    for dose_idx in range(len(stoch_results)):
        # one sample trajectory per dose for deterministic panel
        sub_0 = pseudo_data[
            (pseudo_data["dose_idx"] == dose_idx) & (pseudo_data["traj_idx"] == 0)
        ]
        y = np.where(sub_0["P"] > 0,
                     np.log10(sub_0["P"].astype(float) + 1e-3), -0.5)
        ax_b.scatter(sub_0["time"], y, c="gray", s=20, zorder=5, alpha=0.6)

    ax_b.set_xlabel("time")
    ax_b.set_ylabel("viral titers (log$_{10}$)")
    ax_b.set_title("B  deterministic")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# -------------------------------------------------------------------------
# Fig 5A – Model fit trajectories with credible band over data
# -------------------------------------------------------------------------
def fig5_trajectories(idata, df: pd.DataFrame, out: str = "figures/fig5a.pdf"):
    """
    For each posterior sample, integrate the ODE and collect P(t) trajectories.
    Plot median + 95% credible band for wMelTET (gray) and wMelBR (green),
    overlaid on the raw data points.
    """
    from .model import solve
    from .inference import DOSE_ORDER, TIMEPOINTS

    t_plot = np.linspace(0, 21, 500)
    post   = idata.posterior

    def _get(name):
        return post[name].values.flatten()

    r_s       = _get("r");        k_s    = _get("k")
    dp_s      = _get("delta_p");  al_s   = _get("alpha");  gm_s = _get("gamma")
    lT_s      = _get("lam_TET");  lW_s   = _get("lam_wMEL")
    dRT_s     = _get("delta_R_TET"); dRW_s = _get("delta_R_wMEL")
    P0T_s     = _get("P0_TET");   P0W_s  = _get("P0_wMEL")
    dfT_s     = _get("dosefold_TET"); dfW_s = _get("dosefold_wMEL")

    n_samples = len(r_s)
    # thin to at most 200 samples for speed
    idx_thin  = np.random.default_rng(0).choice(n_samples,
                    size=min(200, n_samples), replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    tet_color  = "#888888"   # gray for wMelTET
    wmel_color = "#2ca02c"   # green for wMelBR

    for group, lam_s, dR_s, P0_s, df_s, color in [
        ("TET",  lT_s,  dRT_s, P0T_s, dfT_s, tet_color),
        ("wMEL", lW_s,  dRW_s, P0W_s, dfW_s, wmel_color),
    ]:
        for dose_idx, dose_exp in enumerate(DOSE_ORDER):
            dose_trajs = []
            for si in idx_thin:
                params = dict(r=float(r_s[si]), delta_p=float(dp_s[si]),
                              k=float(k_s[si]), alpha=float(al_s[si]),
                              lam=float(lam_s[si]), gamma=float(gm_s[si]),
                              delta_R=float(dR_s[si]))
                idx_d = 8 - dose_exp
                P0    = float(P0_s[si]) / (float(df_s[si]) ** idx_d)
                traj  = solve(params, P0, t_plot)
                P     = np.maximum(traj[:, 0], 1e-3)
                dose_trajs.append(np.log10(P))

            trajs_arr = np.array(dose_trajs)
            med  = np.median(trajs_arr, axis=0)
            lo   = np.percentile(trajs_arr, 2.5,  axis=0)
            hi   = np.percentile(trajs_arr, 97.5, axis=0)

            ax.plot(t_plot, med, color=color, lw=1.5, alpha=0.8)
            ax.fill_between(t_plot, lo, hi, color=color, alpha=0.15)

    # overlay raw data
    tet = df[df["group"].str.upper().str.contains("TET")]
    br  = df[~df["group"].str.upper().str.contains("TET")]
    for grp_df, color in [(tet, tet_color), (br, wmel_color)]:
        for dose_idx, dose_exp in enumerate(DOSE_ORDER):
            for dpi in TIMEPOINTS:
                sub = grp_df[(grp_df["dose_exp"] == dose_exp) & (grp_df["dpi"] == dpi)]
                y   = sub["denv_titer"].values
                y_plot = np.where(y > 0, np.log10(y.astype(float)), -0.5)
                ax.scatter(np.full(len(y), dpi), y_plot,
                           c=color, s=20, alpha=0.6, zorder=3)

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("viral titers (log$_{10}$)")
    ax.set_title("Model fit — wMelTET (gray) vs wMelBR (green)")
    ax.set_xlim(0, 21)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# -------------------------------------------------------------------------
# Fig 5B – Posterior parameter distributions
# -------------------------------------------------------------------------
def fig5_posteriors(idata, out: str = "figures/fig5b.pdf"):
    params = ["r", "k", "delta_p", "alpha", "lam_TET", "lam_wMEL",
              "delta_R_TET", "delta_R_wMEL", "gamma",
              "P0_TET", "P0_wMEL", "dosefold_TET", "dosefold_wMEL"]
    existing = [p for p in params if p in idata.posterior]

    ncols = 3
    nrows = int(np.ceil(len(existing) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.5))
    axes = axes.flatten()

    for i, p in enumerate(existing):
        samples = idata.posterior[p].values.flatten()
        # green for wMEL-specific params, salmon for TET-specific and shared
        color = "green" if "wMEL" in p else "salmon"
        axes[i].hist(samples, bins=50, color=color, alpha=0.7, density=True)
        axes[i].set_title(p, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Posterior parameter distributions", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
