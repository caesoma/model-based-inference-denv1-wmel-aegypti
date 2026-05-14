#!/usr/bin/env python3
"""Forward simulation at paper parameters: stochastic + deterministic (Fig 4)."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.simulation import run_deterministic, run_stochastic, sample_pseudo_data
from denv_wmel.plots import fig4, fig1, fig2
from denv_wmel.preprocessing import preprocess

pathlib.Path("figures").mkdir(exist_ok=True)

print("Running deterministic simulation…")
det = run_deterministic()

print("Running stochastic ensemble (this takes ~1 min)…")
stoch = run_stochastic()
pseudo = sample_pseudo_data(stoch)

fig4(det, stoch, pseudo)

df = preprocess()
fig1(df)
fig2(df)
