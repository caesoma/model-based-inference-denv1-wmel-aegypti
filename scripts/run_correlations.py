#!/usr/bin/env python3
"""Compute DENV-1 / Wolbachia correlations and render Fig 3."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.preprocessing import preprocess
from denv_wmel.correlations import compute_correlations
from denv_wmel.plots import fig3

pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("figures").mkdir(exist_ok=True)

df = preprocess()
corr = compute_correlations(df)
corr.to_csv("results/correlations_table.csv", index=False)
print(corr.to_string(index=False))

fig3(df, corr)
