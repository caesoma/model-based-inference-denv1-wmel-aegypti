#!/usr/bin/env python3
"""
Bayesian MCMC inference of ODE parameters (Fig 5).

Usage:
  uv run python scripts/run_inference.py [--draws N] [--chains N] [--tune N]

Note: ODE integration makes each step slow (~2 s/draw on a single core).
  Full reproduction (--draws 2000 --tune 2000 --chains 4) takes several hours.
  A quick test run (--draws 200 --tune 200 --chains 2) takes ~15 min.
"""

import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import arviz as az
from denv_wmel.preprocessing import preprocess
from denv_wmel.inference import build_and_sample
from denv_wmel.plots import fig5_posteriors, fig5_trajectories

parser = argparse.ArgumentParser()
parser.add_argument("--draws",  type=int, default=2000)
parser.add_argument("--tune",   type=int, default=2000)
parser.add_argument("--chains", type=int, default=4)
args = parser.parse_args()

pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("figures").mkdir(exist_ok=True)

df = preprocess()
idata = build_and_sample(df, draws=args.draws, tune=args.tune, chains=args.chains)

idata.to_netcdf("results/posterior.nc")
summary = az.summary(idata)
summary.to_csv("results/posterior_summary.csv")
print(summary)

fig5_posteriors(idata)
fig5_trajectories(idata, df)
