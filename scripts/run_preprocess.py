#!/usr/bin/env python3
"""Preprocess raw CT data and save clean titer table."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.preprocessing import preprocess

df = preprocess()
pathlib.Path("results").mkdir(exist_ok=True)
df.to_csv("results/data_clean.csv", index=False)
print(f"Saved results/data_clean.csv ({len(df)} rows)")
print(df.groupby(["group", "dose_exp", "dpi"])[["denv_titer", "wmel_titer"]].describe().to_string())
