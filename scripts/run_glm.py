#!/usr/bin/env python3
"""Fit GLM: log(denv_titer) ~ group * dose * time (supplementary analysis)."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.preprocessing import preprocess
from denv_wmel.glm import fit_glm

pathlib.Path("results").mkdir(exist_ok=True)

df = preprocess()
result, anova = fit_glm(df)

txt = result.summary().as_text() + "\n\n" + anova.to_string()
pathlib.Path("results/glm_summary.txt").write_text(txt)
print(txt)
