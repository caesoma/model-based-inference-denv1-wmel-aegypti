# model-based-inference-denv1-wmel-aegypti
Data and code for publication 'Model-based inference from multiple dose, time course data reveals Wolbachia effects on infection profiles of type 1 dengue virus in Aedes aegypti'

[![DOI](https://zenodo.org/badge/123297378.svg)](https://zenodo.org/badge/latestdoi/123297378)

---

## Reproduction

### Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (dependency manager)

Install all dependencies into a local virtual environment:

```bash
uv sync
```

Key libraries (see `pyproject.toml` for full list):

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data wrangling |
| `scipy` | ODE integration (`odeint`), statistics (`linregress`) |
| `statsmodels` | GLM / OLS |
| `pymc` (v5) | Bayesian MCMC inference |
| `pytensor` | Wraps scipy ODE as a non-differentiable op for PyMC |
| `arviz` | MCMC diagnostics and posterior summaries |
| `matplotlib` | All figures |

### Running the analysis

Run scripts in order. All commands use `uv run` to ensure the project virtual environment is used.

**1. Preprocess raw qPCR data → normalised integer titers**
```bash
uv run python scripts/run_preprocess.py
# output: results/data_clean.csv
```

**2. DENV-1 / Wolbachia correlations (Fig 3, Table 1)**
```bash
uv run python scripts/run_correlations.py
# output: results/correlations_table.csv, figures/fig3.pdf
```

**3. Forward simulation at paper parameters (Figs 1, 2, 4)**
```bash
uv run python scripts/run_simulation.py
# output: figures/fig1.pdf, fig2.pdf, fig4.pdf
```

**4. GLM: log(titer) ~ group × dose × time (supplementary)**
```bash
uv run python scripts/run_glm.py
# output: results/glm_summary.txt
```

**5. Bayesian MCMC inference — ODE model fit (Fig 5)**
```bash
# Quick test (~30 min, 2 chains)
uv run python scripts/run_inference.py --draws 200 --tune 200 --chains 2

# Full run matching the paper (~several hours, 4 chains)
uv run python scripts/run_inference.py --draws 2000 --tune 2000 --chains 4

# output: results/posterior.nc, results/posterior_summary.csv,
#         figures/fig5a.pdf, figures/fig5b.pdf
```

> **Note:** Each MCMC step integrates the ODE numerically (~2 s/draw on a single core). Metropolis-Hastings is used instead of NUTS because the ODE solver is wrapped as a non-differentiable op.

### Tests

```bash
uv run python -m pytest tests/ -v
```

### Output files

| File | Contents |
|---|---|
| `results/data_clean.csv` | Preprocessed per-mosquito titer table |
| `results/correlations_table.csv` | Pearson R², slope, p-value per stratum |
| `results/glm_summary.txt` | OLS ANOVA table |
| `results/posterior.nc` | Full MCMC chains (arviz NetCDF) |
| `results/posterior_summary.csv` | Posterior means, SD, HDI, r-hat |
| `figures/fig1.pdf` | DENV-1 viral titers by dose and time |
| `figures/fig2.pdf` | Wolbachia titers (wMelBR) |
| `figures/fig3.pdf` | DENV-1 vs Wolbachia correlation |
| `figures/fig4.pdf` | Forward simulation (stochastic + deterministic) |
| `figures/fig5a.pdf` | Model fit trajectories with credible bands |
| `figures/fig5b.pdf` | Posterior parameter distributions |
