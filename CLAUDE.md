# CLAUDE.md — popcount

## Project Overview

Population count (popcount) models for neural spike data. Implements homogeneous exponential family models and alternating shrinking higher-order interaction models for sparse neural population activity.

Based on: Rodríguez-Domínguez & Shimazaki (2023), "Alternating Shrinking Higher-order Interactions for Sparse Neural Population Activity" (arXiv:2308.13257).

## Environment

- Conda env: `popcount` (Python 3.6.4)
- Activate: `conda activate popcount`
- Dependencies: numpy, scipy, matplotlib, tqdm

## Running Tests

```bash
conda activate popcount
python -m pytest -v
```

Note: `run_tests.py` requires `pytest-cov` which may not be installed. Use `python -m pytest -v` directly.

## Project Structure

### Models
- `model_homogeneous_exp.py` — Homogeneous exponential family model (ML/MAP estimation, EM, sampling)
- `model_alternating_shrinking.py` — Alternating shrinking model with Gibbs sampling
- `model_deformed_exp.py` — Deformed exponential model
- `model_dichotomized_gaussian.py` — Dichotomized Gaussian model
- `model_leaky_integrate_and_fire.py` — Leaky integrate-and-fire model

### Figure Scripts
- `fig_*.py` — Each generates a specific figure for the paper/analysis

### Tests
- `test_homogeneous_exp.py` — Tests for homogeneous exponential model
- `test_model_alternating_shrinking.py` — Tests for alternating shrinking model
- `test_model_homogeneous_exp.py` — Tests for homogeneous exponential model (expanded)

## Key API Patterns

The core model function signature pattern:
```python
P(n) = C(N,n) * h(n) * exp(sum_k C(n,k) * theta_k) / Z(theta)
```
- `N` — system size (number of neurons)
- `K` — model order (number of theta parameters)
- `theta` — natural parameters array
- `h` — base rate function (callable, default h(n)=1)

## Known Issues

- Python 3.6 lacks `math.comb` (added in 3.8); use `scipy.special.comb` instead
- Some test failures exist due to API signature mismatches between tests and model code
