# Population Count Model

This repository contains code for fitting a population count model using maximum likelihood and Bayesian methods.

## Model

The model is defined by the probability mass function:

    P(n) = C(N,n) * h(n) * exp[ Σ_{k=1}^n C(n,k) * θ_k ] / Z(θ)

where:
- N is the total number of items
- n is the observed count (0 ≤ n ≤ N)
- C(n,k) is the binomial coefficient
- h(n) is a base rate function
- θ_k are the natural parameters
- Z(θ) is the partition function

## Features

- Homogeneous exponential model implementation
- EM-based parameter estimation
- MAP estimation with and without EM
- Sample generation utilities
- Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shimazaki/popcount.git
cd popcount
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Example: Homogeneous Exponential Model with EM

```python
import numpy as np
from model_homogeneous_exp_em import homogeneous_probabilities, em_update
import generate_homogeneous_exp_samples as generate_samples

# Define parameters
N = 10
theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

# Define base rate function
def h(n):
    return 1  # here h(n) ≡ 1

# Generate samples
sample_size = 5000
samples = generate_samples.sample_counts(N, theta, h, size=sample_size)

# Compute probabilities
probs = homogeneous_probabilities(N, theta, h)

# Fit model using EM
q = np.ones(N) * 10.0  # prior variances
theta0 = np.zeros(N)
q, theta_map, Sigma, res = em_update(N, samples, h, q, theta0)
```

### Example: Basic MAP Estimation

```python
import numpy as np
from model_homogeneous_exp import estimate_map_parameters
import generate_homogeneous_exp_samples as generate_samples

# Define parameters
N = 10
theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

# Define base rate function
def h(n):
    return 1  # here h(n) ≡ 1

# Generate samples
sample_size = 5000
samples = generate_samples.sample_counts(N, theta, h, size=sample_size)

# Fit model using MAP estimation
theta0 = np.zeros(N)
result = estimate_map_parameters(N, samples, h, theta0)
theta_map = result.x
```

## Project Structure

- `model_homogeneous_exp_em.py`: EM-based parameter estimation with MAP
- `model_homogeneous_exp.py`: Basic MAP estimation
- `generate_homogeneous_exp_samples.py`: Sample generation utilities
- `figure_homogeneous_exp.py`: Visualization tools
- `example_homogeneous_exp.py`: Basic MAP estimation examples
- `example_homogeneous_exp_bayes.py`: EM-based estimation examples

## License

This project is licensed under the MIT License - see the LICENSE file for details. 