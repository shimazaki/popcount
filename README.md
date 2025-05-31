# Population Count Models

This repository contains implementations of various population count models, including homogeneous exponential models with both EM and Bayesian inference methods.

## Features

- Homogeneous exponential model implementation
- EM-based parameter estimation
- Bayesian inference with MAP estimation
- Sample generation utilities
- Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/popcount.git
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
from model_homogeneous_exp_em import homogeneous_probabilities, em_update_q_from_sufficient

# Define parameters
N = 10
true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

def h(n):
    return 1

# Compute probabilities
probs = homogeneous_probabilities(N, true_theta, h)

# Fit model using EM
q = np.ones(N) * 10.0
theta0 = np.zeros(N)
q, theta_map, Sigma, res = em_update_q_from_sufficient(N, samples, h, q, theta0)
```

### Example: Homogeneous Exponential Model with Bayesian Inference

```python
from model_homogeneous_exp_bayes import fit_theta_map_sufficient

# Fit model using MAP estimation
res = fit_theta_map_sufficient(N, S, M, h, q, theta0)
theta_map = res.x
```

## Project Structure

- `model_homogeneous_exp_em.py`: EM-based parameter estimation
- `model_homogeneous_exp_bayes.py`: Bayesian inference implementation
- `generate_homogeneous_exp_samples.py`: Sample generation utilities
- `figure_homogeneous_exp.py`: Visualization tools
- `example_homogeneous_exp.py`: Basic usage examples
- `example_homogeneous_exp_bayes.py`: Bayesian inference examples

## License

This project is licensed under the MIT License - see the LICENSE file for details. 