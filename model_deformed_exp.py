"""
Population Count Model with Homogeneous Deformed Exponential Family

Author: Hideaki Shimazaki
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from tqdm import tqdm


def q_exponential(z, q):
    """
    q-exponential function: exp_q(z) = [1 + (1 - q) * z]_+^{1 / (1 - q)}
    When q=1, this recovers the standard exponential: exp(z)
    \gamma = 1 - q
    """
    if q == 1:
        return np.exp(z)
    else:
        base = 1 + (1 - q) * z
        return np.where(base > 0, base ** (1 / (1 - q)), 0.0)


def homogeneous_deformed_probabilities(N, K, theta, q=1.0, h=None):
    """
    Compute probabilities P_q(n) for a deformed homogeneous exponential family.
    
    Args:
        N (int): System size.
        K (int): Maximum interaction order.
        theta (array): Parameter vector (length K).
        q (float): Deformation parameter (q=1 recovers exponential family).
        h (callable, optional): Base rate function. Default is h(n)=1.
    
    Returns:
        array: Probabilities P_q(n) for n = 0, ..., N.
    """
    if h is None:
        h = lambda n: 1.0
    if len(theta) != K:
        raise ValueError(f"Expected theta of length {K}, got {len(theta)}.")
    
    theta_int = np.concatenate(([0.0], theta))
    ns = np.arange(N + 1)

    # Compute log binomial coefficients
    log_binom = sp.gammaln(N + 1) - sp.gammaln(ns + 1) - sp.gammaln(N - ns + 1)
    
    # Compute base rate terms
    log_h = np.log([h(n) for n in ns])
    
    # Compute interaction terms
    exponents = np.array([
        sum(sp.comb(n, k) * theta_int[k] for k in range(1, min(n, K) + 1))
        for n in ns
    ])
    
    # Apply q-exponential
    unnormalized = np.exp(log_binom + log_h) * q_exponential(exponents, q)

    # Normalize
    Z_q = np.sum(unnormalized)
    P_q = unnormalized / Z_q
    return P_q


if __name__ == "__main__":
    N = 10       # System size
    K = 2        # Up to pairwise interactions
    theta = [-2, 0.5]  # Some arbitrary parameters
    q = 1      # Deformation parameter
    P_q = homogeneous_deformed_probabilities(N, K, theta, q)
    print(P_q)

