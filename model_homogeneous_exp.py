"""
Population Count Model with Homogeneous Exponential Family
(Modified for K-th order interactions)

Author: Hideaki Shimazaki
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from tqdm import tqdm


class ModelCache:
    """Precomputed quantities for a given (N, K, h) that don't depend on theta."""

    def __init__(self, N, K, h=None):
        if h is None:
            h = lambda n: 1.0
        self.N = N
        self.K = K
        self.h = h
        ns = np.arange(N + 1)

        # log C(N, n) — (N+1,)
        self.log_binom = (sp.gammaln(N + 1)
                          - sp.gammaln(ns + 1)
                          - sp.gammaln(N - ns + 1))

        # log h(n) — (N+1,)
        self.log_h = np.log(np.array([h(n) for n in ns]))

        # C(n, k) for n=0..N, k=1..K — (N+1, K)
        self.C = np.array([[sp.comb(n, k, exact=True) for k in range(1, K + 1)]
                           for n in ns], dtype=np.float64)

        # Static part of L: log_binom + log_h — (N+1,)
        self.L_static = self.log_binom + self.log_h


def _log_probs_from_cache(cache, theta):
    """Compute (logP, logZ) using precomputed cache. theta has length K."""
    exponents = cache.C @ theta  # (N+1,)
    L = cache.L_static + exponents
    logZ = sp.logsumexp(L)
    logP = L - logZ
    return logP, logZ


def homogeneous_probabilities(N, theta, h=None):
    """
    Compute probabilities P(n) for n=0,...,N for a homogeneous exponential model.
    """
    K = len(theta)
    cache = ModelCache(N, K, h)
    logP, _ = _log_probs_from_cache(cache, theta)
    return np.exp(logP)


def log_homogeneous_probabilities(N, K, theta, h=None):
    """
    Compute log probabilities and log partition function for the K-th order model.

    The model is defined as:
        P(n) = C(N,n) * h(n) * exp(Σ_{k=1}^{min(n,K)} C(n,k)*θ_k) / Z(θ)
    """
    if len(theta) != K:
        raise ValueError(f"Length of theta ({len(theta)}) must be equal to K ({K}).")
    cache = ModelCache(N, K, h)
    return _log_probs_from_cache(cache, theta)


def compute_sufficient_statistics(ns, K):
    """
    Compute sufficient statistics S_k = Σ_i C(n_i, k) for k=1..K.
    """
    ns = np.asarray(ns)
    S = np.zeros(K)
    for k in range(1, K + 1):
        S[k - 1] = np.sum(sp.comb(ns, k))
    return S


def compute_map_gradient(N, K, S, M, h=None, q=None, theta=None, cache=None):
    """
    Compute gradient of log-posterior for MAP estimation for a K-th order model.

    The gradient is: ∇_j = S_j - M*E[C(n,j)] - θ_j/q_j for j=1..K
    """
    if cache is None:
        cache = ModelCache(N, K, h)
    logP, _ = _log_probs_from_cache(cache, theta)
    Pn = np.exp(logP)
    E_C = Pn @ cache.C  # (K,)
    return S - M * E_C - theta / q


def estimate_map_parameters(N, K, S, M, h=None, q=None, theta=None, cache=None):
    """
    Find MAP estimate of θ (length K) given sufficient statistics.
    """
    if cache is None:
        cache = ModelCache(N, K, h)

    def negative_log_posterior(th):
        _, logZ = _log_probs_from_cache(cache, th)
        ll = np.dot(S, th) - M * logZ
        prior = -0.5 * np.sum(th**2 / q)
        return -(ll + prior)

    def gradient_negative_log_posterior(th):
        return -compute_map_gradient(N, K, S, M, h, q, th, cache=cache)

    res = minimize(negative_log_posterior, theta,
                   jac=gradient_negative_log_posterior,
                   method='L-BFGS-B',
                   options={'disp': False})
    return res


def compute_posterior_covariance(N, K, theta_map, h=None, q=None, M=None, cache=None):
    """
    Compute posterior covariance matrix for a K-th order model.
    """
    if cache is None:
        cache = ModelCache(N, K, h)
    logP, _ = _log_probs_from_cache(cache, theta_map)
    Pn = np.exp(logP)

    E1 = Pn @ cache.C  # (K,)
    E2 = cache.C.T @ (Pn[:, None] * cache.C)  # (K, K)
    Cov_C = E2 - np.outer(E1, E1)

    H = M * Cov_C + np.diag(1.0 / q)
    Sigma = np.linalg.inv(H)
    return Sigma


def em_update(N, samples, h=None, K=None, q_init=None, theta0=None, max_iter=100, tol=1e-6):
    """
    Empirical-Bayes EM algorithm for a K-th order model.

    Returns:
        tuple: (theta_map, Sigma, q, res)
    """
    if h is None:
        h = lambda n: 1.0
    if K is None:
        K = N
    S = compute_sufficient_statistics(samples, K)
    M = len(samples)
    if q_init is None:
        q_init = 1 * np.ones(K)
    if theta0 is None:
        theta0 = np.zeros(K)
    q = q_init.copy()
    theta_est = theta0.copy()

    cache = ModelCache(N, K, h)

    for itr in tqdm(range(max_iter), desc="EM iteration"):
        # E-step: Find MAP estimate
        res = estimate_map_parameters(N, K, S, M, h, q, theta_est, cache=cache)
        theta_map = res.x

        # Compute posterior variance
        Sigma = compute_posterior_covariance(N, K, theta_map, h, q, M, cache=cache)
        var_theta = np.diag(Sigma)

        # M-step: Update q
        q_new = theta_map**2 + var_theta
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q, theta_est = q_new, theta_map

    return theta_map, Sigma, q, res


def estimate_ml_parameters(N, samples, h=None, K=None, theta0=None):
    """
    Estimate ML parameters for a K-th order model.
    """
    if h is None:
        h = lambda n: 1.0
    if K is None:
        K = N
    S = compute_sufficient_statistics(samples, K)
    M = len(samples)
    if theta0 is None:
        theta0 = np.zeros(K)

    cache = ModelCache(N, K, h)

    def negative_log_likelihood(th):
        _, logZ = _log_probs_from_cache(cache, th)
        return -(np.dot(S, th) - M * logZ)

    def gradient_negative_log_likelihood(th):
        logP, _ = _log_probs_from_cache(cache, th)
        Pn = np.exp(logP)
        E_C = Pn @ cache.C
        return -(S - M * E_C)

    res = minimize(negative_log_likelihood, theta0,
                   jac=gradient_negative_log_likelihood,
                   method='L-BFGS-B',
                   options={'disp': False})
    return res


def sample_counts(N, theta, h=None, size=1):
    """Sample counts from the model."""
    if h is None:
        h = lambda n: 1.0
    probs = homogeneous_probabilities(N, theta, h)
    return np.random.choice(N + 1, size=size, p=probs)


def sample_patterns(N, theta, h=None, size=1):
    """Sample binary patterns from the model."""
    if h is None:
        h = lambda n: 1.0
    counts = sample_counts(N, theta, h, size)
    patterns = np.zeros((size, N), dtype=int)
    for i, n in enumerate(counts):
        if n > 0:
            idx = np.random.choice(N, n, replace=False)
            patterns[i, idx] = 1
    return patterns


# ----------------------------
# Usage example
# ----------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # --------------------------
    # Parameters
    N = 10  # System size
    K = 4   # Maximum order of interaction (K ≤ N)
    h = lambda n: 1.0  # Base rate function (constant)

    # --------------------------
    # Generate synthetic data
    true_theta_full = np.array([-2.5, 0.5, -0.2, 0.1, 0, 0, 0, 0, 0, 0])
    samples = sample_counts(N, true_theta_full, h, size=1000)

    print(f"\nSystem size N = {N}")
    print(f"Fitting up to K = {K}-th order interactions.\n")

    print("True full θ (N-dim):", true_theta_full)
    print("\nFirst 20 samples:", samples[:20])

    # --------------------------
    print(f"\nFitting using maximum likelihood (K={K})...")
    # Run ML estimation
    result_ml = estimate_ml_parameters(N, samples, h, K)
    print("ML‐Estimated θ (K-dim):", result_ml.x)
    print("Log‐likelihood:", -result_ml.fun)

    # --------------------------
    print(f"\nFitting using MAP with EM (K={K})...")
    # Run EM algorithm
    theta_map, Sigma, q, res_map = em_update(N, samples, h, K)
    print("MAP‐Estimated θ (K-dim):", theta_map)
    print("Final learned q (K-dim):", q)
    print("Log‐posterior:", -res_map.fun)

    # --------------------------
    # Compare probabilities
    print("\n--- Probabilities P(n) ---")
    true_probs = homogeneous_probabilities(N, true_theta_full, h)
    ml_probs = homogeneous_probabilities(N, result_ml.x, h)
    map_probs = homogeneous_probabilities(N, theta_map, h)

    print("True :", true_probs)
    print("ML   :", ml_probs)
    print("MAP  :", map_probs)
