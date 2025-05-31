"""
Population Count Model with Homogeneous Exponential Family

Author: Hideaki Shimazaki
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from tqdm import tqdm

def homogeneous_probabilities(N, theta, h):
    """
    Compute probabilities P(n) for all n=0..N using the homogeneous model.
    
    Args:
        N (int): Maximum count value
        theta (array): Model parameters θ_k for k=1..N
        h (callable): Base rate function h(n)
    
    Returns:
        array: P(n) for n=0..N
    """
    logP, _ = log_homogeneous_probabilities(N, theta, h)
    return np.exp(logP)

def log_homogeneous_probabilities(N, theta, h):
    """
    Compute log probabilities and log partition function for the homogeneous model.
    
    The model is defined as:
        P(n) = C(N,n) * h(n) * exp(Σ_{k=1}^n C(n,k)*θ_k) / Z(θ)
    
    Args:
        N (int): Maximum count value
        theta (array): Model parameters θ_k for k=1..N
        h (callable): Base rate function h(n)
    
    Returns:
        tuple: (logP, logZ) where
            logP[n] = log P(n) for n=0..N
            logZ = log partition function Z(θ)
    """
    # Pad theta with 0 for k=0
    theta_int = np.concatenate(([0.0], theta))
    ns = np.arange(N+1)

    # Compute log binomial coefficients C(N,n)
    log_binom = (sp.gammaln(N+1)
               - sp.gammaln(ns+1)
               - sp.gammaln(N-ns+1))
    
    # Compute log base rates h(n)
    log_h = np.log([h(n) for n in ns])
    
    # Compute exponent terms Σ_{k=1}^n C(n,k)*θ_k
    exponents = np.array([
        sum(sp.comb(n, k) * theta_int[k] for k in range(1, n+1))
        for n in ns
    ])

    # Combine terms and normalize
    L = log_binom + log_h + exponents
    logZ = sp.logsumexp(L)  # Compute log partition function
    logP = L - logZ         # Normalize to get log probabilities
    return logP, logZ

def compute_sufficient_statistics(ns, N):
    """
    Compute sufficient statistics S_k = Σ_i C(n_i, k) for k=1..N.
    
    Args:
        ns (array): Observed counts n_i
        N (int): Maximum count value
    
    Returns:
        array: S[k-1] = Σ_i C(n_i, k) for k=1..N
    """
    S = np.zeros(N)
    for n in ns:
        for k in range(1, N+1):
            S[k-1] += sp.comb(n, k)
    return S

def compute_map_gradient(N, S, M, h, q, theta):
    """
    Compute gradient of log-posterior for MAP estimation.
    
    The gradient is:
        ∇_j = S_j - M*E[C(n,j)] - θ_j/q_j
    
    Args:
        N (int): Maximum count value
        S (array): Sufficient statistics
        M (int): Number of samples
        h (callable): Base rate function
        q (array): Prior variances
        theta (array): Current parameter values
    
    Returns:
        array: Gradient vector
    """
    logP, _ = log_homogeneous_probabilities(N, theta, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)
    
    # Compute expected sufficient statistics
    C = np.array([[sp.comb(n, k) for k in range(1, N+1)] for n in ns])
    E_C = Pn @ C
    
    return S - M*E_C - theta/q

def estimate_map_parameters(N, S, M, h, q, theta0=None):
    """
    Find MAP estimate of θ given sufficient statistics.
    
    Args:
        N (int): Maximum count value
        S (array): Sufficient statistics
        M (int): Number of samples
        h (callable): Base rate function
        q (array): Prior variances
        theta0 (array, optional): Initial parameter values
    
    Returns:
        OptimizeResult: Result from scipy.optimize.minimize
    """
    if theta0 is None:
        theta0 = np.zeros(N)

    def negative_log_posterior(th):
        """Negative log-posterior (objective function)"""
        logP, logZ = log_homogeneous_probabilities(N, th, h)
        ll = np.dot(S, th) - M*logZ
        prior = -0.5 * np.sum(th**2 / q)
        return -(ll + prior)

    def gradient_negative_log_posterior(th):
        """Gradient of negative log-posterior"""
        return -compute_map_gradient(N, S, M, h, q, th)

    res = minimize(negative_log_posterior, theta0, 
                  jac=gradient_negative_log_posterior, 
                  method='BFGS', 
                  options={'disp':False})
    return res

def compute_posterior_covariance(N, theta_map, h, q, M):
    """
    Compute posterior covariance matrix using Laplace approximation.
    
    The covariance is approximated as:
        H = M*Cov_C + diag(1/q)
        Σ ≈ H^{-1}
    
    Args:
        N (int): Maximum count value
        theta_map (array): MAP estimate of θ
        h (callable): Base rate function
        q (array): Prior variances
        M (int): Number of samples
    
    Returns:
        array: Posterior covariance matrix
    """
    logP, _ = log_homogeneous_probabilities(N, theta_map, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)

    # Compute covariance of sufficient statistics
    C = np.array([[sp.comb(n, k) for k in range(1, N+1)] for n in ns])
    E1 = Pn @ C
    E2 = C.T @ (Pn[:,None] * C)
    Cov_C = E2 - np.outer(E1, E1)

    # Compute Hessian and invert
    H = M * Cov_C + np.diag(1.0/q)
    Sigma = np.linalg.inv(H)
    return Sigma

def em_update(N, samples, h, q_init, theta0=None, max_iter=100, tol=1e-6):
    """
    Empirical-Bayes EM algorithm to update prior variances q_j.
    
    The algorithm alternates between:
        E-step: Find MAP estimate θ_map given current q
        M-step: Update q_j = θ_map_j^2 + Var(θ_j)
    
    Args:
        N (int): Maximum count value
        samples (array): Observed counts
        h (callable): Base rate function
        q_init (array): Initial prior variances
        theta0 (array, optional): Initial parameter values
        max_iter (int): Maximum number of EM iterations
        tol (float): Convergence tolerance
    
    Returns:
        tuple: (theta_map, Sigma, q, res) where
            theta_map: Final MAP estimate
            Sigma: Posterior covariance
            q: Updated prior variances
            res: Optimization result
    """
    S = compute_sufficient_statistics(samples, N)
    M = len(samples)
    q = q_init.copy()
    theta0 = np.zeros(N) if theta0 is None else theta0

    for itr in tqdm(range(max_iter), desc="EM iteration"):
        # E-step: Find MAP estimate
        res = estimate_map_parameters(N, S, M, h, q, theta0)
        theta_map = res.x

        # Compute posterior variance
        Sigma = compute_posterior_covariance(N, theta_map, h, q, M)
        var_theta = np.diag(Sigma)

        # M-step: Update q
        q_new = theta_map**2 + var_theta
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q, theta0 = q_new, theta_map
        
    return theta_map, Sigma, q, res

def estimate_ml_parameters(N, ns, h, theta0=None):
    """
    Estimate θ by maximum likelihood using only sufficient statistics.
    
    This is a simpler version of estimate_map_parameters without priors.
    
    Args:
        N (int): Maximum count value
        ns (array): Observed counts
        h (callable): Base rate function
        theta0 (array, optional): Initial parameter values
    
    Returns:
        OptimizeResult: Result from scipy.optimize.minimize
    """
    S = compute_sufficient_statistics(ns, N)
    M = len(ns)
    if theta0 is None:
        theta0 = np.zeros(N)

    def negative_log_likelihood(th):
        """Negative log-likelihood (objective function)"""
        logP, logZ = log_homogeneous_probabilities(N, th, h)
        return -(np.dot(S, th) - M*logZ)

    def gradient_negative_log_likelihood(th):
        """Gradient of negative log-likelihood"""
        logP, _ = log_homogeneous_probabilities(N, th, h)
        Pn = np.exp(logP)
        ns = np.arange(N+1)
        
        # Compute expected sufficient statistics
        C = np.array([[sp.comb(n, k) for k in range(1, N+1)] for n in ns])
        E_C = Pn @ C
        
        return -(S - M*E_C)

    res = minimize(negative_log_likelihood, theta0,
                  jac=gradient_negative_log_likelihood,
                  method='BFGS',
                  options={'disp': True})
    return res

def sample_counts(N, theta, h, size):
    """
    Draw samples of n from P(n) defined by the count distribution.

    Parameters
    ----------
    N : int
        Total number of items.
    theta : sequence of float, length N
        Natural parameters θ₁…θ_N.
    h : callable
        Weight function h(n).
    size : int
        Number of samples to draw.

    Returns
    -------
    counts : ndarray of int, shape (size,)
        Drawn values of n ~ P(n).
    """
    # Step 1: sample spike counts
    probs = homogeneous_probabilities(N, theta, h)
    counts = np.random.choice(np.arange(N + 1), size=size, p=probs)
    return counts

def sample_patterns(N, theta, h, size):
    """
    Draw binary patterns from the model by first sampling counts, then generating patterns.
    Parameters
    ----------
    N : int
        Length of each pattern.
    theta : sequence of float, length N
        Natural parameters θ₁…θ_N.
    h : callable
        Weight function h(n).
    size : int
        Number of patterns to generate.
    Returns
    -------
    patterns : ndarray, shape (size, N)
        Binary patterns with the specified number of ones.
    """
    counts = sample_counts(N, theta, h, size)
    patterns = np.zeros((size, N), dtype=int)
    for i, k in enumerate(counts):
        if k > 0:
            indices = np.random.choice(N, k, replace=False)
            patterns[i, indices] = 1
    return patterns

# ----------------------------
# Usage example
# ----------------------------
if __name__ == "__main__":
    # --------------------------
    # 1) Define true model & sample data
    # --------------------------
    N = 10
    true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

    def h(n):
        """Base rate function: h(n) = 1 for all n"""
        return 1

    print("\nTrue θ:", true_theta)
    true_probs = homogeneous_probabilities(N, true_theta, h)
    print("True P(n):", true_probs)

    # --------------------------
    # 2) Sample data
    # --------------------------
    sample_size = 10000
    samples = sample_counts(N, true_theta, h, size=sample_size)
    print("\nFirst 20 samples:", samples[:20])
    print("Empirical freq.:", 
            np.bincount(samples, minlength=N+1) / sample_size)

    # --------------------------
    # 3) ML‐fit θ
    # --------------------------
    print("\nFitting using maximum likelihood...")
    theta0 = np.zeros(N)
    result = estimate_ml_parameters(N, samples, h, theta0)
    theta_ml = result.x
    print("ML‐Estimated θ:", theta_ml)
    print("Log‐likelihood:", -result.fun)

    # --------------------------
    # 4) MAP‐fit θ with EM
    # --------------------------
    print("\nFitting using MAP with EM...")
    q = np.ones(N) * 1.0  # prior variances
    theta_map, Sigma, q, res = em_update(N, samples, h, q, theta0)
    print("MAP‐Estimated θ:", theta_map)
    print("Log‐posterior:", -res.fun)

    # --------------------------
    # 5) Show probabilities
    # --------------------------
    ml_probs = homogeneous_probabilities(N, theta_ml, h)
    map_probs = homogeneous_probabilities(N, theta_map, h)
    print("\nML P(n):", ml_probs)
    print("MAP P(n):", map_probs)

    # --------------------------
    # 6) Sample patterns
    # --------------------------
    print("\nSampling patterns...")
    patterns = sample_patterns(N, true_theta, h, size=5)
    print("Sampled patterns:\n", patterns)
