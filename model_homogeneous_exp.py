"""
Population Count Model with Homogeneous Exponential Family
(Modified for K-th order interactions)

Author: Hideaki Shimazaki
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from tqdm import tqdm

def homogeneous_probabilities(N, theta, h):
    """
    Compute probabilities P(n) for n=0,...,N for a homogeneous exponential model.
    
    Args:
        N (int): Maximum count value.
        theta (array): Parameter values (length K).
        h (callable): Base rate function.
    
    Returns:
        array: Probabilities P(n) for n=0,...,N.
    """
    K = len(theta)
    ns = np.arange(N+1)
    logP, _ = log_homogeneous_probabilities(N, K, theta, h)
    return np.exp(logP)

def log_homogeneous_probabilities(N, K, theta, h):
    """
    Compute log probabilities and log partition function for the K-th order model.
    
    The model is defined as:
        P(n) = C(N,n) * h(n) * exp(Σ_{k=1}^{min(n,K)} C(n,k)*θ_k) / Z(θ)
    
    Args:
        N (int): Maximum count value (system size).
        K (int): Maximum order of interaction to consider.
        theta (array): Model parameters θ_k for k=1..K.
        h (callable): Base rate function h(n).
    
    Returns:
        tuple: (logP, logZ) where
            logP[n] = log P(n) for n=0..N
            logZ = log partition function Z(θ)
    """
    if len(theta) != K:
        raise ValueError(f"Length of theta ({len(theta)}) must be equal to K ({K}).")
    theta_int = np.concatenate(([0.0], theta))
    ns = np.arange(N+1)

    # Compute log binomial coefficients C(N,n)
    log_binom = (sp.gammaln(N+1)
               - sp.gammaln(ns+1)
               - sp.gammaln(N-ns+1))
    
    # Compute log base rates h(n)
    log_h = np.log([h(n) for n in ns])
    
    # Sum up to min(n, K)
    exponents = np.array([
        sum(sp.comb(n, k) * theta_int[k] for k in range(1, min(n, K) + 1))
        for n in ns
    ])

    # Combine terms and normalize
    L = log_binom + log_h + exponents
    logZ = sp.logsumexp(L)  # Compute log partition function
    logP = L - logZ         # Normalize to get log probabilities
    return logP, logZ

def compute_sufficient_statistics(ns, K):
    """
    Compute sufficient statistics S_k = Σ_i C(n_i, k) for k=1..K.
    
    Args:
        ns (array): Observed counts n_i.
        K (int): Maximum order of interaction.
    
    Returns:
        array: S[k-1] = Σ_i C(n_i, k) for k=1..K.
    """
    S = np.zeros(K)
    for n in ns:
        for k in range(1, K + 1):
            S[k-1] += sp.comb(n, k)
    return S

def compute_map_gradient(N, K, S, M, h, q, theta):
    """
    Compute gradient of log-posterior for MAP estimation for a K-th order model.
    
    The gradient is: ∇_j = S_j - M*E[C(n,j)] - θ_j/q_j for j=1..K
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        S (array): Sufficient statistics (length K).
        M (int): Number of samples.
        h (callable): Base rate function.
        q (array): Prior variances (length K).
        theta (array): Current parameter values (length K).
    
    Returns:
        array: Gradient vector (length K).
    """
    logP, _ = log_homogeneous_probabilities(N, K, theta, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)
    
    # Compute expected sufficient statistics up to order K.
    C = np.array([[sp.comb(n, k) for k in range(1, K+1)] for n in ns])
    E_C = Pn @ C
    
    return S - M*E_C - theta/q

def estimate_map_parameters(N, K, S, M, h, q, theta):
    """
    Find MAP estimate of θ (length K) given sufficient statistics.
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        S (array): Sufficient statistics (length K).
        M (int): Number of samples.
        h (callable): Base rate function.
        q (array): Prior variances (length K).
        theta (array): Current parameter values (length K).
    
    Returns:
        OptimizeResult: Result from scipy.optimize.minimize.
    """
    def negative_log_posterior(th):
        """Negative log-posterior (objective function)"""
        _, logZ = log_homogeneous_probabilities(N, K, th, h)
        ll = np.dot(S, th) - M*logZ
        prior = -0.5 * np.sum(th**2 / q)
        return -(ll + prior)

    def gradient_negative_log_posterior(th):
        """Gradient of negative log-posterior"""
        return -compute_map_gradient(N, K, S, M, h, q, th)

    res = minimize(negative_log_posterior, theta, 
                  jac=gradient_negative_log_posterior, 
                  method='L-BFGS-B', 
                  options={'disp':False})
    return res

def compute_posterior_covariance(N, K, theta_map, h, q, M):
    """
    Compute posterior covariance matrix for a K-th order model.
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        theta_map (array): MAP estimate of θ (length K).
        h (callable): Base rate function.
        q (array): Prior variances (length K).
        M (int): Number of samples.
    
    Returns:
        array: Posterior covariance matrix (K x K).
    """
    logP, _ = log_homogeneous_probabilities(N, K, theta_map, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)

    # Compute covariance of sufficient statistics up to order K.
    C = np.array([[sp.comb(n, k) for k in range(1, K+1)] for n in ns])
    E1 = Pn @ C
    E2 = C.T @ (Pn[:,None] * C)
    Cov_C = E2 - np.outer(E1, E1)

    # Compute Hessian and invert
    H = M * Cov_C + np.diag(1.0/q)
    Sigma = np.linalg.inv(H)
    return Sigma

def em_update(N, samples, h, K=None, q_init=None, theta0=None, max_iter=100, tol=1e-6):
    """
    Empirical-Bayes EM algorithm for a K-th order model.
    
    Args:
        N (int): Maximum count value.
        samples (array): Observed counts.
        h (callable): Base rate function.
        K (int, optional): Maximum order of interaction. If None, uses K=N.
        q_init (array, optional): Initial prior variances (length K). If None, uses ones.
        theta0 (array, optional): Initial parameter values (length K). If None, uses zeros.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        tuple: (theta_map, Sigma, q, res) where
            theta_map: Final MAP estimate (length K)
            Sigma: Posterior covariance (K x K)
            q: Updated prior variances (length K)
            res: Optimization result
    """
    if K is None:
        K = N
    S = compute_sufficient_statistics(samples, K)
    M = len(samples)
    if q_init is None:
        q_init = np.ones(K)
    if theta0 is None:
        theta0 = np.zeros(K)
    q = q_init.copy()
    theta_est = theta0.copy()

    for itr in tqdm(range(max_iter), desc="EM iteration"):
        # E-step: Find MAP estimate
        res = estimate_map_parameters(N, K, S, M, h, q, theta_est)
        theta_map = res.x

        # Compute posterior variance
        Sigma = compute_posterior_covariance(N, K, theta_map, h, q, M)
        var_theta = np.diag(Sigma)

        # M-step: Update q
        q_new = theta_map**2 + var_theta
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q, theta_est = q_new, theta_map
        
    return theta_map, Sigma, q, res

def estimate_ml_parameters(N, samples, h, K=None, theta0=None):
    """
    Estimate ML parameters for a K-th order model.
    
    Args:
        N (int): Maximum count value.
        samples (array): Observed counts.
        h (callable): Base rate function.
        K (int, optional): Maximum order of interaction. If None, uses K=N.
        theta0 (array, optional): Initial parameter values (length K). If None, uses zeros.
    
    Returns:
        OptimizeResult: Optimization result containing ML estimate.
    """
    if K is None:
        K = N
    S = compute_sufficient_statistics(samples, K)
    M = len(samples)
    if theta0 is None:
        theta0 = np.zeros(K)
    
    def negative_log_likelihood(th):
        """Negative log-likelihood (objective function)"""
        _, logZ = log_homogeneous_probabilities(N, K, th, h)
        return -(np.dot(S, th) - M*logZ)
    
    def gradient_negative_log_likelihood(th):
        """Gradient of negative log-likelihood"""
        logP, _ = log_homogeneous_probabilities(N, K, th, h)
        Pn = np.exp(logP)
        ns = np.arange(N+1)
        
        # Compute expected statistics up to order K.
        C = np.array([[sp.comb(n, k) for k in range(1, K+1)] for n in ns])
        E_C = Pn @ C
        
        return -(S - M*E_C)
    
    res = minimize(negative_log_likelihood, theta0,
                  jac=gradient_negative_log_likelihood,
                  method='L-BFGS-B',
                  options={'disp': True})
    return res

def sample_counts(N, theta, h, size):
    """Draw samples of n from P(n) defined by the full N-order model."""
    probs = homogeneous_probabilities(N, theta, h)
    counts = np.random.choice(np.arange(N + 1), size=size, p=probs)
    return counts

def sample_patterns(N, theta, h, size):
    """Draw binary patterns from the full N-order model."""
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
