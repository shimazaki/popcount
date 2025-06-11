"""
Population Count Model with Homogeneous Exponential Family
(Modified for K-th order interactions)

Author: Hideaki Shimazaki
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from tqdm import tqdm

def homogeneous_probabilities(N, K, theta, h):
    """
    Compute probabilities P(n) for all n=0..N using the K-th order homogeneous model.
    
    Args:
        N (int): Maximum count value (system size).
        K (int): Maximum order of interaction to consider.
        theta (array): Model parameters θ_k for k=1..K.
        h (callable): Base rate function h(n).
    
    Returns:
        array: P(n) for n=0..N.
    """
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

def estimate_map_parameters(N, K, S, M, h, q, theta0=None):
    """
    Find MAP estimate of θ (length K) given sufficient statistics.
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        S (array): Sufficient statistics (length K).
        M (int): Number of samples.
        h (callable): Base rate function.
        q (array): Prior variances (length K).
        theta0 (array, optional): Initial parameter values (length K).
    
    Returns:
        OptimizeResult: Result from scipy.optimize.minimize.
    """
    if theta0 is None:
        theta0 = np.zeros(K)

    def negative_log_posterior(th):
        """Negative log-posterior (objective function)"""
        _, logZ = log_homogeneous_probabilities(N, K, th, h)
        ll = np.dot(S, th) - M*logZ
        prior = -0.5 * np.sum(th**2 / q)
        return -(ll + prior)

    def gradient_negative_log_posterior(th):
        """Gradient of negative log-posterior"""
        return -compute_map_gradient(N, K, S, M, h, q, th)

    res = minimize(negative_log_posterior, theta0, 
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

def em_update(N, K, samples, h, q_init, theta0=None, max_iter=100, tol=1e-6):
    """
    Empirical-Bayes EM algorithm for a K-th order model.
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        samples (array): Observed counts.
        h (callable): Base rate function.
        q_init (array): Initial prior variances (length K).
        theta0 (array, optional): Initial parameter values (length K).
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        tuple: (theta_map, Sigma, q, res) where
            theta_map: Final MAP estimate (length K)
            Sigma: Posterior covariance (K x K)
            q: Updated prior variances (length K)
            res: Optimization result
    """
    S = compute_sufficient_statistics(samples, K)
    M = len(samples)
    q = q_init.copy()
    theta_est = np.zeros(K) if theta0 is None else theta0

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

def estimate_ml_parameters(N, K, ns, h, theta0=None):
    """
    Estimate θ (length K) by maximum likelihood for a K-th order model.
    
    Args:
        N (int): Maximum count value.
        K (int): Maximum order of interaction.
        ns (array): Observed counts.
        h (callable): Base rate function.
        theta0 (array, optional): Initial parameter values (length K).
    
    Returns:
        OptimizeResult: Result from scipy.optimize.minimize.
    """
    S = compute_sufficient_statistics(ns, K)
    M = len(ns)
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
    probs = homogeneous_probabilities(N, len(theta), theta, h)
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
    # 1) Define true model & sample data
    # --------------------------
    N = 10
    # K-th order model
    K = 4
    
    # True model can have interactions up to N
    true_theta_full = np.array([-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4))

    def h(n):
        """Base rate function: h(n) = 1 for all n"""
        return 1

    print(f"\nSystem size N = {N}")
    print(f"Fitting up to K = {K}-th order interactions.")
    print("\nTrue full θ (N-dim):", true_theta_full)

    # --------------------------
    # 2) Sample data from the true N-dim model
    # --------------------------
    sample_size = 10000
    samples = sample_counts(N, true_theta_full, h, size=sample_size)
    print("\nFirst 20 samples:", samples[:20])
    
    # --------------------------
    # 3) ML‐fit θ (K-th order)
    # --------------------------
    print(f"\nFitting using maximum likelihood (K={K})...")
    # K-th order change: Initialize theta of length K.
    theta0_k = np.zeros(K)
    # K-th order change: Pass K to the estimator.
    result_ml = estimate_ml_parameters(N, K, samples, h, theta0_k)
    theta_ml = result_ml.x
    print("ML‐Estimated θ (K-dim):", theta_ml)
    print("Log‐likelihood:", -result_ml.fun)

    # --------------------------
    # 4) MAP‐fit θ with EM (K-th order)
    # --------------------------
    print(f"\nFitting using MAP with EM (K={K})...")
    # K-th order change: Initialize q and theta of length K.
    q_k = np.ones(K) * 1.0
    theta0_k = np.zeros(K)
    # K-th order change: Pass K to the EM updater.
    theta_map, Sigma, q, res_map = em_update(N, K, samples, h, q_k, theta0_k)
    print("MAP‐Estimated θ (K-dim):", theta_map)
    print("Final learned q (K-dim):", q)
    print("Log‐posterior:", -res_map.fun)

    # --------------------------
    # 5) Show probabilities
    # --------------------------
    true_probs = homogeneous_probabilities(N, N, true_theta_full, h)
    ml_probs = homogeneous_probabilities(N, K, theta_ml, h)
    map_probs = homogeneous_probabilities(N, K, theta_map, h)
    
    print("\n--- Probabilities P(n) ---")
    print("True :", true_probs)
    print("ML   :", ml_probs)
    print("MAP  :", map_probs)
