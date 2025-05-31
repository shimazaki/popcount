import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
import generate_homogeneous_exp_samples as generate_samples

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

def em_update(N, samples, h, q_init, theta0=None, max_iter=20, tol=1e-6):
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
        tuple: (q, theta_map, Sigma, res) where
            q: Updated prior variances
            theta_map: Final MAP estimate
            Sigma: Posterior covariance
            res: Optimization result
    """
    S = compute_sufficient_statistics(samples, N)
    M = len(samples)
    q = q_init.copy()
    theta0 = np.zeros(N) if theta0 is None else theta0

    for itr in range(max_iter):
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
        
    return q, theta_map, Sigma, res

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
    samples = generate_samples.sample_counts(N, true_theta, h, size=sample_size)
    print("\nFirst 20 samples:", samples[:20])
    print("Empirical freq.:", 
            np.bincount(samples, minlength=N+1) / sample_size)

    # --------------------------
    # 3) MAP‐fit θ
    # --------------------------
    q = np.ones(N) * 10.0  # prior variances
    theta0 = np.zeros(N)

    print(h)

    q, theta_map, Sigma, res = em_update(N, samples, h, q, theta0)
    theta_est = theta_map
    print("\nMAP‐Estimated θ:", theta_est)
    print("Log‐posterior:", -res.fun)

    # --------------------------
    # 4) Show MAP probabilities
    # --------------------------
    est_probs = homogeneous_probabilities(N, theta_est, h)
    print("MAP P(n):", est_probs)
