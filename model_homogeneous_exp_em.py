import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
import generate_homogeneous_exp_samples as generate_samples

def homogeneous_probabilities(N, theta, h):
    """
    Compute probabilities for all n=0..N using the homogeneous model.
    """
    logP, _ = log_homogeneous_probabilities(N, theta, h)
    return np.exp(logP)

def log_homogeneous_probabilities(N, theta, h):
    """
    Returns (logP, logZ) where
      logP[n] = log P(n) for n=0..N,
      logZ    = log partition function.
    """
    theta_int = np.concatenate(([0.0], theta))
    ns = np.arange(N+1)

    # log binomial C(N,n)
    log_binom = (sp.gammaln(N+1)
               - sp.gammaln(ns+1)
               - sp.gammaln(N-ns+1))
    # log h(n)
    log_h = np.log([h(n) for n in ns])
    # exponent term Σ_{k=1}^n C(n,k)*θ_k
    exponents = np.array([
        sum(sp.comb(n, k) * theta_int[k] for k in range(1, n+1))
        for n in ns
    ])

    L = log_binom + log_h + exponents
    logZ = sp.logsumexp(L)
    logP = L - logZ
    return logP, logZ

def compute_sufficient_statistics(ns, N):
    """S[k-1] = Σ_i C(n_i, k) for k=1…N."""
    S = np.zeros(N)
    for n in ns:
        for k in range(1, N+1):
            S[k-1] += sp.comb(n, k)
    return S

def gradient_map_from_sufficient(N, S, M, h, q, theta):
    """
    ∇_j = S_j - M*E[C(n,j)] - θ_j/q_j
    using only P(n|θ) & S,M.
    """
    logP, _ = log_homogeneous_probabilities(N, theta, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)
    # C[n,j-1] = C(n,j)
    C = np.array([[sp.comb(n, k) for k in range(1, N+1)] for n in ns])
    E_C = Pn @ C
    return S - M*E_C - theta/q

def fit_theta_map_sufficient(N, S, M, h, q, theta0=None):
    """
    MAP‐estimate θ given S and M.
    """
    if theta0 is None:
        theta0 = np.zeros(N)

    def nlp(th):
        logP, logZ = log_homogeneous_probabilities(N, th, h)
        ll = np.dot(S, th) - M*logZ
        prior = -0.5 * np.sum(th**2 / q)
        return -(ll + prior)

    def grad_nlp(th):
        return -gradient_map_from_sufficient(N, S, M, h, q, th)

    res = minimize(nlp, theta0, jac=grad_nlp, method='BFGS', options={'disp':False})
    return res

def posterior_laplace_stats(N, theta_map, h, q, M):
    """
    Given θ_map, approximate posterior covariance via Laplace:
      H = M*Cov_C + diag(1/q)
      Σ ≈ H^{-1}.
    """
    logP, _ = log_homogeneous_probabilities(N, theta_map, h)
    Pn = np.exp(logP)
    ns = np.arange(N+1)

    C = np.array([[sp.comb(n, k) for k in range(1, N+1)] for n in ns])
    E1 = Pn @ C
    E2 = C.T @ (Pn[:,None] * C)
    Cov_C = E2 - np.outer(E1, E1)

    H = M * Cov_C + np.diag(1.0/q)
    Sigma = np.linalg.inv(H)
    return Sigma

def em_update_q_from_sufficient(N, samples, h, q_init, theta0=None, max_iter=20, tol=1e-6):
    """
    Empirical‐Bayes EM on prior variances q_j:
      E-step: θ_map = MAP estimate given q
      compute Var(θ) from Laplace = Σ[j,j]
      M-step: q_j = θ_map_j^2 + Var(θ_j)
    """
    S = compute_sufficient_statistics(samples, N)
    M = len(samples)
    q = q_init.copy()
    theta0 = np.zeros(N) if theta0 is None else theta0

    for itr in range(max_iter):
        res = fit_theta_map_sufficient(N, S, M, h, q, theta0)
        theta_map = res.x

        Sigma = posterior_laplace_stats(N, theta_map, h, q, M)
        var_theta = np.diag(Sigma)

        q_new = theta_map**2 + var_theta
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q, theta0 = q_new, theta_map
        #q, theta0 = q, theta_map
        
    return q, theta_map, Sigma, res

def fit_theta_map(N, counts, h, q, theta0):
    """MAP‐fit θ under the homogeneous model."""
    # Compute sufficient statistics
    T1 = len(counts)
    T2 = np.sum(counts)
    
    # Define objective function (negative log‐posterior)
    def objective(theta):
        # Compute log P(n) for each n
        log_probs = np.zeros(N+1)
        for n in range(N+1):
            log_probs[n] = np.log(h(n)) + theta[0] + n*theta[1] + sp.comb(N, n, exact=True)
        
        # Normalize to get probabilities
        log_probs -= sp.logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Compute log‐likelihood
        log_lik = T1 * np.log(probs[0]) + T2 * np.log(probs[1]/probs[0])
        
        # Compute log‐prior
        log_prior = -0.5 * np.sum(theta**2 / q)
        
        return -(log_lik + log_prior)
    
    # Optimize
    result = minimize(objective, theta0, method='BFGS')
    return result

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
        return 1      # here h(n) ≡ 1

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
    #result = fit_theta_map(N, samples, h, q, theta0)
    #theta_est = result.x

    print(h)

    q, theta_map, Sigma, res = em_update_q_from_sufficient(N, samples, h, q, theta0)
    theta_est = theta_map
    print("\nMAP‐Estimated θ:", theta_est)
    print("Log‐posterior:", -res.fun)

    # --------------------------
    # 4) Show MAP probabilities
    # --------------------------
    est_probs = homogeneous_probabilities(N, theta_est, h)
    print("MAP P(n):", est_probs)
