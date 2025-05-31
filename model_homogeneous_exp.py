import numpy
import scipy.special
from scipy.optimize import minimize

def homogeneous_probabilities(N, theta, h):
    """
    Compute the normalized weights
        P[n] = (N choose n) * h(n) * exp[ Σ_{k=1}^n (n choose k) * theta[k] ] / Z
    for n = 0…N, where
        Z = Σ_{n=0}^N (N choose n) * h(n) * exp[ Σ_{k=1}^n (n choose k) * theta[k] ].

    Parameters
    ----------
    N : int
        Total number of items.
    theta : sequence of float, length N
        Natural parameters θ₁…θ_N.
    h : callable
        Weight function h(n).

    Returns
    -------
    probs : ndarray, shape (N+1,)
        Normalized weights P[0], P[1], …, P[N].
    """
    # prepend zero so that theta_internal[k] = θ_k
    theta_internal = [0.0] + list(theta)
    # n = 0…N
    Ns = numpy.arange(0, N + 1)
    # exponent for each n: Σ_{k=1}^n (n choose k) * θ_k
    exponents = numpy.array([
        sum(scipy.special.comb(n, k) * theta_internal[k] for k in range(1, n + 1))
        for n in Ns
    ])
    # binomial coefficients (N choose n)
    binoms_Nn = numpy.array([scipy.special.comb(N, n) for n in Ns])
    # weight h(n)
    h_vals = numpy.array([h(n) for n in Ns])
    # unnormalized weights
    weights = binoms_Nn * h_vals * numpy.exp(exponents)
    # normalize
    Z = weights.sum()
    return weights / Z

def expected_binomial_coefficient(N, theta, k, h):
    """
    Compute E[ C(n, k) ] under the distribution P(n) returned by homogeneous_probabilities.

    Parameters
    ----------
    N : int
        Total number of items.
    theta : sequence of float, length N
        Natural parameters θ₁…θ_N.
    k : int
        The lower index in C(n, k).
    h : callable
        Weight function h(n).

    Returns
    -------
    float
        E[ C(n, k) ].
    """
    P = homogeneous_probabilities(N, theta, h)
    ns = numpy.arange(0, N + 1)
    c_nk = scipy.special.comb(ns, k)
    return numpy.dot(P, c_nk)

def compute_sufficient_statistics(ns, N):
    """
    Compute the sufficient statistics S_k = Σ_i C(n_i, k) for k = 1…N.

    Parameters
    ----------
    ns : sequence of int
        Observed counts n_i.
    N : int
        Total number of items.

    Returns
    -------
    S : ndarray, shape (N,)
        S[k-1] = Σ_i C(n_i, k).
    """
    S = numpy.zeros(N)
    for n_i in ns:
        # for k > n_i, comb(n_i, k) = 0 automatically
        for k in range(1, N+1):
            S[k-1] += scipy.special.comb(n_i, k)
    return S

def compute_partition(N, theta, h):
    """
    Compute the partition function
        Z(θ) = Σ_{m=0}^N C(N,m) h(m) exp[ Σ_{k=1}^m C(m,k) θ_k ].
    """
    theta_internal = [0.0] + list(theta)
    ms = numpy.arange(0, N+1)
    # exponent for each m
    exponents = numpy.array([
        sum(scipy.special.comb(m, k) * theta_internal[k] for k in range(1, m+1))
        for m in ms
    ])
    binoms = numpy.array([scipy.special.comb(N, m) for m in ms])
    h_vals = numpy.array([h(m) for m in ms])
    weights = binoms * h_vals * numpy.exp(exponents)
    return weights.sum()

def log_likelihood_sufficient(theta, S, M, N, h):
    """
    Compute log‐likelihood
        L(θ) = Σ_{k=1}^N S_k θ_k − M log Z(θ) + const.
    """
    Z = compute_partition(N, theta, h)
    return numpy.dot(S, theta) - M * numpy.log(Z)

def gradient_sufficient(theta, S, M, N, h):
    """
    Compute ∇_j L = S_j − M E_θ[C(n, j)] for j = 1…N.
    """
    # get P(m) = weights/Z
    # reuse compute_partition to get Z
    Z = compute_partition(N, theta, h)
    theta_internal = [0.0] + list(theta)
    ms = numpy.arange(0, N+1)
    exponents = numpy.array([
        sum(scipy.special.comb(m, k) * theta_internal[k] for k in range(1, m+1))
        for m in ms
    ])
    binoms = numpy.array([scipy.special.comb(N, m) for m in ms])
    h_vals = numpy.array([h(m) for m in ms])
    weights = binoms * h_vals * numpy.exp(exponents)
    Pm = weights / Z  # shape (N+1,)

    # build C matrix: shape (N+1, N), C[m, j-1] = C(m, j)
    C = numpy.array([
        [scipy.special.comb(m, j) for j in range(1, N+1)]
        for m in ms
    ])  # shape (N+1, N)

    # expected C(n, j): E[j-1] = Σ_m P(m) * C(m, j)
    E = Pm @ C  # shape (N,)

    return S - M * E

def fit_theta_sufficient(N, ns, h, theta0=None):
    """
    Estimate θ by maximizing L(θ) using only sufficient statistics.

    Returns
    -------
    result : OptimizeResult
    """
    S = compute_sufficient_statistics(ns, N)
    M = len(ns)
    if theta0 is None:
        theta0 = numpy.zeros(N)

    # Negative log‐likelihood and gradient
    nll = lambda th: -log_likelihood_sufficient(th, S, M, N, h)
    grad_nll = lambda th: -gradient_sufficient(th, S, M, N, h)

    result = minimize(
        fun=nll,
        x0=theta0,
        jac=grad_nll,
        method='BFGS',
        options={'disp': True}
    )
    return result

# --------------------------
# Usage example
# --------------------------
if __name__ == "__main__":
    # Total number of items
    N = 10
    # Observed counts
    ns = [2, 3, 3, 1, 4, 2, 3]
    # Weight function h(n)=n
    def h(n):
        return n

    # Fit θ
    result = fit_theta_sufficient(N, ns, h)
    print("Estimated θ:", result.x)
    print("Log‐likelihood:", -result.fun)
