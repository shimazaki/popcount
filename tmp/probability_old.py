import numpy
import scipy.special
import math
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

def log_likelihood(theta, N, ns, h):
    """
    Compute the log‐likelihood of observed data {n_i} under our model.

    ℓ(θ) = Σ_i [log C(N,n_i) + log h(n_i) + Σ_{k=1}^{n_i} C(n_i,k) θ_k] − M log Z(θ)

    Parameters
    ----------
    theta : ndarray, shape (N,)
        Parameters θ₁…θ_N.
    N : int
        Total number of items.
    ns : sequence of int
        Observed counts n_i.
    h : callable
        Weight function h(n).

    Returns
    -------
    float
        Log‐likelihood value.
    """
    M = len(ns)
    # compute Z(θ) via homogeneous_probabilities internals
    theta_internal = [0.0] + list(theta)
    ms = numpy.arange(0, N + 1)
    exponents = numpy.array([
        sum(scipy.special.comb(m, k) * theta_internal[k] for k in range(1, m + 1))
        for m in ms
    ])
    binoms_Nm = numpy.array([scipy.special.comb(N, m) for m in ms])
    h_vals = numpy.array([h(m) for m in ms])
    weights = binoms_Nm * h_vals * numpy.exp(exponents)
    Z = weights.sum()

    ll = 0.0
    for n_i in ns:
        ll += math.log(scipy.special.comb(N, n_i))
        ll += math.log(h(n_i))
        ll += sum(scipy.special.comb(n_i, k) * theta_internal[k]
                  for k in range(1, n_i + 1))
    ll -= M * math.log(Z)
    return ll

def gradient(theta, N, ns, h):
    """
    Compute gradient ∂ℓ/∂θ_j = Σ_i C(n_i,j) − M * E[C(n,j)].

    Returns
    -------
    grad : ndarray, shape (N,)
    """
    M = len(ns)
    # expected C(n,j)
    ms = numpy.arange(0, N + 1)
    Pm = homogeneous_probabilities(N, theta, h)
    grad = numpy.zeros_like(theta)
    for j in range(1, N + 1):
        sum_obs = sum(scipy.special.comb(n_i, j) for n_i in ns)
        expected = numpy.dot(Pm, scipy.special.comb(ms, j))
        grad[j - 1] = sum_obs - M * expected
    return grad

def fit_theta(N, ns, h, theta0=None):
    """
    Estimate θ by maximizing the log‐likelihood using BFGS.

    Returns
    -------
    result : OptimizeResult
    """
    if theta0 is None:
        theta0 = numpy.zeros(N)
    # minimize negative log‐likelihood
    nll = lambda th: -log_likelihood(th, N, ns, h)
    grad_nll = lambda th: -gradient(th, N, ns, h)
    result = minimize(fun=nll,
                      x0=theta0,
                      jac=grad_nll,
                      method='BFGS',
                      options={'disp': True})
    return result

# --------------------------
# Usage example
# --------------------------
if __name__ == "__main__":
    import math

    N = 10
    # Observed counts
    ns = [2, 3, 3, 1, 4, 2, 3]
    # Define h(n) = n
    def h(n):
        return n

    # Initial guess θ₁…θ₁₀ = 0
    theta0 = numpy.zeros(N)

    # Fit θ
    result = fit_theta(N, ns, h, theta0)
    print("Estimated θ:", result.x)
    print("Log-likelihood:", -result.fun)
