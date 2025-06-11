"""
Population Count Model with Dichotomized Gaussian

This implementation is based on the Dichotomized Gaussian (DG) model from:
Amari, S., Nakahara, H., Wu, S. & Sakai, Y. Synchronous firing and higher-order
interactions in neuron pool. Neural Comput 15, 127–142 (2003).

The model is used to analyze higher-order interactions in neural populations as described in:
Shimazaki H, Sadeghi K, Ishikawa T, Ikegaya Y, Toyoizumi T. Simultaneous silence 
organizes structured higher-order interactions in neural populations. 
Scientific Reports (2015) 5, 9821.

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import comb

def F(epsilon, h, alpha):
    """
    Calculates the probability F(epsilon), which is the success probability
    in the binomial distribution, conditional on the common factor epsilon.
    This is P(z_i > h | epsilon). (Eq.3 in Shimazaki et al. 2015)
    """
    if alpha == 1.0:
        return 1.0 if h - epsilon < 0 else 0.0
    return norm.sf((h - epsilon * np.sqrt(alpha)) / np.sqrt(1 - alpha))

def pk_integrand(epsilon, k, N, h, alpha):
    """
    The full function to be integrated over epsilon to find P_k.
    This is [Binomial PMF] * [Standard Normal PDF].
    A part of Eq.4 in Shimazaki et al. 2015.
    """
    # Standard normal PDF for epsilon
    phi_epsilon = norm.pdf(epsilon)
    
    # Binomial probability part
    f_eps = F(epsilon, h, alpha)
    binomial_part = comb(N, k) * (f_eps**k) * ((1 - f_eps)**(N - k))
    
    return binomial_part * phi_epsilon

def calculate_pk(k, N, h, alpha):
    """
    Calculates P_k by numerically integrating the integrand from -inf to +inf.
    """
    # quad returns the result and an estimated error, we only need the result
    result, _ = quad(pk_integrand, -np.inf, np.inf, args=(k, N, h, alpha))
    return result

def calculate_pk_values(N, h, alpha, verbose=True):
    """
    Calculate P_k values using numerical integration for k from 0 to N.
    
    Parameters
    ----------
    N : int
        Maximum count value
    h : float
        Threshold parameter
    alpha : float
        Scale parameter
    verbose : bool, optional
        If True, print the calculated values
    
    Returns
    -------
    pk_values : ndarray
        Array of probabilities P(k) for k = 0 to N
    """
    if verbose:
        print("Calculating P_k values via numerical integration...")
    
    # Calculate Pk for k from 0 to N
    pk_values = np.array([calculate_pk(k, N, h, alpha) for k in range(N + 1)])

    # Normalize to ensure probabilities sum to 1, correcting for minor numerical errors
    pk_values /= np.sum(pk_values)
    
    if verbose:
        print("Calculated P_k values:")
        for i, p in enumerate(pk_values):
            print(f"P({i}) = {p:.6f}")
        print("-" * 25)
    
    return pk_values

def solve_theta_parameters(pk_values, N, verbose=True):
    """
    Solve for theta parameters using the linear system derived from the probability distribution.
    
    The system is derived from: log(P_k) = log(C(N,k)) + Sum_{j=1 to k}[C(k,j)*theta_j] + log(P_0)
    Rearranging gives: Sum_{j=1 to k}[C(k,j)*theta_j] = log(P_k / P_0) - log(C(N,k))
    
    Parameters
    ----------
    pk_values : ndarray
        Array of probabilities P(k) for k = 0 to N
    N : int
        Maximum count value
    verbose : bool, optional
        If True, print the calculated parameters
    
    Returns
    -------
    theta_params : ndarray
        Array of theta parameters θ_j for j = 1 to N
    """
    P0 = pk_values[0]

    # Right-hand side vector R for the linear system M*theta = R
    R = np.zeros(N)
    for k in range(1, N + 1):
        # The k-th equation (corresponds to index k-1)
        R[k-1] = np.log(pk_values[k] / P0) - np.log(comb(N, k))

    # Coefficient Matrix M
    M = np.zeros((N, N))
    for i in range(N):      # Row index (for k = i+1)
        for j in range(N):  # Column index (for theta_j+1)
            k = i + 1
            theta_idx = j + 1
            if theta_idx <= k:
                M[i, j] = comb(k, theta_idx)

    # Solve the system M * theta = R for theta
    try:
        theta_params = np.linalg.solve(M, R)
        if verbose:
            print("Successfully solved for Theta parameters:")
            for i, theta in enumerate(theta_params):
                print(f"θ({i+1}) = {theta:.6f}")
    except np.linalg.LinAlgError:
        if verbose:
            print("Could not solve for Theta parameters. The matrix may be singular.")
        theta_params = np.full(N, np.nan)  # Fill with NaN if solving fails
    
    return theta_params

def sample_spike_counts(N, h, alpha, size):
    """
    Sample spike counts from the homogeneous DG model.
    
    Parameters
    ----------
    N : int
        Number of neurons
    h : float
        Threshold parameter
    alpha : float
        Input correlation parameter
    size : int
        Number of samples to generate
    
    Returns
    -------
    counts : ndarray
        Array of spike counts
    """
    # Sample epsilon (common input) from standard normal
    epsilon = np.random.normal(0, 1, size)
    
    # Calculate probability of activation for each epsilon
    p = F(epsilon, h, alpha)
    
    # Sample counts from binomial distribution
    counts = np.random.binomial(N, p)
    
    return counts

def sample_patterns(N, h, alpha, size):
    """
    Sample binary patterns from the homogeneous DG model.
    
    Parameters
    ----------
    N : int
        Number of neurons
    h : float
        Threshold parameter
    alpha : float
        Input correlation parameter
    size : int
        Number of patterns to generate
    
    Returns
    -------
    patterns : ndarray, shape (size, N)
        Binary patterns
    """
    # First sample spike counts
    counts = sample_spike_counts(N, h, alpha, size)
    
    # Then generate patterns with the specified number of ones
    patterns = np.zeros((size, N), dtype=int)
    for i, k in enumerate(counts):
        if k > 0:
            indices = np.random.choice(N, k, replace=False)
            patterns[i, indices] = 1
            
    return patterns

if __name__ == "__main__":
    # --- 1. Set Model Parameters ---
    N = 10  # Total number of neurons
    alpha = 0.2  # correlation parameter
    h = 1.0  # threshold value
    
    # Calculate q from h: q = P(z > h)
    q = norm.sf(h)
    
    print(f"\nModel Parameters:")
    print(f"N = {N}")
    print(f"h = {h:.4f} (threshold)")
    print(f"q = {q:.4f} (activation probability)")
    print(f"alpha = {alpha} (correlation parameter)")
    print("-" * 25)

    # Calculate P_k values
    pk_values = calculate_pk_values(N, h, alpha)
    
    # Solve for theta parameters
    theta_params = solve_theta_parameters(pk_values, N)
    
    # Sample patterns
    n_samples = 1000
    patterns = sample_patterns(N, h, alpha, n_samples)
    print(f"\nGenerated {n_samples} sample patterns")
    print(f"First 5 patterns:")
    print(patterns[:5])