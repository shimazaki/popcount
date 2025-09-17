"""
Population Count Model with Alternating Shrinking

Author: Hideaki Shimazaki
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from functools import lru_cache
from math import factorial

def compute_n_spike_pmf_with_func(N, f, Cj_func):
    """
    Computes the n-spike count PMF using a user-defined function for C_j.

    Parameters
    ----------
    N : int
        Total number of neurons
    f : float
        Sparsity-inducing parameter (𝓯)
    Cj_func : callable
        Function Cj_func(j) returning C_j for j=1 to N

    Returns
    -------
    pmf : ndarray
        Array of probabilities P(n) for n = 0 to N
    """
    n_values = np.arange(N + 1)
    r_values = n_values / N

    # Exponent term for each n
    exponent_terms = np.zeros_like(r_values)
    for idx, r in enumerate(r_values):
        exponent_terms[idx] = -f * sum(
            ((-1) ** (j + 1)) * Cj_func(j) * (r ** j)
            for j in range(1, N + 1)
        )

    # Unnormalized probabilities
    unnormalized_p = np.exp(exponent_terms)

    # Normalize
    Z = np.sum(unnormalized_p)
    pmf = unnormalized_p / Z

    return pmf

def sample_spike_counts(N, f, Cj_func, size):
    """
    Sample spike counts from the PMF computed by compute_n_spike_pmf_with_func.

    Parameters
    ----------
    N : int
        Total number of neurons
    f : float
        Sparsity-inducing parameter (𝓯)
    Cj_func : callable
        Function Cj_func(j) returning C_j for j=1 to N
    size : int
        Number of samples to draw

    Returns
    -------
    counts : ndarray of int, shape (size,)
        Drawn values of n ~ P(n)
    """
    pmf = compute_n_spike_pmf_with_func(N, f, Cj_func)
    counts = np.random.choice(np.arange(N + 1), size=size, p=pmf)
    return counts

def sample_patterns(N, f, Cj_func, size):
    """
    Sample binary patterns by first sampling spike counts, then generating patterns with the specified number of ones.

    Parameters
    ----------
    N : int
        Length of each pattern
    f : float
        Sparsity-inducing parameter (𝓯)
    Cj_func : callable
        Function Cj_func(j) returning C_j for j=1 to N
    size : int
        Number of patterns to generate

    Returns
    -------
    patterns : ndarray, shape (size, N)
        Binary patterns with the specified number of ones
    """
    counts = sample_spike_counts(N, f, Cj_func, size)
    patterns = np.zeros((size, N), dtype=int)
    for i, k in enumerate(counts):
        if k > 0:
            indices = np.random.choice(N, k, replace=False)
            patterns[i, indices] = 1
    return patterns

def gibbs_sampler(N, f, Cj_func, h_func, steps=10000, burn_in=1000, seed=None):
    """
    Gibbs sampler for binary patterns from the sparse population model,
    with a general base measure function h(n).

    Parameters
    ----------
    N : int
        Number of neurons
    f : float
        Sparsity-inducing parameter
    Cj_func : callable
        Function returning C_j for j=1 to N
    h_func : callable
        Function returning h(n) for n in [0, N]
    steps : int, optional
        Total number of Gibbs steps
    burn_in : int, optional
        Number of steps to discard before collecting samples
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    samples : ndarray
        Collected binary samples, shape (steps - burn_in, N)
    """
    if seed is not None:
        np.random.seed(seed)

    def S(n):
        r = n / N
        return sum(((-1)**(j+1)) * Cj_func(j) * r**j for j in range(1, N + 1))

    x = np.random.randint(0, 2, N)  # Initial random binary pattern
    samples = []
    n = np.sum(x)  # Current total spike count

    # Create progress bar for total steps
    pbar = tqdm(total=steps, desc="Gibbs sampling")
    
    for step in range(steps):
        for i in range(N):
            xi = x[i]
            # n_except_i: spike count of all neurons except neuron i
            n_except_i = n - xi

            # Calculate unnormalized probabilities for x[i]=0 and x[i]=1
            # If x[i]=0, total spike count is n_except_i
            p0 = h_func(n_except_i)     * np.exp(-f * S(n_except_i))
            # If x[i]=1, total spike count is n_except_i + 1
            p1 = h_func(n_except_i + 1) * np.exp(-f * S(n_except_i + 1))

            # Probability of setting x[i]=1, given the rest
            prob_1 = p1 / (p0 + p1)
            x[i] = 1 if np.random.rand() < prob_1 else 0
            n = n_except_i + x[i]  # Update current total spike count

        if step >= burn_in:
            samples.append(x.copy())
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    return np.array(samples)


# Conversion between C_j and theta_k
# --- Stirling numbers ---
@lru_cache(maxsize=None)
def _S2(n: int, k: int) -> int:
    """Stirling numbers of the second kind S(n,k)."""
    if n == k == 0: return 1
    if n == 0 or k == 0: return 0
    if k > n: return 0
    return k * _S2(n - 1, k) + _S2(n - 1, k - 1)

@lru_cache(maxsize=None)
def _s1(n: int, k: int) -> int:
    """SIGNED Stirling numbers of the first kind s(n,k)."""
    if n == k == 0: return 1
    if n == 0 or k == 0: return 0
    if k > n: return 0
    return _s1(n - 1, k - 1) - (n - 1) * _s1(n - 1, k)

# --- Conversions: use 0-based lists: C[j-1] == C_j, theta[k-1] == theta_k ---
def cj_to_theta(C, f):
    """
    Convert C_j -> theta_k.
    Inputs:
      C: list/array length N with C[0]=C_1,...,C[N-1]=C_N
      f: sparsity parameter (mathpzc{f})
    Returns:
      theta: list length N with theta[0]=theta_1,...,theta[N-1]=theta_N
    """
    N = len(C)
    theta = [0.0] * N
    for k in range(1, N + 1):
        acc = 0.0
        for l in range(k, N + 1):
            acc += ((-1) ** l) * f * C[l - 1] / (N ** l) * (factorial(k) * _S2(l, k))
        theta[k - 1] = acc
    return theta

def theta_to_cj(theta, f):
    """
    Convert theta_k -> C_j (inverse).
    Inputs:
      theta: list/array length N with theta[0]=theta_1,...,theta[N-1]=theta_N
      f: sparsity parameter (mathpzc{f})
    Returns:
      C: list length N with C[0]=C_1,...,C[N-1]=C_N
    """
    N = len(theta)
    C = [0.0] * N
    for l in range(1, N + 1):
        inner = 0.0
        for k in range(l, N + 1):
            inner += _s1(k, l) * (theta[k - 1] / factorial(k))
        C[l - 1] = ((-1) ** l) * (N ** l) / f * inner
    return C


if __name__ == "__main__":
    # Parameters
    N = 10
    h_func = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0

    #Polylogarithmic exponential distribution
    #f = 50.0
    # m = 1.0
    # Cj_func = lambda j: 1 / j**m

    #Shifted-geometric exponential distribution
    f = 20.0
    tau = 0.8
    Cj_func = lambda j: tau**j

    # Compute PMF
    pmf = compute_n_spike_pmf_with_func(N, f, Cj_func)
    
    # Print statistics
    mean = np.sum(np.arange(N+1) * pmf)
    variance = np.sum((np.arange(N+1) - mean)**2 * pmf)
    print(f"Mean number of active neurons: {mean:.2f}")
    print(f"Variance: {variance:.2f}")
    
    # Create fig directory if it doesn't exist
    os.makedirs('fig', exist_ok=True)
    
    # Generate samples
    n_samples = 3000
    print("\nGenerating exact samples...")
    exact_counts = sample_spike_counts(N, f, Cj_func, size=n_samples)
    
    print("\nGenerating Gibbs samples...")
    gibbs_samples = gibbs_sampler(N, f, Cj_func, h_func, steps=n_samples+500, burn_in=500, seed=42)
    gibbs_counts = np.sum(gibbs_samples, axis=1)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot analytical PMF
    plt.plot(np.arange(N+1), pmf, 'k-', label='Analytical PMF', linewidth=2)
    
    # Plot histograms
    plt.hist(exact_counts, bins=np.arange(N+2)-0.5, density=True, alpha=0.5, 
             label='Exact sampling', color='blue')
    plt.hist(gibbs_counts, bins=np.arange(N+2)-0.5, density=True, alpha=0.5,
             label='Gibbs sampling', color='red')
    
    plt.title(f"n-Spike Count PMF (f={f})")
    plt.xlabel("Number of Active Neurons (n)")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig('fig/model_alternating_shrinking_pmf.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print sample statistics
    print("\nExact sampling method:")
    print("First 20 sampled spike counts:", exact_counts[:20])
    print(f"Mean: {np.mean(exact_counts):.2f}")
    print(f"Variance: {np.var(exact_counts):.2f}")
    
    print("\nGibbs sampling method:")
    print("First 5 Gibbs samples:\n", gibbs_samples[:5])
    print(f"Mean: {np.mean(gibbs_counts):.2f}")
    print(f"Variance: {np.var(gibbs_counts):.2f}")
    
    # Test Cj -> theta -> Cj conversion
    Cj_original = [Cj_func(j) for j in range(1, N+1)]
    theta = cj_to_theta(Cj_original, f)
    Cj_recovered = theta_to_cj(theta, f)
    max_error = max(abs(Cj_original[i] - Cj_recovered[i]) for i in range(N))
    print(f"\nCj->theta->Cj conversion test:")
    print(f"Original Cj (first 5): {[f'{x:.6f}' for x in Cj_original[:5]]}")
    print(f"Recovered Cj (first 5): {[f'{x:.6f}' for x in Cj_recovered[:5]]}")
    print(f"Max error = {max_error:.2e}") 