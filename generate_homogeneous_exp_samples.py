import numpy as np
import scipy.special

import model_homogeneous_exp as probability

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
    probs = probability.homogeneous_probabilities(N, theta, h)
    counts = np.random.choice(np.arange(N + 1), size=size, p=probs)
    return counts

# Function to sample patterns
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

# --------------------------
# Usage example
# --------------------------
if __name__ == "__main__":
    # Parameters
    N = 10
    theta = [-2.5, 0.2, -0.1] + [0.0] * (N - 3)
    def h(n): return 1  # example h(n)=1

    # Draw samples
    counts = sample_counts(N, theta, h, size=10000)
    print("\nFirst 20 counts:", counts[:20])
    print("\nEmpirical frequencies:", 
          np.bincount(counts, minlength=N+1) / 10000.0)

    # Draw patterns directly
    patterns = sample_patterns(N, theta, h, size=20)
    print("\nDirectly sampled patterns:\n", patterns)