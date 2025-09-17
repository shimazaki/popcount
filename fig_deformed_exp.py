"""
Visualization tools for the Deformed Exponential Population Count Model

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
import model_deformed_exp as probability
from scipy.special import comb

def plot_deformed_probabilities(N, K, theta, q_values, h=None):
    """
    Create a figure comparing probability distributions for different q values.
    
    Parameters
    ----------
    N : int
        System size
    K : int
        Maximum interaction order
    theta : array
        Parameter vector
    q_values : array
        Array of q values to compare
    h : callable, optional
        Base rate function
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot probabilities in linear scale
    for i, q in enumerate(q_values):
        P_q = probability.homogeneous_deformed_probabilities(N, K, theta, q, h)
        color = plt.cm.viridis(i / len(q_values))
        ax1.plot(P_q, 'o-', color=color, label=f'q = {q}')
    
    ax1.set_xlim(0, N)
    ax1.set_xlabel('Count (n)')
    ax1.set_ylabel('Probability P(n)')
    ax1.set_title('Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot probabilities in log scale
    for i, q in enumerate(q_values):
        P_q = probability.homogeneous_deformed_probabilities(N, K, theta, q, h)
        color = plt.cm.viridis(i / len(q_values))
        ax2.plot(P_q, 'o-', color=color, label=f'q = {q}')
    
    ax2.set_xlim(0, N)
    ax2.set_xlabel('Count (n)')
    ax2.set_ylabel('Probability P(n)')
    ax2.set_yscale('log')
    ax2.set_title('Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Model parameters
    N = 10       # System size
    K = 2        # Up to pairwise interactions
    theta = [-3, 0.5/N]  # Some arbitrary parameters
    
    # Define base measure as lambda function
    h = lambda n: 1.0
    # Entropy canceling base measure from alternating shrinking model
    #h = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0

    # Different q values to compare
    # \gamma = 1 - q, q = 1 - \gamma
    q_values = 1 - np.array([-0.2, 0, 0.2]) /N
    
    # Create and save probability comparison figure
    fig = plot_deformed_probabilities(N, K, theta, q_values, h)
    plt.suptitle('Deformed Exponential Model - Entropy Canceling Base Measure', y=1.02)
    plt.savefig('fig/fig_deformed_exp.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("Probability distributions for entropy canceling base measure (different q values):")
    for q in q_values:
        P_q = probability.homogeneous_deformed_probabilities(N, K, theta, q, h)
        mean_val = np.sum(np.arange(N+1) * P_q)
        std_val = np.sqrt(np.sum((np.arange(N+1) - mean_val)**2 * P_q))
        print(f"q = {q:.4f}: mean = {mean_val:.3f}, std = {std_val:.3f}")
