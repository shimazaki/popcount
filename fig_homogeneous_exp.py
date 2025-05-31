"""
Visualization tools for the Population Count Model

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
import model_homogeneous_exp as probability

def plot_probability_comparison(true_probs, est_probs, N, true_theta, est_theta):
    """
    Create a figure comparing true and estimated probabilities and parameters.
    
    Parameters
    ----------
    true_probs : ndarray
        True probability distribution P(n)
    est_probs : ndarray
        Estimated probability distribution P(n)
    N : int
        Maximum count value
    true_theta : ndarray
        True model parameters
    est_theta : ndarray
        Estimated model parameters
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot probabilities in linear scale
    ax1.plot(true_probs, 'o-', label='True probabilities')
    ax1.plot(est_probs, 'x-', label='Estimated probabilities')
    ax1.legend()
    ax1.set_xlim(0, N)
    ax1.set_xlabel('Count (n)')
    ax1.set_ylabel('Probability P(n)')
    ax1.set_title('Linear Scale')
    ax1.grid(True, alpha=0.3)

    # Plot probabilities in log scale
    ax2.plot(true_probs, 'o-', label='True probabilities')
    ax2.plot(est_probs, 'x-', label='Estimated probabilities')
    ax2.legend()
    ax2.set_xlim(0, N)
    ax2.set_xlabel('Count (n)')
    ax2.set_ylabel('Probability P(n)')
    ax2.set_yscale('log')
    ax2.set_title('Log Scale')
    ax2.grid(True, alpha=0.3)

    # Plot parameter comparison
    ax3.plot(range(1, N+1), true_theta, 'o-', label='True θ')
    ax3.plot(range(1, N+1), est_theta, 'x-', label='Estimated θ')
    ax3.legend()
    ax3.set_xlabel('Parameter index (k)')
    ax3.set_ylabel('Parameter value (θₖ)')
    ax3.set_title('Parameter Comparison')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Model parameters
    N = 10
    true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

    def h(n):
        """Base rate function: h(n) = 1 for all n"""
        return 1

    # Compute true probabilities
    true_probs = probability.homogeneous_probabilities(N, true_theta, h)

    # Generate samples and fit using MAP with EM
    sample_size = 5000
    samples = probability.sample_counts(N, true_theta, h, size=sample_size)
    q = np.ones(N) * 10.0  # prior variances
    theta0 = np.zeros(N)
    q, theta_map, Sigma, res = probability.em_update(N, samples, h, q, theta0)
    est_probs = probability.homogeneous_probabilities(N, theta_map, h)

    # Create and save figure
    fig = plot_probability_comparison(true_probs, est_probs, N, true_theta, theta_map)
    plt.savefig('fig/modelhomogeneous_exp_pmf.png', dpi=300, bbox_inches='tight')
    plt.show()