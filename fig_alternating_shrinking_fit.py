"""
Visualization tools for the Alternating Shrinking Model

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
import model_alternating_shrinking
import model_homogeneous_exp as probability

def plot_probability_comparison(true_probs, est_probs, N, true_theta, est_theta, Sigma=None):
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
    Sigma : ndarray, optional
        Posterior covariance matrix for parameter uncertainty
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

    # Plot parameter comparison with error bars
    x = np.arange(1, N+1)
    if Sigma is not None:
        # Compute standard errors from diagonal of covariance matrix
        std_errors = np.sqrt(np.diag(Sigma))
        ax3.errorbar(x, est_theta, yerr=std_errors, fmt='x-', 
                    label='Estimated θ', capsize=5, capthick=1)
    else:
        ax3.plot(x, est_theta, 'x-', label='Estimated θ')
    
    ax3.plot(x, true_theta, 'o-', label='True θ')
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
    f = 10.0  # sparsity-inducing parameter
    m = 1.0   # power law exponent
    Cj_func = lambda j: 1 / j**m

    # Compute true probabilities
    true_probs = model_alternating_shrinking.compute_n_spike_pmf_with_func(N, f, Cj_func)

    # Generate samples and fit using MAP with EM
    sample_size = 5000
    samples = model_alternating_shrinking.sample_spike_counts(N, f, Cj_func, size=sample_size)
    q = np.ones(N) * 1.0  # prior variances
    theta0 = np.zeros(N)
    def h(n):
        return 1
    theta_map, Sigma, q, res = probability.em_update(N, samples, h, q, theta0)
    est_probs = probability.homogeneous_probabilities(N, theta_map, h)

    # Create and save figure
    true_theta = np.zeros(N)
    fig = plot_probability_comparison(true_probs, est_probs, N, true_theta, theta_map, Sigma)
    plt.savefig('fig/model_alternating_shrinking_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
