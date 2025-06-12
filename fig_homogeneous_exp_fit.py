"""
Visualization tools for the Population Count Model

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
import model_homogeneous_exp as probability
from scipy.special import comb

def plot_probability_comparison(true_probs, map_probs, ml_probs, N, true_theta, map_theta, ml_theta, Sigma=None):
    """
    Create a figure comparing true, MAP, and MLE probabilities and parameters.
    
    Parameters
    ----------
    true_probs : ndarray
        True probability distribution P(n)
    map_probs : ndarray
        MAP-estimated probability distribution P(n)
    ml_probs : ndarray
        MLE-estimated probability distribution P(n)
    N : int
        Maximum count value
    true_theta : ndarray
        True model parameters
    map_theta : ndarray
        MAP-estimated model parameters
    ml_theta : ndarray
        MLE-estimated model parameters
    Sigma : ndarray, optional
        Posterior covariance matrix for parameter uncertainty
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Create handles and labels for legend
    handles = []
    labels = []

    # Plot probabilities in linear scale
    h1 = ax1.plot(true_probs, 'o-', color='black')[0]
    h2 = ax1.plot(ml_probs, 's-', color='tab:blue')[0]
    h3 = ax1.plot(map_probs, 'x-', color='tab:orange')[0]
    ax1.legend(['True probabilities', 'MLE probabilities', 'MAP probabilities'])
    ax1.set_xlim(0, N)
    ax1.set_xlabel('Count (n)')
    ax1.set_ylabel('Probability P(n)')
    ax1.set_title('Linear Scale')
    ax1.grid(True, alpha=0.3)

    # Plot probabilities in log scale
    ax2.plot(true_probs, 'o-', color='black')
    ax2.plot(ml_probs, 's-', color='tab:blue')
    ax2.plot(map_probs, 'x-', color='tab:orange')
    ax2.legend(['True probabilities', 'MLE probabilities', 'MAP probabilities'])
    ax2.set_xlim(0, N)
    ax2.set_xlabel('Count (n)')
    ax2.set_ylabel('Probability P(n)')
    ax2.set_yscale('log')
    ax2.set_title('Log Scale')
    ax2.grid(True, alpha=0.3)

    # Plot parameter comparison with error bars
    x = np.arange(1, N+1)
    
    # Draw true θ first
    h1 = ax3.plot(x, true_theta, 'o-', color='black')[0]
    handles.append(h1)
    labels.append('True θ')
    
    # Draw MLE θ
    h2 = ax3.plot(x, ml_theta, 's-', color='tab:blue')[0]
    handles.append(h2)
    labels.append('MLE θ')
    
    # Draw MAP θ with error bars
    if Sigma is not None:
        std_errors = np.sqrt(np.diag(Sigma))
        h3 = ax3.errorbar(x, map_theta, yerr=2*std_errors, fmt='x-', color='tab:orange',
                         capsize=5, capthick=1)[0]
    else:
        h3 = ax3.plot(x, map_theta, 'x-', color='tab:orange')[0]
    handles.append(h3)
    labels.append('MAP θ')
    
    # Add legend with specified order
    ax3.legend(handles, labels)
    ax3.set_xlabel('Parameter index (k)')
    ax3.set_ylabel('Parameter value (θₖ)')
    ax3.set_title('Parameter Comparison')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Set random seed for reproducibility
    #np.random.seed(42)
    
    # Model parameters
    N = 10
    true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

    # Define base measure as lambda function
    #h = lambda n: 1.0 / comb(N, n)
    h = lambda n: 1.0

    # Compute true probabilities
    true_probs = probability.homogeneous_probabilities(N, true_theta, h)

    # Generate samples
    sample_size = 5000
    samples = probability.sample_counts(N, true_theta, h, size=sample_size)

    # Fit using MAP with EM
    q = np.ones(N) * 1.0  # prior variances
    theta_map, Sigma, q, res = probability.em_update(N, samples, max_iter=100)
    map_probs = probability.homogeneous_probabilities(N, theta_map)
    
    # Fit using MLE
    result_ml = probability.estimate_ml_parameters(N, samples)
    theta_ml = result_ml.x
    ml_probs = probability.homogeneous_probabilities(N, theta_ml)

    # Create and save figure
    fig = plot_probability_comparison(true_probs, map_probs, ml_probs, N, 
                                    true_theta, theta_map, theta_ml, Sigma)
    plt.savefig('fig/fig_homogeneous_exp_fit.png', dpi=300, bbox_inches='tight')
    plt.show()