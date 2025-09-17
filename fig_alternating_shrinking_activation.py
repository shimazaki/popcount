import numpy as np
from scipy.special import comb
import scipy.special as sp
import matplotlib.pyplot as plt
import model_alternating_shrinking
import model_homogeneous_exp as probability

def activation_theta(n, N, theta):
    """
    Compute P(x_i=1 | sum_{j≠i} x_j = n) - the probability that neuron i is active
    given that n other neurons are active.
    
    Parameters:
    -----------
    n : int or array-like
        Number of other active neurons (can be single value or array)
    N : int
        Total number of neurons
    theta : array-like
        Model parameters with theta[k-1] = θ_k for k=1,...,N
        
    Returns:
    --------
    float or array
        Activation probabilities for each value of n
    """
    theta = np.asarray(theta, dtype=float)
    if len(theta) != N:
        raise ValueError("theta must have length N with theta[k-1] = θ_k.")
    
    # Convert input to array for consistent processing
    # n_values will contain the number of other active neurons
    n_values = np.atleast_1d(n).astype(int)
    if (n_values < 0).any() or (n_values > N-1).any():
        raise ValueError("n must be in [0, N-1].")

    # Store activation probabilities for each n
    activation_probs = np.empty_like(n_values, dtype=float)
    
    for i, n_val in enumerate(n_values):
        # Compute log-odds for activation
        log_odds = -np.log((N - n_val) / (n_val + 1)) + theta[n_val]
        
        # Add interaction terms if there are other active neurons
        if n_val > 0:
            log_odds += sum(comb(n_val, k - 1) * theta[k - 1] 
                           for k in range(1, n_val + 1))
        
        # Convert log-odds to probability
        activation_probs[i] = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Return single value if input was scalar, otherwise return array
    return activation_probs[0] if np.isscalar(n) else activation_probs

def tilde_h(n, N):
    n_arr = np.atleast_1d(n).astype(int)
    val = np.log((n_arr + 1) / (N - n_arr))
    return val[0] if np.isscalar(n) else val

def tilde_Q(n, theta):
    theta = np.asarray(theta, float)
    n_arr = np.atleast_1d(n).astype(int)
    out = np.empty_like(n_arr, dtype=float)
    for i, nn in enumerate(n_arr):
        s = theta[nn]  # θ_{n+1}
        if nn:
            s += sum(comb(nn, k - 1) * theta[k - 1] for k in range(1, nn + 1))
        out[i] = s
    return out[0] if np.isscalar(n) else out

def plot_activation(N, theta, spike_probs, pairwise_theta, pairwise_spike_probs, save_path=None, dpi=300):
    n = np.arange(N)
    p = activation_theta(n, N, theta)
    h_vals = tilde_h(n, N)
    Q_vals = tilde_Q(n, theta)
    
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 0: Spike count probability distribution
    n_spike = np.arange(N + 1)
    ax0.plot(n_spike, spike_probs, linewidth=2, color='purple', label='Alternating-shrinking model')
    # ax0.plot(n_spike, pairwise_spike_probs, '--', linewidth=2, color='orange', label='Pairwise model (h(n)=1)')
    
    ax0.set_xlabel("# of active neurons")
    ax0.set_ylabel("P(n)")
    ax0.set_title("Spike Count Distribution")
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.text(-0.10, 1.05, 'a', transform=ax0.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    # Panel 1: Activation probability
    ax1.plot(n, p, linewidth=2, color='blue', label='Alternating-shrinking model')
    # Add pairwise model with h=1 activation function
    pairwise_p = activation_theta(n, N, pairwise_theta)
    ax1.plot(n, pairwise_p, '--', linewidth=2, color='orange', label='Pairwise model (h(n)=1)')
    ax1.set_xlabel("# of active input units")
    ax1.set_ylabel(r"P(x_i=1 | n)")
    ax1.set_title(r"Activation Function $P(x_i=1 | n)$")
    ax1.text(-0.10, 1.05, 'b', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: tilde_Q
    ax2.plot(n, Q_vals, linewidth=2, color='green', label='Alternating-shrinking model')
    # Add pairwise model Q̃(n) - only theta[0] and theta[1] terms
    pairwise_Q = np.array([pairwise_theta[0] + pairwise_theta[1] * n_val for n_val in n])
    ax2.plot(n, pairwise_Q, '--', linewidth=2, color='orange', label='Pairwise model')
    ax2.set_xlabel("# of active input units")
    ax2.set_ylabel(r"$\tilde{Q}(n)$")
    ax2.set_title(r"Polynomial term $\tilde{Q}(n)$")
    ax2.text(-0.10, 1.05, 'c', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: tilde_h
    ax3.plot(n, h_vals, linewidth=2, color='red', label=r'h(n) = 1 \left/ \binom{N}{n} \right.')
    # Add h=1 reference line
    ax3.axhline(y=0, color='orange', linestyle='--', linewidth=2, label='h(n) = 1')
    ax3.set_xlabel("# of active input units")
    ax3.set_ylabel(r"$\tilde{h}(n)$")
    ax3.set_title(r"Base Measure Function $\tilde{h}(n)$")
    ax3.text(-0.10, 1.05, 'd', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example parameters
    N = 30
    
    #Polylogarithmic exponential distribution
    #f = 50.0
    # m = 1.0
    # Cj_func = lambda j: 1 / j**m

    #Shifted-geometric exponential distribution
    f = 80#80 #30.0
    tau = 0.9#0.8 #0.8
    Cj_func = lambda j: tau**j

    # Convert Cj to theta using the existing function
    Cj_original = [Cj_func(j) for j in range(1, N+1)]
    theta = model_alternating_shrinking.cj_to_theta(Cj_original, f)
    theta = np.array(theta)  # Convert to numpy array
    
    # Compute spike count probability distribution
    spike_probs = model_alternating_shrinking.compute_n_spike_pmf_with_func(N, f, Cj_func)
    
    # # Fit pairwise model to spike_probs data using maximum likelihood
    # # Generate samples analytically from spike_probs distribution
    # h_func = lambda n: 1.0  # h(n) = 1 for pairwise model
    
    # # Create samples by sampling from the multinomial distribution
    # # Each count n appears with probability spike_probs[n]
    # n_samples = 10000  # Number of samples to generate
    # sample_counts = np.random.choice(N + 1, size=n_samples, p=spike_probs)
    
    # # Use the existing ML estimation function
    # result_ml = probability.estimate_ml_parameters(N, sample_counts, h_func, K=2)
    # pairwise_theta = np.array([result_ml.x[0], result_ml.x[1]] + [0] * (N-2))

    # Use first and second theta values from alternating model for pairwise model
    pairwise_theta = np.array([theta[0], theta[1]] + [0] * (N-2))
    
    # Compute pairwise model probabilities
    h_func = lambda n: 1.0  # h(n) = 1 for pairwise model
    pairwise_spike_probs = probability.homogeneous_probabilities(N, pairwise_theta, h_func)
    
    print(f"Plotting activation for N={N} neurons")
    print(f"Theta values: {theta}")
    print(f"Pairwise theta values: {pairwise_theta}")
    
    # Save plot to PNG file
    import os
    os.makedirs('fig', exist_ok=True)
    plot_activation(N, theta, spike_probs, pairwise_theta, pairwise_spike_probs, save_path='fig/fig_alternating_shrinking_activation.png')
    plot_activation(N, theta, spike_probs, pairwise_theta, pairwise_spike_probs, dpi=300, save_path='fig/fig_rodriguez.eps')
