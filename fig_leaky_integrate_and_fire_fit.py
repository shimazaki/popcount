import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from model_leaky_integrate_and_fire import simulate_lif_neurons
import model_homogeneous_exp as probability

def get_binary_spikes(spike_times, neuron_indices, dt=0.05, max_time=None):
    """
    Convert spike times to binary spike train.
    
    Args:
        spike_times (array): Array of spike times
        neuron_indices (array): Array of neuron indices for each spike
        dt (float): Time bin size in seconds
        max_time (float): Maximum time to consider
    
    Returns:
        tuple: (binary_spikes, time_bins)
            binary_spikes: Binary matrix of shape [time_bins, neurons]
            time_bins: Array of time points
    """
    if max_time is None:
        max_time = np.max(spike_times)
    
    # Number of time bins and neurons
    n_bins = int(np.ceil(max_time / dt))
    n_neurons = np.max(neuron_indices) + 1
    
    # Initialize binary spike matrix
    binary_spikes = np.zeros((n_bins, n_neurons))
    
    # Convert spike times to bin indices
    bin_indices = np.floor(spike_times / dt).astype(int)
    
    # Set spikes in binary matrix
    for t, n in zip(bin_indices, neuron_indices):
        if t < n_bins:  # Ensure we don't exceed array bounds
            binary_spikes[t, n] = 1
    
    # Create time bins array
    time_bins = np.arange(n_bins) * dt
    
    return binary_spikes, time_bins

def plot_homogeneous_model(pop_counts, est_probs_em, est_probs_ml, theta_map, theta_ml, N, K, save_path=None):
    """
    Plot the results of homogeneous model fitting.
    
    Args:
        pop_counts (array): Observed population spike counts
        est_probs_em (array): Estimated probabilities from EM (MAP)
        est_probs_ml (array): Estimated probabilities from MLE
        theta_map (array): Estimated θ values from EM (MAP)
        theta_ml (array): Estimated θ values from MLE
        N (int): Maximum count value
        K (int): Number of parameters
        save_path (str, optional): Path to save the figure
    """
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])  # Empirical vs Fitted
    ax2 = fig.add_subplot(gs[1])  # Theta values

    # Plotting
    ax1.bar(np.arange(N+1), np.bincount(pop_counts.astype(int), minlength=N+1)/len(pop_counts), alpha=0.5, label='Empirical')
    ax1.plot(np.arange(N+1), est_probs_em, 'o-', color='tab:orange', label='EM (MAP) fit')
    ax1.plot(np.arange(N+1), est_probs_ml, 'x--', color='tab:blue', label='MLE fit')
    ax1.set_xlabel('Population spike count')
    ax1.set_ylabel('Probability')
    ax1.legend()
    ax1.set_title('Empirical vs Fitted Homogeneous Model')
    ax2.plot(np.arange(1, K+1), theta_map, 'o-', color='tab:orange', label='EM (MAP)')
    ax2.plot(np.arange(1, K+1), theta_ml, 'x--', color='tab:blue', label='MLE')
    ax2.set_xlabel('Parameter index (k)')
    ax2.set_ylabel('θ value')
    ax2.set_title('Estimated θ values')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    #np.random.seed(42)
    N = 30  # Override default N
    T = 300  # Override default T
    dt = 0.02   # Bin size in seconds
    
    # Run simulation
    print("Running simulation...")
    time_array, V, spike_times, neuron_indices = simulate_lif_neurons(N=N, T=T)

    binary_spikes, _ = get_binary_spikes(spike_times, neuron_indices, dt=dt)

    # Compute pop_counts and fit model in main
    pop_counts = np.sum(binary_spikes, axis=1)
    
    # Fit homogeneous model
    K = 10
    theta0 = np.zeros(K)
    def h(n): return 1
    
    # EM (MAP) estimation
    theta_map, Sigma, q, res = probability.em_update(N, pop_counts.astype(int), h, K=K, theta0=theta0, max_iter=100)
    print("EM‐Estimated θ:", theta_map)
    print("Log‐likelihood (EM):", -res.fun)
    est_probs_em = probability.homogeneous_probabilities(N, theta_map, h)
    
    # MLE estimation
    result_ml = probability.estimate_ml_parameters(N, pop_counts.astype(int), h, K=K, theta0=theta0)
    theta_ml = result_ml.x
    print("MLE‐Estimated θ:", theta_ml)
    print("Log‐likelihood (MLE):", -result_ml.fun)
    est_probs_ml = probability.homogeneous_probabilities(N, theta_ml, h)
    
    # Plot results
    plot_homogeneous_model(pop_counts, est_probs_em, est_probs_ml, theta_map, theta_ml, N, K, 
                          save_path="fig/fig_leaky_integrate_and_fire_fit_homogeneous.png")
    