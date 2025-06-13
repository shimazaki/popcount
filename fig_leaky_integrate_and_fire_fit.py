import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from model_leaky_integrate_and_fire import simulate_lif_neurons
import model_homogeneous_exp as probability

def get_binary_spikes(spike_times_per_neuron, dt=0.05, max_time=None):
    """
    Convert spike times per neuron to binary spike trains.
    
    Args:
        spike_times_per_neuron (list of arrays): List of arrays, where each array contains the spike times for a single neuron
        dt (float): Time step in seconds
        max_time (float or None): Maximum time to consider
    
    Returns:
        tuple: (binary_spikes, time_points)
            binary_spikes: Binary spike matrix (time_steps x N)
            time_points: Array of time points
    """
    if max_time is None:
        max_time = max(max(spikes) for spikes in spike_times_per_neuron if len(spikes) > 0)
    
    time_points = np.arange(0, max_time + dt, dt)
    N = len(spike_times_per_neuron)
    binary_spikes = np.zeros((len(time_points), N))
    
    for neuron_idx, spikes in enumerate(spike_times_per_neuron):
        for spike_time in spikes:
            spike_idx = int(spike_time / dt)
            if spike_idx < len(time_points):
                binary_spikes[spike_idx, neuron_idx] = 1
    
    return binary_spikes, time_points

def plot_homogeneous_model(pop_counts, est_probs_em, est_probs_ml, theta_map, theta_ml, N, K, dt, params, save_path=None):
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
        params (tuple): Simulation parameters for title
        save_path (str, optional): Path to save the figure
    """
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])  # Empirical vs Fitted
    ax2 = fig.add_subplot(gs[1])  # Theta values

    # Ensure pop_counts is an integer array
    pop_counts_int = pop_counts.astype(int)

    # Debug prints
    print("Shape of pop_counts_int:", pop_counts_int.shape)
    print("Content of pop_counts_int:", pop_counts_int)

    # Plotting
    ax1.bar(np.arange(N+1), np.bincount(pop_counts_int, minlength=N+1)/len(pop_counts_int), alpha=0.5, label='Empirical')
    ax1.plot(np.arange(N+1), est_probs_em, 'o-', color='tab:orange', label='EM (MAP) fit')
    ax1.plot(np.arange(N+1), est_probs_ml, 'x--', color='tab:blue', label='MLE fit')
    ax1.set_xlabel('Population spike count')
    ax1.set_ylabel('Probability')
    ax1.legend()
    # Unpack params tuple for use in the plot title
    N, dt_s, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in, seed = params
    title = f'Empirical vs Fitted Homogeneous Model\n dt={dt*1000:.0f}ms, g_L={g_L*1e9:.1f}nS, I_base={I_base*1e12:.0f}pA, noise_amp={noise_amp*1e12:.0f}pA, c_in={c_in:.1f}'
    ax1.set_title(title, fontsize=12)
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
    np.random.seed(42)
    N = 10  # Override default N
    T = 3000  # Override default T
    dt = 0.05   # Bin size in seconds

    # Neuron Parameters
    #C_m = 200e-12    # Capacitance in Farads (200 pF)
    #E_L = -70e-3     # Leak potential in Volts (-70 mV)
    #V_th = -50e-3    # Firing threshold in Volts (-50 mV)
    #V_reset = -65e-3 # Reset potential in Volts (-65 mV)
    g_L = .1e-9       # Leak conductance in Siemens (10 nS)

    # Input and Noise
    I_base = 0e-12   # Base current in Amps
    noise_amp = 20e-12 # Noise amplitude in Amps
    c_in = 0.3      # Input correlation coefficient

    # Run simulation
    print("Running simulation...")
    spike_times_per_neuron, time_array, V, params  = simulate_lif_neurons(
        N=N, T=T, g_L = g_L, I_base = I_base, noise_amp = noise_amp, c_in = c_in)

    # Convert spike times to binary spike trains
    binary_spikes, time_points = get_binary_spikes(spike_times_per_neuron, dt=0.05)
    print("Shape of binary_spikes:", binary_spikes.shape)
    print("Shape of time_points:", time_points.shape)

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
    plot_homogeneous_model(pop_counts, est_probs_em, est_probs_ml, theta_map, theta_ml, N, K, dt, params,
                          save_path="fig/fig_leaky_integrate_and_fire_fit_homogeneous.png")
    