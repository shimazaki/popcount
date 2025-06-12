import numpy as np
import matplotlib.pyplot as plt
import os
from model_leaky_integrate_and_fire import simulate_lif_neurons

def plot_lif_neurons(time, V, spike_times, neuron_indices, N, V_th, example_neurons=None, save_path=None):
    """
    Plot LIF neuron simulation results.
    
    Args:
        time (array): Time points
        V (array): Voltage matrix (N x time_steps)
        spike_times (array): Array of spike times
        neuron_indices (array): Array of neuron indices for each spike
        N (int): Number of neurons
        V_th (float): Firing threshold in Volts
        example_neurons (list): List of neuron indices to plot in detail (default: [0, 1])
        save_path (str or None): If given, save the figure to this path.
    """
    if example_neurons is None:
        example_neurons = [0, 1]
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                  gridspec_kw={'height_ratios': [3, 1]})

    # 1. Raster Plot
    if len(spike_times) > 0:
        for spike_time, neuron_idx in zip(spike_times, neuron_indices):
            ax1.axvline(spike_time, ymin=(neuron_idx-0.4)/N, ymax=(neuron_idx+0.4)/N, 
                       color='black', linewidth=1)

    ax1.set_title('LIF Neurons Raster Plot\nN=50, V_th=-50mV, V_reset=-65mV, E_L=-70mV, g_L=5nS, C_m=200pF, I_base=30pA, noise_amp=20pA, c_in=0.4', fontsize=12)
    ax1.set_ylabel('Neuron Index')
    ax1.set_ylim(-0.5, N - 0.5)
    ax1.set_xlim(time[0], time[-1])
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_yticks(range(0, N, 10))  # Show every 10th neuron for clarity

    # 2. Example Neurons Membrane Potential
    colors = ['b', 'g']  # Colors for each neuron
    for i, n in enumerate(example_neurons):
        color = colors[i]
        ax2.plot(time, V[n, :] * 1e3, linewidth=0.5, label=f'Neuron {n}', color=color)
        # Add vertical ticks for spikes using actual spike times
        if len(spike_times) > 0:
            neuron_spikes = spike_times[neuron_indices == n]
            print(f"Neuron {n} spikes at times: {neuron_spikes}")  # Debug print
            if len(neuron_spikes) > 0:
                ax2.vlines(neuron_spikes, ymin=V_th * 1e3, ymax=V_th * 1e3 + 10, 
                          colors=color, linewidth=1.0, alpha=0.8)
    ax2.axhline(y=V_th * 1e3, color='k', linestyle='--', label='Threshold')
    ax2.set_title('Example Neurons')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.close()  # Close the figure instead of showing it

def get_binary_spikes(spike_times, neuron_indices, dt=0.05, max_time=None):
    if max_time is None:
        max_time = np.max(spike_times)
    n_bins = int(np.ceil(max_time / dt))
    n_neurons = np.max(neuron_indices) + 1
    binary_spikes = np.zeros((n_bins, n_neurons))
    bin_indices = np.floor(spike_times / dt).astype(int)
    for t, n in zip(bin_indices, neuron_indices):
        if t < n_bins:
            binary_spikes[t, n] = 1
    time_bins = np.arange(n_bins) * dt
    return binary_spikes, time_bins

def plot_spike_analysis(time, V, spike_times, neuron_indices, N, binary_spikes, dt=0.05, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # Raster plot
    if len(spike_times) > 0:
        for spike_time, neuron_idx in zip(spike_times, neuron_indices):
            ax1.axvline(spike_time, ymin=(neuron_idx-0.)/N, ymax=(neuron_idx+1.)/N, color='black', linewidth=1)
    ax1.set_title('LIF Neurons Raster Plot\nN=50, V_th=-50mV, V_reset=-65mV, E_L=-70mV, g_L=5nS, C_m=200pF, I_base=30pA, noise_amp=20pA, c_in=0.4', fontsize=12)
    ax1.set_ylabel('Neuron Index')
    ax1.set_ylim(0, N)
    ax1.set_xlim(time[0], time[-1])
    if N <= 20:
        ax1.set_yticks(np.arange(0, N))
    else:
        ax1.set_yticks(np.arange(0, N, max(1, N // 10)))
    bin_edges = np.arange(0, time[-1] + dt, dt)
    for edge in bin_edges:
        ax1.axvline(edge, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # Binary spike train
    for n in range(N):
        spike_bins = np.where(binary_spikes[:, n] == 1)[0]
        if len(spike_bins) > 0:
            spike_times_bin = spike_bins * dt
            ax2.vlines(spike_times_bin, n, n + 1, color='blue', linewidth=0.5)
    ax2.set_ylabel('Neuron')
    ax2.set_title(f'Binary Spike Train (dt = {dt*1000:.1f} ms)')
    ax2.set_ylim(0, N)
    if N <= 20:
        ax2.set_yticks(np.arange(0, N))
    else:
        ax2.set_yticks(np.arange(0, N, max(1, N // 10)))
    for edge in bin_edges:
        ax2.axvline(edge, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    plt.close()  # Close the figure instead of showing it

if __name__ == "__main__":
    np.random.seed(42)
    
    # Run simulation
    print("Running simulation...")
    time_array, V, spike_times, neuron_indices = simulate_lif_neurons()
    print("\nPlotting and saving results...")
    plot_lif_neurons(time_array, V, spike_times, neuron_indices, N=50, V_th=-50e-3, save_path="fig/fig_leaky_integrate_and_fire.png")
    binary_spikes, _ = get_binary_spikes(spike_times, neuron_indices)
    plot_spike_analysis(time_array, V, spike_times, neuron_indices, N=50, binary_spikes=binary_spikes, save_path="fig/fig_leaky_integrate_and_fire_data.png")


