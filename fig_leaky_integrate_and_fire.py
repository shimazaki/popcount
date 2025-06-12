import numpy as np
import matplotlib.pyplot as plt

def plot_lif_neurons(time, V, spike_times, neuron_indices, N, V_th, example_neurons=None):
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

    ax1.set_title('LIF Neurons Raster Plot', fontsize=12)
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
    plt.show()

if __name__ == "__main__":
    # Import simulation function
    from model_leaky_integrate_and_fire import simulate_lif_neurons_jit_with_progress
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # --- Parameters ---
    N = 50          # Number of neurons in the population
    dt = 0.05 / 1000      # Time step in second
    T = 100          # Total simulation time in second

    # Neuron Parameters
    C_m = 200e-12    # Capacitance in Farads (200 pF)
    E_L = -70e-3     # Leak potential in Volts (-70 mV)
    V_th = -50e-3    # Firing threshold in Volts (-50 mV)
    V_reset = -65e-3 # Reset potential in Volts (-65 mV)
    g_L = 5e-9       # Leak conductance in Siemens (5 nS)

    # Input and Noise
    I_base = 30e-12   # Base current in Amps
    noise_amp = 10e-12 # Noise amplitude in Amps
    c_in = 0.4      # Input correlation coefficient

    # Run simulation
    print("Running simulation...")
    time_array, V, spike_times, neuron_indices = simulate_lif_neurons_jit_with_progress(
        N, dt, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_lif_neurons(time_array, V, spike_times, neuron_indices, N, V_th) 


