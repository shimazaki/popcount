import numpy as np
import time
from numba import jit

@jit(nopython=True)
def simulate_lif_neurons(N=50, dt=0.05/1000, T=5, E_L=-70e-3, V_th=-50e-3, 
                           V_reset=-65e-3, g_L=5e-9, C_m=200e-12, I_base=30e-12, 
                           noise_amp=20e-12, c_in=0.4):
    """
    JIT-compiled version of LIF neuron simulation.
    
    Args:
        N (int): Number of neurons (default: 50)
        dt (float): Time step in seconds (default: 0.05 ms)
        T (float): Total simulation time in seconds (default: 5 s)
        E_L (float): Leak potential in Volts (default: -70 mV)
        V_th (float): Firing threshold in Volts (default: -50 mV)
        V_reset (float): Reset potential in Volts (default: -65 mV)
        g_L (float): Leak conductance in Siemens (default: 5 nS)
        C_m (float): Membrane capacitance in Farads (default: 200 pF)
        I_base (float): Base current in Amps (default: 30 pA)
        noise_amp (float): Noise amplitude in Amps (default: 20 pA)
        c_in (float): Input correlation coefficient (default: 0.4)
    
    Returns:
        tuple: (time, V, spike_times, neuron_indices)
            time: array of time points
            V: voltage matrix (N x time_steps)
            spike_times: array of spike times
            neuron_indices: array of neuron indices for each spike
    """
    # Time array
    time = np.arange(0, T, dt)
    num_steps = len(time)
    
    # Initialize voltage matrix and spike recording list
    V = np.zeros((N, num_steps))
    V[:, 0] = E_L
    spikes = []

    # Create covariance matrix and its Cholesky decomposition
    cov_matrix = np.ones((N, N)) * c_in
    for i in range(N):
        cov_matrix[i, i] = 1.0
    L = np.linalg.cholesky(cov_matrix)

    # Simulation Loop
    for i in range(1, num_steps):
        # Generate correlated noise for this time step
        I_noise = noise_amp * (L @ np.random.randn(N))
        
        # Update the membrane potential for all neurons at once (vectorized)
        dV = (-g_L * (V[:, i-1] - E_L) + I_noise) / C_m * dt + I_noise / C_m * np.sqrt(dt)
        V[:, i] = V[:, i-1] + dV

        # Find which neurons spiked and reset them
        spiked_neurons = V[:, i] > V_th
        V[spiked_neurons, i] = V_reset

        # Record the spikes for the raster plot
        for neuron_idx in range(N):
            if spiked_neurons[neuron_idx]:
                spikes.append((time[i], neuron_idx))

    # Convert spikes to numpy arrays
    if len(spikes) > 0:
        spike_times = np.zeros(len(spikes))
        neuron_indices = np.zeros(len(spikes), dtype=np.int64)
        for i, (t, n) in enumerate(spikes):
            spike_times[i] = t
            neuron_indices[i] = n
    else:
        spike_times = np.zeros(0)
        neuron_indices = np.zeros(0, dtype=np.int64)
        
    return time, V, spike_times, neuron_indices

def calculate_spike_statistics(spike_times, neuron_indices, N, T):
    """
    Calculate basic statistics of spike trains.
    
    Args:
        spike_times (array): Array of spike times
        neuron_indices (array): Array of neuron indices
        N (int): Total number of neurons
        T (float): Total simulation time in seconds
    
    Returns:
        dict: Dictionary containing spike statistics
    """
    stats = {}
    
    # Calculate firing rates for each neuron
    firing_rates = np.zeros(N)
    spike_counts = np.zeros(N)
    for n in range(N):
        neuron_spikes = spike_times[neuron_indices == n]
        spike_counts[n] = len(neuron_spikes)
        firing_rates[n] = len(neuron_spikes) / T  # Hz
    
    stats['mean_firing_rate'] = np.mean(firing_rates)
    stats['std_firing_rate'] = np.std(firing_rates)
    stats['min_firing_rate'] = np.min(firing_rates)
    stats['max_firing_rate'] = np.max(firing_rates)
    stats['total_spikes'] = len(spike_times)
    stats['mean_spikes_per_neuron'] = np.mean(spike_counts)
    stats['std_spikes_per_neuron'] = np.std(spike_counts)
    
    # Calculate ISI statistics for each neuron
    isi_mean = []
    isi_std = []
    for n in range(N):
        neuron_spikes = spike_times[neuron_indices == n]
        if len(neuron_spikes) > 1:
            isi = np.diff(neuron_spikes)
            isi_mean.append(np.mean(isi))
            isi_std.append(np.std(isi))
    
    if isi_mean:
        stats['mean_isi'] = np.mean(isi_mean)
        stats['std_isi'] = np.mean(isi_std)
    
    return stats

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # --- Parameters ---
    N = 50          # Number of neurons in the population
    dt = 0.05 / 1000      # Time step in second
    T = 5          # Total simulation time in second

    # Neuron Parameters
    C_m = 200e-12    # Capacitance in Farads (200 pF)
    E_L = -70e-3     # Leak potential in Volts (-70 mV)
    V_th = -50e-3    # Firing threshold in Volts (-50 mV)
    V_reset = -65e-3 # Reset potential in Volts (-65 mV)
    g_L = 5e-9       # Leak conductance in Siemens (5 nS)

    # Input and Noise
    I_base = 30e-12   # Base current in Amps
    noise_amp = 20e-12 # Noise amplitude in Amps
    c_in = 0.4      # Input correlation coefficient

    # Run JIT simulation
    print("Starting JIT simulation...")
    sim_start = time.time()
    time_array, V, spike_times, neuron_indices = simulate_lif_neurons(
        N, dt, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in
    )
    sim_end = time.time()
    print(f"JIT simulation completed in {sim_end - sim_start:.3f} seconds")

    # Calculate and display spike statistics
    print("\nSpike Statistics:")
    stats = calculate_spike_statistics(spike_times, neuron_indices, N, T)
    print(f"Mean firing rate: {stats['mean_firing_rate']:.2f} Hz")
    print(f"Std of firing rates: {stats['std_firing_rate']:.2f} Hz")
    print(f"Min firing rate: {stats['min_firing_rate']:.2f} Hz")
    print(f"Max firing rate: {stats['max_firing_rate']:.2f} Hz")
    print(f"Total number of spikes: {stats['total_spikes']}")
    print(f"Mean spikes per neuron: {stats['mean_spikes_per_neuron']:.2f}")
    print(f"Std of spikes per neuron: {stats['std_spikes_per_neuron']:.2f}")
    if 'mean_isi' in stats:
        print(f"Mean ISI: {stats['mean_isi']*1000:.2f} ms")
        print(f"Std of ISI: {stats['std_isi']*1000:.2f} ms")
