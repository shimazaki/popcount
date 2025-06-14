import numpy as np
import time
from numba import jit

@jit(nopython=True)
def simulate_lif_neurons(N=50, dt_s=0.05/1000, T=5, E_L=-70e-3, V_th=-50e-3, 
                           V_reset=-65e-3, g_L=10e-9, C_m=200e-12, I_base=0e-12, 
                           noise_amp=20e-12, c_in=0.3, seed=None):
    """
    JIT-compiled version of LIF neuron simulation.
    
    Args:
        N (int): Number of neurons (default: 50)
        dt_s (float): Time step in seconds (default: 0.05 ms)
        T (float): Total simulation time in seconds (default: 5 s)
        E_L (float): Leak potential in Volts (default: -70 mV)
        V_th (float): Firing threshold in Volts (default: -50 mV)
        V_reset (float): Reset potential in Volts (default: -65 mV)
        g_L (float): Leak conductance in Siemens (default: 10 nS)
        C_m (float): Membrane capacitance in Farads (default: 200 pF)
        I_base (float): Base current in Amps (default: 0 pA)
        noise_amp (float): Noise amplitude in Amps (default: 15 pA)
        c_in (float): Input correlation coefficient (default: 0.3)
        seed (int or None): Random seed for reproducibility (default: None)
    
    Returns:
        tuple: (spike_times_per_neuron, time, V, (N, dt, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in, seed))
            spike_times_per_neuron: list of arrays, where each array contains the spike times for a single neuron
            time: array of time points
            V: voltage matrix (N x time_steps)
            (N, dt, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in, seed): tuple of input parameters
    """
    # Set random seed for Numba if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Time array
    time = np.arange(0, T, dt_s)
    num_steps = len(time)
    
    # Initialize voltage matrix with random initial conditions
    V = np.zeros((N, num_steps))
    V[:, 0] = E_L + 10e-3 * np.random.randn(N)  # Random initial conditions with 10 mV std
    # Ensure initial voltages are below threshold
    V[:, 0] = np.minimum(V[:, 0], V_th - 1e-3)
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
        dV = (-g_L * (V[:, i-1] - E_L) + I_base) / C_m * dt_s + I_noise / C_m * np.sqrt(dt_s)
        V[:, i] = V[:, i-1] + dV

        # Find which neurons spiked and reset them
        spiked_neurons = V[:, i] > V_th
        V[spiked_neurons, i] = V_reset

        # Record the spikes for the raster plot
        for neuron_idx in range(N):
            if spiked_neurons[neuron_idx]:
                spikes.append((time[i], neuron_idx))

    # Convert spikes to a list of arrays, where each array contains the spike times for a single neuron
    spike_times_per_neuron = [np.array([t for t, n in spikes if n == i]) for i in range(N)]
    
    params = (N, dt_s, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in, seed)
    return spike_times_per_neuron, time, V, params

def calculate_spike_statistics(spike_times_per_neuron, N, T):
    """
    Calculate basic statistics of spike trains from spike_times_per_neuron.
    Args:
        spike_times_per_neuron (list of arrays): List of arrays, where each array contains the spike times for a single neuron
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
        neuron_spikes = spike_times_per_neuron[n]
        spike_counts[n] = len(neuron_spikes)
        firing_rates[n] = len(neuron_spikes) / T  # Hz
    stats['mean_firing_rate'] = np.mean(firing_rates)
    stats['std_firing_rate'] = np.std(firing_rates)
    stats['min_firing_rate'] = np.min(firing_rates)
    stats['max_firing_rate'] = np.max(firing_rates)
    stats['total_spikes'] = int(np.sum(spike_counts))
    stats['mean_spikes_per_neuron'] = np.mean(spike_counts)
    stats['std_spikes_per_neuron'] = np.std(spike_counts)
    # Calculate ISI statistics for each neuron
    isi_mean = []
    isi_std = []
    for n in range(N):
        neuron_spikes = spike_times_per_neuron[n]
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
    dt_s = 0.001 / 1000 #0.05 / 1000      # Time step in second
    T = 1          # Total simulation time in second

    # Neuron Parameters
    C_m = 200e-12    # Capacitance in Farads (200 pF)
    E_L = -70e-3     # Leak potential in Volts (-70 mV)
    V_th = -50e-3    # Firing threshold in Volts (-50 mV)
    V_reset = -65e-3 # Reset potential in Volts (-65 mV)
    g_L = 10e-9       # Leak conductance in Siemens (10 nS)

    # Input and Noise
    I_base = 0e-12   # Base current in Amps
    noise_amp = 20e-12 # Noise amplitude in Amps
    c_in = 0.3      # Input correlation coefficient

    # Run JIT simulation
    print("Starting simulation...")
    sim_start = time.time()
    spike_times_per_neuron, time_array, V, params = simulate_lif_neurons(
        N, dt_s, T, E_L, V_th, V_reset, g_L, C_m, I_base, noise_amp, c_in, seed=42
    )
    sim_end = time.time()
    print(f"Simulation completed in {sim_end - sim_start:.3f} seconds")

    # Calculate and display spike statistics
    print("\nSpike Statistics:")
    stats = calculate_spike_statistics(spike_times_per_neuron, N, T)
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
