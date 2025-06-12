"""
Figure generation for Dichotomized Gaussian Model Analysis

This script generates figures for the Dichotomized Gaussian (DG) model analysis,
showing both theoretical and empirical distributions for different parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from model_dichotomized_gaussian import (
    calculate_pk_values, solve_theta_parameters,
    sample_patterns
)

def plot_raster_plots(N, h_values, alpha):
    """
    Generate raster plots for different threshold values using vertical lines for spikes.
    
    Parameters
    ----------
    N : int
        Number of neurons
    h_values : list
        List of threshold values to analyze
    alpha : float
        Input correlation parameter
    """
    fig = plt.figure(figsize=(12, 5))
    colors = ['blue', 'green', 'red']
    
    for i, (h, color) in enumerate(zip(h_values, colors)):
        ax_raster = fig.add_subplot(1, 3, i+1)
        q = norm.sf(h)
        patterns = sample_patterns(N, h, alpha, 1000)  # Increased to 1000 samples
        for sample_idx, pattern in enumerate(patterns):
            for neuron in range(N):
                if pattern[neuron]:
                    ax_raster.axvline(sample_idx, ymin=(neuron-0.4)/N, ymax=(neuron+0.4)/N, color=color, linewidth=1)
        ax_raster.set_xlim(-0.5, 999.5)  # Updated for 1000 samples
        ax_raster.set_ylim(-0.5, N - 0.5)
        ax_raster.set_title(f'Raster: h={h:.1f} (q={q:.3f})', fontsize=12)
        ax_raster.set_xlabel('Sample')
        ax_raster.set_ylabel('Neuron')
        ax_raster.set_xticks([0, 499, 999])  # Updated for 1000 samples
        ax_raster.set_yticks(range(0, N, 10))  # Show every 10th neuron for clarity
    
    fig.suptitle(f'Dichotomized Gaussian (DG) Model Raster Plots\nInput Correlation $c_{{in}} = {alpha}$', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('fig/fig_dichotomized_gaussian_raster.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution_analysis(N, h_values, alpha, n_samples=10000):
    """
    Generate P_k and theta parameter plots.
    
    Parameters
    ----------
    N : int
        Number of neurons
    h_values : list
        List of threshold values to analyze
    alpha : float
        Input correlation parameter
    n_samples : int, optional
        Number of samples for empirical distribution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['blue', 'green', 'red']
    
    for i, (h, color) in enumerate(zip(h_values, colors)):
        q = norm.sf(h)
        pk_values = calculate_pk_values(N, h, alpha)
        theta_params = solve_theta_parameters(pk_values, N)
        k_values = np.arange(N + 1)
        ax1.semilogy(k_values, pk_values, marker='o', linestyle='-', color=color, 
                label=f'h = {h:.1f} (q = {q:.3f})')
        ax1.set_title('Probability Distribution $P_k$ (log scale)', fontsize=14)
        ax1.set_xlabel('k (Number of active units)', fontsize=12)
        ax1.set_ylabel('Probability (log scale)', fontsize=12)
        ax1.set_xticks(k_values)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        j_values = np.arange(1, N + 1)
        ax2.plot(j_values, theta_params, marker='o', linestyle='-', color=color,
                label=f'h = {h:.1f} (q = {q:.3f})')
        ax2.set_title('Interaction Parameters $\\theta_j$', fontsize=14)
        ax2.set_xlabel('j (Order of interaction)', fontsize=12)
        ax2.set_ylabel(r'Parameter value $\theta_j$', fontsize=12)
        ax2.set_xticks(j_values)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
    
    fig.suptitle(f'Dichotomized Gaussian (DG) Model Analysis\nInput Correlation $c_{{in}} = {alpha}$', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('fig/fig_dichotomized_gaussian.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    N_raster = 100  # Number of neurons for raster plot
    N_dist = 10    # Number of neurons for distribution analysis
    alpha = 0.3    # correlation parameter
    h_values = [1.0, 1.5, 2.0]  # Different threshold values
    plot_raster_plots(N_raster, h_values, alpha)
    plot_distribution_analysis(N_dist, h_values, alpha) 