"""
Visualization tools for the Population Count Model

Author: Hideaki Shimazaki
"""

import numpy as np
import matplotlib.pyplot as plt
import model_homogeneous_exp as probability

# Define parameters
N = 10
true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

def h(n):
    return 1  # here h(n) ≡ 1

# Compute true probabilities
true_probs = probability.homogeneous_probabilities(N, true_theta, h)

# Generate samples and fit using MAP with EM
sample_size = 5000
samples = probability.sample_counts(N, true_theta, h, size=sample_size)
q = np.ones(N) * 10.0  # prior variances
theta0 = np.zeros(N)
q, theta_map, Sigma, res = probability.em_update(N, samples, h, q, theta0)
est_probs = probability.homogeneous_probabilities(N, theta_map, h)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Left subplot - linear scale probabilities
ax1.plot(true_probs, label='True probabilities')
ax1.plot(est_probs, label='Estimated probabilities')
ax1.legend()
ax1.set_xlim(0, N)
ax1.set_title('Linear Scale')

# Middle subplot - log scale probabilities
ax2.plot(true_probs, label='True probabilities')
ax2.plot(est_probs, label='Estimated probabilities')
ax2.legend()
ax2.set_xlim(0, N)
ax2.set_yscale('log')
ax2.set_title('Log Scale')

# Right subplot - theta comparison
ax3.plot(range(1, N+1), true_theta, 'o-', label='True theta')
ax3.plot(range(1, N+1), theta_map, 'x-', label='Estimated theta')
ax3.legend()
ax3.set_xlabel('k')
ax3.set_title('Theta Comparison')

plt.tight_layout()
plt.savefig('fig/probabilities.png')
plt.show()