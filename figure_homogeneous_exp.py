import numpy as np
import matplotlib.pyplot as plt
import example_homogeneous_exp_bayes

true_probs = example_homogeneous_exp_bayes.true_probs
est_probs = example_homogeneous_exp_bayes.est_probs
N = example_homogeneous_exp_bayes.N

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
ax3.plot(range(1, N+1), example_homogeneous_exp_bayes.true_theta, 'o-', label='True theta')
ax3.plot(range(1, N+1), example_homogeneous_exp_bayes.theta_est, 'x-', label='Estimated theta')
ax3.legend()
ax3.set_xlabel('k')
ax3.set_title('Theta Comparison')



plt.tight_layout()
plt.savefig('fig/probabilities.png')
plt.show()