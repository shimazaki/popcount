import numpy
import model_homogeneous_exp as probability
import generate_homogeneous_exp_samples as generate_samples

# Set numpy print options for cleaner output
numpy.set_printoptions(precision=3, suppress=True)

# Parameters
N = 10
theta = [-3.5, 0.2, -0.1] + [0.0] * (N - 3)
def h(n): return 1  # example h(n)=n

# Show true model probabilities
print("\nTrue theta:", theta)
true_probs = probability.homogeneous_probabilities(N, theta, h)
print("True probabilities:", true_probs)

# Draw samples
sample_size = 5000
samples = generate_samples.sample_counts(N, theta, h, size=sample_size)
print("\nFirst 20 samples:", samples[:20])
print("Empirical frequencies:", 
        numpy.bincount(samples, minlength=N+1) / sample_size)

# Fit theta
theta0 = numpy.zeros(N)
result = probability.fit_theta_sufficient(N, samples, h, theta0)
theta_est = result.x
print("\nEstimated theta:", theta_est)
print("Log-likelihood:", -result.fun)

# Show estimated probabilities
est_probs = probability.homogeneous_probabilities(N, theta_est, h)
print("Estimated probabilities:", est_probs)