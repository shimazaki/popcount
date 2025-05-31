import numpy
#import model_homogeneous_exp_bayes as probability
import model_homogeneous_exp_em as probability
import generate_homogeneous_exp_samples as generate_samples

# Set numpy print options for cleaner output
numpy.set_printoptions(precision=3, suppress=True)

# --------------------------
# 1) Define true model & sample data
# --------------------------
N = 10
true_theta = [-2.5, 0.5, -0.2, 0.1] + [0.0]*(N-4)

def h(n):
    return 1      # here h(n) ≡ 1

print("\nTrue θ:", true_theta)
true_probs = probability.homogeneous_probabilities(N, true_theta, h)
print("True P(n):", true_probs)

# --------------------------
# 2) Sample data
# --------------------------
sample_size = 5000
samples = generate_samples.sample_counts(N, true_theta, h, size=sample_size)
print("\nFirst 20 samples:", samples[:20])
print("Empirical freq.:", 
        numpy.bincount(samples, minlength=N+1) / sample_size)

# --------------------------
# 3) MAP‐fit θ
# --------------------------
q = numpy.ones(N) * 10.0  # prior variances
theta0 = numpy.zeros(N)
q, theta_map, Sigma, res = probability.em_update(N, samples, h, q, theta0)
theta_est = theta_map
print("\nMAP‐Estimated θ:", theta_est)
print("Log‐posterior:", -res.fun)

# --------------------------
# 4) Show MAP probabilities
# --------------------------
est_probs = probability.homogeneous_probabilities(N, theta_est, h)
print("MAP P(n):", est_probs)
