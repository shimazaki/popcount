import numpy as np
from model_homogeneous_exp import (
    homogeneous_probabilities,
    log_homogeneous_probabilities,
    sample_counts,
    sample_patterns,
    compute_map_gradient,
    estimate_map_parameters,
    compute_posterior_covariance,
    em_update,
    estimate_ml_parameters
)

def test_basic_functions():
    print("Testing basic functions...")
    N = 10
    K = 5
    theta = np.random.randn(K)
    
    # Test with default h
    print("\n1. Testing with default h (None):")
    P1 = homogeneous_probabilities(N, theta)
    logP1, logZ1 = log_homogeneous_probabilities(N, K, theta)
    counts1 = sample_counts(N, theta, size=5)
    patterns1 = sample_patterns(N, theta, size=5)
    
    print("Probabilities shape:", P1.shape)
    print("Log probabilities shape:", logP1.shape)
    print("Sample counts:", counts1)
    print("Sample patterns shape:", patterns1.shape)
    
    # Test with custom h
    print("\n2. Testing with custom h:")
    h = lambda n: 1.0 / (n + 1)
    P2 = homogeneous_probabilities(N, theta, h)
    logP2, logZ2 = log_homogeneous_probabilities(N, K, theta, h)
    counts2 = sample_counts(N, theta, h, size=5)
    patterns2 = sample_patterns(N, theta, h, size=5)
    
    print("Probabilities shape:", P2.shape)
    print("Log probabilities shape:", logP2.shape)
    print("Sample counts:", counts2)
    print("Sample patterns shape:", patterns2.shape)

def test_estimation_functions():
    print("\nTesting estimation functions...")
    N = 10
    K = 5
    M = 1000
    
    # Generate some test data
    theta_true = np.random.randn(K)
    samples = sample_counts(N, theta_true, size=M)
    S = np.array([sum(np.array([np.math.comb(n, k) for n in samples])) for k in range(1, K+1)])
    
    # Test MAP estimation
    print("\n1. Testing MAP estimation:")
    res = estimate_map_parameters(N, K, S, M)
    print("MAP estimation successful:", res.success)
    
    # Test posterior covariance
    print("\n2. Testing posterior covariance:")
    Sigma = compute_posterior_covariance(N, K, res.x)
    print("Covariance matrix shape:", Sigma.shape)
    
    # Test EM update
    print("\n3. Testing EM update:")
    theta_est, Sigma, q, res = em_update(N, samples)
    print("EM estimation successful:", res.success)
    print("Final q values:", q)
    
    # Test ML estimation
    print("\n4. Testing ML estimation:")
    theta_ml, res = estimate_ml_parameters(N, samples)
    print("ML estimation successful:", res.success)

if __name__ == "__main__":
    np.random.seed(42)
    test_basic_functions()
    test_estimation_functions() 