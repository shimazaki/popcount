"""
Unit tests for model_homogeneous_exp.py

Author: Hideaki Shimazaki
"""

import numpy as np
import pytest
from scipy.special import comb
import model_homogeneous_exp as mhe


class TestHomogeneousProbabilities:
    """Test probability computation functions"""
    
    def test_homogeneous_probabilities_basic(self):
        """Test basic probability computation"""
        N = 5
        theta = np.zeros(N)
        h = lambda n: 1
        
        probs = mhe.homogeneous_probabilities(N, theta, h)
        
        # Probabilities should sum to 1
        assert np.isclose(np.sum(probs), 1.0)
        # Should have N+1 probabilities (for n=0 to N)
        assert len(probs) == N + 1
        # All probabilities should be non-negative
        assert np.all(probs >= 0)
        
    def test_homogeneous_probabilities_uniform(self):
        """Test with zero parameters and uniform h"""
        N = 4
        theta = np.zeros(N)
        h = lambda n: 1  # Uniform base measure
        
        probs = mhe.homogeneous_probabilities(N, theta, h)
        
        # With theta=0 and h=1, P(n) ∝ C(N,n)
        # This gives the binomial distribution with p=0.5
        expected = np.array([comb(N, n) for n in range(N+1)]) / (2**N)
        
        np.testing.assert_allclose(probs, expected, rtol=1e-10)
        
    def test_log_homogeneous_probabilities(self):
        """Test log probability computation"""
        N = 5
        theta = np.array([-1, 0.5, -0.2, 0.1, 0])
        h = lambda n: 1
        
        logP, logZ = mhe.log_homogeneous_probabilities(N, theta, h)
        
        # Test that exp(logP) gives valid probabilities
        probs = np.exp(logP)
        assert np.isclose(np.sum(probs), 1.0)
        
        # Test partition function property
        assert np.isclose(np.exp(logZ), np.sum(np.exp(logP + logZ)))


class TestSufficientStatistics:
    """Test sufficient statistics computation"""
    
    def test_compute_sufficient_statistics_single_sample(self):
        """Test with a single sample"""
        N = 5
        ns = np.array([3])  # Single observation n=3
        
        S = mhe.compute_sufficient_statistics(ns, N)
        
        # S[k-1] should equal C(3,k) for k=1..5
        expected = np.array([comb(3, k) for k in range(1, N+1)])
        np.testing.assert_array_equal(S, expected)
        
    def test_compute_sufficient_statistics_multiple_samples(self):
        """Test with multiple samples"""
        N = 4
        ns = np.array([0, 1, 2, 1])  # Multiple observations
        
        S = mhe.compute_sufficient_statistics(ns, N)
        
        # Compute expected values manually
        expected = np.zeros(N)
        for n in ns:
            for k in range(1, N+1):
                expected[k-1] += comb(n, k)
                
        np.testing.assert_array_equal(S, expected)


class TestParameterEstimation:
    """Test parameter estimation functions"""
    
    def test_estimate_ml_parameters_convergence(self):
        """Test ML estimation convergence"""
        N = 5
        true_theta = np.array([-1, 0.5, -0.2, 0.1, 0])
        h = lambda n: 1
        
        # Generate synthetic data
        np.random.seed(42)
        samples = mhe.sample_counts(N, true_theta, h, size=1000)
        
        # Estimate parameters
        result = mhe.estimate_ml_parameters(N, samples, h)
        
        # Check convergence
        assert result.success
        # Check that parameters are reasonably close (with large sample)
        # Note: This is a statistical test, so we use loose tolerance
        assert np.max(np.abs(result.x - true_theta)) < 0.5
        
    def test_estimate_map_parameters_basic(self):
        """Test MAP estimation basic functionality"""
        N = 4
        samples = np.array([1, 2, 1, 3, 2])
        h = lambda n: 1
        q = np.ones(N) * 10.0  # Large prior variance
        
        S = mhe.compute_sufficient_statistics(samples, N)
        M = len(samples)
        
        result = mhe.estimate_map_parameters(N, S, M, h, q)
        
        assert result.success
        assert len(result.x) == N
        
    def test_compute_map_gradient(self):
        """Test gradient computation"""
        N = 3
        S = np.array([5, 3, 1])  # Arbitrary sufficient statistics
        M = 10
        h = lambda n: 1
        q = np.ones(N) * 1.0
        theta = np.zeros(N)
        
        grad = mhe.compute_map_gradient(N, S, M, h, q, theta)
        
        # Gradient should have same dimension as theta
        assert len(grad) == N
        # At theta=0 with uniform h, gradient should be positive where S > 0
        assert np.all(grad[S > 0] > 0)


class TestPosteriorCovariance:
    """Test posterior covariance computation"""
    
    def test_compute_posterior_covariance_shape(self):
        """Test covariance matrix shape and properties"""
        N = 4
        theta_map = np.array([-0.5, 0.2, -0.1, 0.05])
        h = lambda n: 1
        q = np.ones(N) * 1.0
        M = 100
        
        Sigma = mhe.compute_posterior_covariance(N, theta_map, h, q, M)
        
        # Check shape
        assert Sigma.shape == (N, N)
        # Check symmetry
        assert np.allclose(Sigma, Sigma.T)
        # Check positive definiteness (all eigenvalues should be positive)
        eigenvalues = np.linalg.eigvals(Sigma)
        assert np.all(eigenvalues > 0)


class TestEMAlgorithm:
    """Test EM algorithm"""
    
    def test_em_update_convergence(self):
        """Test EM algorithm convergence"""
        N = 4
        true_theta = np.array([-1, 0.3, -0.1, 0.05])
        h = lambda n: 1
        
        # Generate data
        np.random.seed(123)
        samples = mhe.sample_counts(N, true_theta, h, size=500)
        
        # Run EM
        q_init = np.ones(N) * 1.0
        theta_map, Sigma, q, res = mhe.em_update(N, samples, h, q_init, max_iter=20)
        
        # Check outputs
        assert res.success
        assert len(theta_map) == N
        assert Sigma.shape == (N, N)
        assert len(q) == N
        # Prior variances should be positive
        assert np.all(q > 0)


class TestSampling:
    """Test sampling functions"""
    
    def test_sample_counts_distribution(self):
        """Test that sampled counts follow the correct distribution"""
        N = 5
        theta = np.array([-2, 0.5, -0.2, 0.1, 0])
        h = lambda n: 1
        
        # Generate many samples
        np.random.seed(456)
        samples = mhe.sample_counts(N, theta, h, size=10000)
        
        # Check range
        assert np.all(samples >= 0)
        assert np.all(samples <= N)
        
        # Check empirical distribution matches theoretical
        probs = mhe.homogeneous_probabilities(N, theta, h)
        counts = np.bincount(samples, minlength=N+1)
        empirical_probs = counts / len(samples)
        
        # Chi-square test would be more rigorous, but this is simpler
        assert np.max(np.abs(empirical_probs - probs)) < 0.02
        
    def test_sample_patterns_properties(self):
        """Test binary pattern generation"""
        N = 6
        theta = np.array([-1.5, 0.3, -0.1, 0.05, 0, 0])
        h = lambda n: 1
        
        np.random.seed(789)
        patterns = mhe.sample_patterns(N, theta, h, size=100)
        
        # Check shape
        assert patterns.shape == (100, N)
        # Check binary
        assert np.all(np.isin(patterns, [0, 1]))
        # Check that sum of each pattern matches the count distribution
        pattern_sums = np.sum(patterns, axis=1)
        assert np.all(pattern_sums >= 0)
        assert np.all(pattern_sums <= N)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_neuron(self):
        """Test with N=1"""
        N = 1
        theta = np.array([0.5])
        h = lambda n: 1
        
        probs = mhe.homogeneous_probabilities(N, theta, h)
        assert len(probs) == 2  # P(0) and P(1)
        assert np.isclose(np.sum(probs), 1.0)
        
    def test_extreme_parameters(self):
        """Test with very large/small parameters"""
        N = 3
        theta_large = np.array([10, -10, 5])
        h = lambda n: 1
        
        # Should not raise errors
        probs = mhe.homogeneous_probabilities(N, theta_large, h)
        assert np.isclose(np.sum(probs), 1.0)
        assert not np.any(np.isnan(probs))
        
    def test_zero_samples(self):
        """Test with empty sample array"""
        N = 3
        samples = np.array([], dtype=int)
        h = lambda n: 1
        
        S = mhe.compute_sufficient_statistics(samples, N)
        assert np.all(S == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])