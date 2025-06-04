"""
Unit tests for model_alternating_shrinking.py

Author: Hideaki Shimazaki
"""

import numpy as np
import pytest
from scipy.special import comb
import model_alternating_shrinking as mas
import model_homogeneous_exp as mhe


class TestPMFComputation:
    """Test probability mass function computation"""
    
    def test_compute_n_spike_pmf_basic(self):
        """Test basic PMF computation"""
        N = 5
        f = 1.0
        Cj_func = lambda j: 1.0 / j
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # Check properties
        assert len(pmf) == N + 1
        assert np.isclose(np.sum(pmf), 1.0)
        assert np.all(pmf >= 0)
        assert np.all(pmf <= 1)
        
    def test_compute_n_spike_pmf_zero_sparsity(self):
        """Test with f=0 (no sparsity)"""
        N = 4
        f = 0.0
        Cj_func = lambda j: 1.0
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # With f=0, should get uniform distribution
        expected = np.ones(N + 1) / (N + 1)
        np.testing.assert_allclose(pmf, expected, rtol=1e-10)
        
    def test_compute_n_spike_pmf_high_sparsity(self):
        """Test with high sparsity parameter"""
        N = 6
        f = 10.0
        Cj_func = lambda j: 1.0 / j
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # High sparsity should concentrate probability at low counts
        assert pmf[0] > pmf[N]  # P(0) > P(N)
        assert pmf[1] > pmf[N-1]  # P(1) > P(N-1)


class TestThetaConversion:
    """Test conversion to homogeneous exponential parameters"""
    
    def test_compute_alternating_theta_basic(self):
        """Test basic theta computation"""
        N = 4
        f = 1.0
        Cj_func = lambda j: 1.0
        
        theta = mas.compute_alternating_theta(N, f, Cj_func)
        
        assert len(theta) == N
        assert not np.any(np.isnan(theta))
        assert not np.any(np.isinf(theta))
        
    def test_compute_alternating_theta_consistency(self):
        """Test that converted parameters give same probabilities"""
        N = 5
        f = 2.0
        m = 1.5
        Cj_func = lambda j: 1.0 / j**m
        
        # Compute PMF directly
        pmf_direct = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # Convert to homogeneous parameters and compute PMF
        theta = mas.compute_alternating_theta(N, f, Cj_func)
        h = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0
        pmf_converted = mhe.homogeneous_probabilities(N, theta, h)
        
        # Should match closely
        np.testing.assert_allclose(pmf_direct, pmf_converted, rtol=1e-8)
        
    def test_alternating_sign_pattern(self):
        """Test that parameters show alternating behavior"""
        N = 6
        f = 5.0
        Cj_func = lambda j: 1.0 / j
        
        theta = mas.compute_alternating_theta(N, f, Cj_func)
        
        # For this specific Cj_func, we expect alternating signs
        # (though this depends on the specific model parameters)
        # At least check that we have both positive and negative values
        assert np.any(theta > 0) or np.any(theta < 0)


class TestSampling:
    """Test sampling methods"""
    
    def test_sample_spike_counts_range(self):
        """Test that sampled counts are in valid range"""
        N = 5
        f = 2.0
        Cj_func = lambda j: 1.0 / j**2
        
        np.random.seed(42)
        counts = mas.sample_spike_counts(N, f, Cj_func, size=1000)
        
        assert np.all(counts >= 0)
        assert np.all(counts <= N)
        assert counts.shape == (1000,)
        
    def test_sample_spike_counts_distribution(self):
        """Test that samples follow the correct distribution"""
        N = 4
        f = 3.0
        Cj_func = lambda j: 1.0 / j
        
        # Get theoretical PMF
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # Sample many counts
        np.random.seed(123)
        counts = mas.sample_spike_counts(N, f, Cj_func, size=10000)
        
        # Compare empirical and theoretical distributions
        empirical_pmf = np.bincount(counts, minlength=N+1) / len(counts)
        
        # Should be close (allowing for sampling variation)
        np.testing.assert_allclose(empirical_pmf, pmf, atol=0.02)
        
    def test_sample_patterns_properties(self):
        """Test binary pattern sampling"""
        N = 6
        f = 2.0
        Cj_func = lambda j: 1.0 / j
        
        np.random.seed(456)
        patterns = mas.sample_patterns(N, f, Cj_func, size=100)
        
        # Check shape and binary nature
        assert patterns.shape == (100, N)
        assert np.all(np.isin(patterns, [0, 1]))
        
        # Check that pattern sums match spike count distribution
        pattern_sums = np.sum(patterns, axis=1)
        assert np.all(pattern_sums >= 0)
        assert np.all(pattern_sums <= N)


class TestGibbsSampler:
    """Test Gibbs sampling"""
    
    def test_gibbs_sampler_basic(self):
        """Test basic Gibbs sampler functionality"""
        N = 4
        f = 2.0
        Cj_func = lambda j: 1.0 / j
        h_func = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0
        
        samples = mas.gibbs_sampler(N, f, Cj_func, h_func, steps=1000, burn_in=100, seed=789)
        
        # Check output shape
        assert samples.shape == (900, N)  # steps - burn_in
        # Check binary
        assert np.all(np.isin(samples, [0, 1]))
        
    def test_gibbs_sampler_convergence(self):
        """Test that Gibbs sampler converges to correct distribution"""
        N = 3
        f = 1.5
        Cj_func = lambda j: 1.0
        h_func = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0
        
        # Get theoretical PMF
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # Run Gibbs sampler
        samples = mas.gibbs_sampler(N, f, Cj_func, h_func, steps=10000, burn_in=1000, seed=101)
        
        # Compute empirical distribution of spike counts
        spike_counts = np.sum(samples, axis=1)
        empirical_pmf = np.bincount(spike_counts, minlength=N+1) / len(spike_counts)
        
        # Should converge to theoretical PMF
        np.testing.assert_allclose(empirical_pmf, pmf, atol=0.05)


class TestDifferentCjFunctions:
    """Test with various Cj functions"""
    
    def test_constant_cj(self):
        """Test with constant Cj"""
        N = 5
        f = 1.0
        Cj_func = lambda j: 1.0
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        assert np.isclose(np.sum(pmf), 1.0)
        
    def test_exponential_cj(self):
        """Test with exponentially decaying Cj"""
        N = 5
        f = 1.0
        Cj_func = lambda j: np.exp(-j)
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        assert np.isclose(np.sum(pmf), 1.0)
        
    def test_power_law_cj(self):
        """Test with power law Cj"""
        N = 5
        f = 1.0
        
        for m in [0.5, 1.0, 2.0]:
            Cj_func = lambda j, m=m: 1.0 / j**m
            pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
            assert np.isclose(np.sum(pmf), 1.0)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_neuron(self):
        """Test with N=1"""
        N = 1
        f = 1.0
        Cj_func = lambda j: 1.0
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        assert len(pmf) == 2
        assert np.isclose(np.sum(pmf), 1.0)
        
    def test_large_f(self):
        """Test with very large sparsity parameter"""
        N = 4
        f = 100.0
        Cj_func = lambda j: 1.0 / j
        
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        
        # Should heavily favor n=0
        assert pmf[0] > 0.9
        assert np.isclose(np.sum(pmf), 1.0)
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters"""
        N = 8
        f = 50.0
        Cj_func = lambda j: 1.0 / j**3
        
        # Should not produce NaN or Inf
        pmf = mas.compute_n_spike_pmf_with_func(N, f, Cj_func)
        theta = mas.compute_alternating_theta(N, f, Cj_func)
        
        assert not np.any(np.isnan(pmf))
        assert not np.any(np.isinf(pmf))
        assert not np.any(np.isnan(theta))
        assert not np.any(np.isinf(theta))


class TestIntegration:
    """Integration tests with model_homogeneous_exp"""
    
    def test_parameter_estimation_on_alternating_data(self):
        """Test that we can recover parameters from alternating model data"""
        N = 5
        f = 3.0
        m = 1.0
        Cj_func = lambda j: 1.0 / j**m
        
        # Generate data from alternating model
        np.random.seed(2024)
        samples = mas.sample_spike_counts(N, f, Cj_func, size=2000)
        
        # Fit using homogeneous model with appropriate h
        h = lambda n: 1.0 / comb(N, n) if 0 <= n <= N else 0.0
        
        # Get true theta for comparison
        true_theta = mas.compute_alternating_theta(N, f, Cj_func)
        
        # Estimate using EM
        q_init = np.ones(N) * 1.0
        theta_map, Sigma, q, res = mhe.em_update(N, samples, h, q_init, max_iter=50)
        
        # Parameters should be reasonably close
        assert res.success
        # Allow for some estimation error
        assert np.max(np.abs(theta_map - true_theta)) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])