"""Tests for badgers.core.sampling — Sampler classes and factory."""

import numpy as np
import pytest

from badgers.core.sampling import (
    Sampler,
    NormalSampler,
    UniformSampler,
    ZScoreSampler,
    HypersphereSampler,
    UniformOutOfDistributionSampler,
    create_within_distribution_sampler,
    create_out_of_distribution_sampler,
    _WITHIN_DISTRIBUTION_REGISTRY,
    _OUT_OF_DISTRIBUTION_REGISTRY,
)


# --- Fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X(rng):
    """Simple 2D dataset with known statistics."""
    return rng.normal(loc=[10.0, -5.0], scale=[2.0, 0.5], size=(200, 2))


# --- Sampler base class ---

def test_sampler_is_abstract():
    """Sampler base class should raise NotImplementedError."""
    sampler = Sampler()
    with pytest.raises(NotImplementedError):
        sampler.sample(None, None, None, 10)


# --- NormalSampler ---

def test_normal_sampler_shape(rng, X):
    sampler = NormalSampler()
    result = sampler.sample(rng, X, [0, 1], 50)
    assert result.shape == (50, 2)


def test_normal_sampler_mean_approx(rng, X):
    sampler = NormalSampler()
    result = sampler.sample(rng, X, [0, 1], 10000)
    # Sample mean should be close to data mean
    assert np.allclose(np.mean(result, axis=0), np.mean(X, axis=0), atol=0.1)


def test_normal_sampler_std_approx(rng, X):
    sampler = NormalSampler()
    result = sampler.sample(rng, X, [0, 1], 10000)
    # Sample std should be close to data std
    assert np.allclose(np.std(result, axis=0), np.std(X, axis=0), atol=0.1)


def test_normal_sampler_single_column(rng, X):
    sampler = NormalSampler()
    result = sampler.sample(rng, X, [0], 30)
    assert result.shape == (30, 1)


# --- UniformSampler ---

def test_uniform_sampler_shape(rng, X):
    sampler = UniformSampler()
    result = sampler.sample(rng, X, [0, 1], 50)
    assert result.shape == (50, 2)


def test_uniform_sampler_within_bounds(rng, X):
    sampler = UniformSampler()
    result = sampler.sample(rng, X, [0, 1], 200)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    assert np.all(result >= mins)
    assert np.all(result <= maxs)


def test_uniform_sampler_single_column(rng, X):
    sampler = UniformSampler()
    result = sampler.sample(rng, X, [1], 30)
    assert result.shape == (30, 1)


# --- ZScoreSampler ---

def test_zscore_sampler_shape(rng, X):
    sampler = ZScoreSampler()
    result = sampler.sample(rng, X, [0, 1], 50)
    assert result.shape == (50, 2)


def test_zscore_sampler_far_from_mean(rng, X):
    sampler = ZScoreSampler(scale=1.0)
    result = sampler.sample(rng, X, [0, 1], 100)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # All samples should be at least 3 sigma away from the mean
    z_scores = np.abs((result - means) / stds)
    assert np.all(z_scores >= 3.0)


def test_zscore_sampler_single_column(rng, X):
    sampler = ZScoreSampler(scale=0.5)
    result = sampler.sample(rng, X, [0], 30)
    assert result.shape == (30, 1)
    mean = np.mean(X[:, 0])
    std = np.std(X[:, 0])
    z_scores = np.abs((result[:, 0] - mean) / std)
    assert np.all(z_scores >= 3.0)


# --- HypersphereSampler ---

def test_hypersphere_sampler_shape(rng, X):
    sampler = HypersphereSampler()
    result = sampler.sample(rng, X, [0, 1], 50)
    assert result.shape == (50, 2)


def test_hypersphere_sampler_far_from_mean(rng, X):
    sampler = HypersphereSampler(scale=1.0)
    result = sampler.sample(rng, X, [0, 1], 100)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # All samples should be at least 3 sigma away in Euclidean distance
    distances = np.linalg.norm((result - means) / stds, axis=1)
    assert np.all(distances >= 3.0)


def test_hypersphere_sampler_single_column(rng, X):
    sampler = HypersphereSampler(scale=0.5)
    result = sampler.sample(rng, X, [0], 30)
    assert result.shape == (30, 1)
    mean = np.mean(X[:, 0])
    std = np.std(X[:, 0])
    z_scores = np.abs((result[:, 0] - mean) / std)
    assert np.all(z_scores >= 3.0)


# --- UniformOutOfDistributionSampler ---

def test_uniform_ood_sampler_shape(rng, X):
    sampler = UniformOutOfDistributionSampler(expansion=0.5)
    result = sampler.sample(rng, X, [0, 1], 50)
    assert result.shape == (50, 2)


def test_uniform_ood_sampler_beyond_bounds(rng, X):
    sampler = UniformOutOfDistributionSampler(expansion=0.5)
    result = sampler.sample(rng, X, [0, 1], 200)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    ranges = maxs - mins
    expanded_min = mins - 0.5 * ranges
    expanded_max = maxs + 0.5 * ranges
    assert np.all(result >= expanded_min)
    assert np.all(result <= expanded_max)
    # At least some samples should be outside the original range
    assert np.any((result < mins) | (result > maxs))


def test_uniform_ood_sampler_no_expansion_raises():
    with pytest.raises(ValueError, match="expansion must be > 0"):
        UniformOutOfDistributionSampler(expansion=0)


def test_uniform_ood_sampler_negative_expansion_raises():
    with pytest.raises(ValueError, match="expansion must be > 0"):
        UniformOutOfDistributionSampler(expansion=-0.5)


def test_uniform_ood_sampler_single_column(rng, X):
    sampler = UniformOutOfDistributionSampler(expansion=0.3)
    result = sampler.sample(rng, X, [1], 30)
    assert result.shape == (30, 1)


# --- create_within_distribution_sampler factory ---

def test_create_within_distribution_sampler_normal():
    sampler = create_within_distribution_sampler("normal")
    assert isinstance(sampler, NormalSampler)


def test_create_within_distribution_sampler_uniform():
    sampler = create_within_distribution_sampler("uniform")
    assert isinstance(sampler, UniformSampler)


def test_create_within_distribution_sampler_unknown_raises():
    with pytest.raises(ValueError, match="Unknown within-distribution sampler"):
        create_within_distribution_sampler("nonexistent")


# --- create_out_of_distribution_sampler factory ---

def test_create_out_of_distribution_sampler_zscore():
    sampler = create_out_of_distribution_sampler("zscore", scale=2.0)
    assert isinstance(sampler, ZScoreSampler)
    assert sampler.scale == 2.0


def test_create_out_of_distribution_sampler_hypersphere():
    sampler = create_out_of_distribution_sampler("hypersphere", scale=1.5)
    assert isinstance(sampler, HypersphereSampler)
    assert sampler.scale == 1.5


def test_create_out_of_distribution_sampler_uniform():
    sampler = create_out_of_distribution_sampler("uniform", expansion=0.3)
    assert isinstance(sampler, UniformOutOfDistributionSampler)
    assert sampler.expansion == 0.3


def test_create_out_of_distribution_sampler_unknown_raises():
    with pytest.raises(ValueError, match="Unknown out-of-distribution sampler"):
        create_out_of_distribution_sampler("nonexistent")


def test_create_out_of_distribution_sampler_default_kwargs():
    sampler = create_out_of_distribution_sampler("zscore")
    assert sampler.scale == 1.0


# --- Registries ---

def test_within_distribution_registry():
    assert "normal" in _WITHIN_DISTRIBUTION_REGISTRY
    assert "uniform" in _WITHIN_DISTRIBUTION_REGISTRY


def test_out_of_distribution_registry():
    assert "zscore" in _OUT_OF_DISTRIBUTION_REGISTRY
    assert "hypersphere" in _OUT_OF_DISTRIBUTION_REGISTRY
    assert "uniform" in _OUT_OF_DISTRIBUTION_REGISTRY
