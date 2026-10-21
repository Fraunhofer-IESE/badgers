"""Tests for AdditiveNoiseGenerator (non-causal)."""

import numpy as np
import pytest
import scipy.stats

from badgers.generators.tabular_data.noise.noise import AdditiveNoiseGenerator


@pytest.fixture
def gen():
    return AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))


def _simple_data(rng):
    return rng.normal(0, 1, size=(50, 3))


def test_generate__gaussian_matches_builtin(gen):
    """With scipy normal, additive noise behaves like the classic Gaussian generator."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.norm(scale=0.5), noise_std=1.0)
    assert Xt.shape == X.shape
    assert np.std(Xt - X) == pytest.approx(0.5, rel=0.2)


def test_generate__laplace_heavy_tail(gen):
    """Laplace noise (user-frozen distribution) is added as-is."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.laplace(scale=0.1), noise_std=1.0)
    delta = Xt - X
    # Laplace sample std = scale * sqrt(2)
    assert np.std(delta) == pytest.approx(0.1 * np.sqrt(2), rel=0.3)
    # Kurtosis > 3 (heavier tails than Gaussian)
    excess = scipy.stats.kurtosis(delta, axis=None)
    assert excess > 1.0


def test_generate__noise_std_multiplier(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt1 = gen.generate(X, distribution=scipy.stats.laplace(scale=1), noise_std=0.1)
    Xt2 = gen.generate(X, distribution=scipy.stats.laplace(scale=1), noise_std=0.0)
    assert np.allclose(Xt2, X)
    assert np.std(Xt1 - X) < 0.5


def test_generate__perturbation_nodes_subset(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.norm(scale=0.3),
                      perturbation_nodes=[1])
    # only column 1 changed
    assert np.allclose(Xt[:, 0], X[:, 0])
    assert np.allclose(Xt[:, 2], X[:, 2])
    assert not np.allclose(Xt[:, 1], X[:, 1])


def test_generate__invalid_distribution_raises(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    with pytest.raises(TypeError, match="frozen scipy"):
        gen.generate(X, distribution="laplace")
    with pytest.raises(TypeError, match="rvs"):
        gen.generate(X, distribution=np.random.default_rng(0))


def test_generate__negative_noise_std_raises(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    with pytest.raises(ValueError, match="noise_std"):
        gen.generate(X, distribution=scipy.stats.norm(), noise_std=-0.1)
