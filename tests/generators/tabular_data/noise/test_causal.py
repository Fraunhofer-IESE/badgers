"""Tests for CausalNoiseGenerator."""

import numpy as np
import pytest

from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator


def test_generate__noise_expands(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=3,
        mechanism="always", noise_std=0.5,
    )
    # expand: n -> n * n_copies
    assert Xt.shape == (X.shape[0] * 3, 3)
    # All columns finite (noise, not missingness)
    assert np.all(np.isfinite(Xt))
    assert list(yt) == ["noise"] * (X.shape[0] * 3)


def test_generate__n_copies_one_is_inplace(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=0.5,
    )
    assert Xt.shape == X.shape


def test_generate__noise_magnitude(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=1.0,
    )
    # The injected noise (std ~ noise_std * col_std) is added ON TOP of the
    # DGP forward-pass recompute of X, so the marginal std of Xt must be
    # strictly inflated relative to the original. (The exact total also
    # depends on the fitted latent std, so we check inflation, not equality.)
    assert np.std(Xt[:, 0]) > np.std(X[:, 0])


def test_generate__noise_per_column(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X", "Y"], n_copies=1,
        mechanism="always", noise_std=np.array([0.1, 1.0]),
    )
    # Y (col 1) gets more noise than X (col 0)
    assert np.std(Xt[:, 1]) > np.std(Xt[:, 0])


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator()
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    with pytest.raises(ValueError, match="n_copies"):
        gen.generate(X, y=None, perturbation_nodes=["X"], n_copies=0)
