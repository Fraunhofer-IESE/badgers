"""Tests for CausalNoiseGenerator."""

import numpy as np
import pytest

from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator


def test_generate__noise_expands(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=3,
        mechanism="always", noise_std=0.5,
    )
    # expand: n -> n * n_copies
    assert Xt.shape == (X.shape[0] * 3, 3)
    # All columns finite (noise, not missingness)
    assert np.all(np.isfinite(Xt))
    assert list(yt) == ["noise"] * (X.shape[0] * 3)


def test_generate__n_copies_one_keeps_shape(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=0.5,
    )
    assert Xt.shape == X.shape


def test_generate__noise_magnitude(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=1.0,
    )
    # The injected noise (std ~ noise_std * col_std) is added directly to
    # the original values (delta propagation), so the marginal std of Xt
    # must be strictly inflated relative to the original.
    assert np.std(Xt[:, 0]) > np.std(X[:, 0])


def test_generate__noise_zero_is_identity(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=0.0,
    )
    # noise_std=0 -> no injected delta -> delta propagation adds exactly 0:
    # the output must equal the original data (exogenous noise preserved).
    np.testing.assert_allclose(Xt, X_orig, atol=1e-9)


def test_generate__noise_propagates_exact_delta(chain_data):
    # Noise injected at X on row i: delta_X = eps_X[i]. The delta model
    # requires delta_Y = beta_XY * delta_X and delta_Z = beta_YZ * delta_Y
    # EXACTLY (the nodes' own exogenous noise stays untouched). The oracle
    # goes through the FITTED mechanism itself (do()'s delta =
    # response(parents_new) - response(parents_orig)), so fitted-beta noise
    # cancels exactly.
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(7))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"], n_copies=1,
        mechanism="always", noise_std=0.5,
    )
    dX = Xt[:, 0] - X_orig[:, 0]
    dB = Xt[:, 1] - X_orig[:, 1]
    dC = Xt[:, 2] - X_orig[:, 2]
    dY_exact = (
        gen.dgp.response("Y", Xt[:, [0]])
        - gen.dgp.response("Y", X_orig[:, [0]])
    )
    np.testing.assert_allclose(dB, dY_exact, rtol=1e-12, atol=1e-12)
    dZ_exact = (
        gen.dgp.response("Z", Xt[:, [1]])
        - gen.dgp.response("Z", X_orig[:, [1]])
    )
    np.testing.assert_allclose(dC, dZ_exact, rtol=1e-12, atol=1e-12)
    # Loose sanity: the fitted betas must be near the true coefficients.
    np.testing.assert_allclose(dB, 2.0 * dX, rtol=0.05, atol=1e-12)
    np.testing.assert_allclose(dC, 0.5 * dB, rtol=0.05, atol=1e-12)


def test_generate__noise_per_column(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
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
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="n_copies"):
        gen.generate(X, y=None, perturbation_nodes=["X"], n_copies=0)
