"""Tests for CausalNoiseGenerator."""

import numpy as np
import pytest

from badgers.core.dgp import OlsDGP
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


def test_generate__nan_input_does_not_poison(chain_data):
    # G9b: the noise scale is computed from the stage input. A NaN-bearing
    # perturbation column must not turn the WHOLE column NaN (NaN scale ->
    # NaN offset). NaN semantics: NaN rows stay NaN; the delta propagates
    # to descendants (Z becomes NaN where the Y delta was NaN) -- the same
    # NaN-cascade semantics as the missingness generator.
    X, graph, column_mapping = chain_data
    world_src = CausalNoiseGenerator(random_generator=np.random.default_rng(1))
    world_src.fit(X, graph, column_mapping, dgp=OlsDGP(graph, column_mapping))
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:50, 1] = np.nan  # stage 1 (e.g. missingness) wrote NaNs into Y
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    n_nan_y = int(np.isnan(Xn[:, 1]).sum())
    Xt, _ = gen.generate(Xn, y=None, perturbation_nodes=["Y"],
                         n_copies=1, mechanism="always", noise_std=0.5)
    # The perturbed column gains NO new NaN cells (finite scale).
    assert int(np.isnan(Xt[:, 1]).sum()) == n_nan_y
    # Column X (untouched upstream) gains no NaNs either.
    np.testing.assert_array_equal(np.isnan(Xt[:, 0]), np.isnan(before[:, 0]))
    # Finite Y rows received the noise.
    finite = np.isfinite(Xt[:, 1])
    assert finite.any()
    assert not np.allclose(Xt[finite, 1], before[finite, 1])
    # Descendant Z: NaN exactly where the Y delta was NaN (Y was NaN) or
    # Z was already NaN -- no OTHER new NaNs.
    expected_z_nan = np.isnan(before[:, 2]) | np.isnan(before[:, 1])
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), expected_z_nan)


def test_generate__all_nan_column_does_not_crash(chain_data):
    # G9b edge: a fully NaN column has no finite scale; the generator must
    # fall back to a finite scale (1.0) instead of producing NaN noise
    # (which would silently poison the column).
    X, graph, column_mapping = chain_data
    world_src = CausalNoiseGenerator(random_generator=np.random.default_rng(1))
    world_src.fit(X, graph, column_mapping, dgp=OlsDGP(graph, column_mapping))
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(2))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:, 1] = np.nan
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    Xt, _ = gen.generate(Xn, y=None, perturbation_nodes=["Y"],
                         n_copies=1, mechanism="always", noise_std=0.5)
    # No crash; no NEW NaNs in the perturbed or upstream columns.
    assert int(np.isnan(Xt[:, 1]).sum()) == before.shape[0]
    np.testing.assert_array_equal(np.isnan(Xt[:, 0]), np.isnan(before[:, 0]))
    # Descendant Z is NaN exactly where the Y delta was NaN.
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), np.isnan(before[:, 1]))
