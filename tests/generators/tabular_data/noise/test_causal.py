"""Tests for CausalNoiseGenerator."""

import numpy as np
import pytest

from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator


def test_generate__noise_expands(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=3,
        noise_std=0.5,
    )
    # expand: n -> n * n_copies
    assert Xt.shape == (X.shape[0] * 3, 3)
    # All columns finite (noise, not missingness)
    assert np.all(np.isfinite(Xt))
    labels = gen.get_noise_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__n_copies_one_keeps_shape(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=1,
        noise_std=0.5,
    )
    assert Xt.shape == X.shape


def test_generate__noise_magnitude(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=1,
        noise_std=1.0,
    )
    # The injected noise (std ~ noise_std * col_std) is added directly to
    # the original values (delta propagation), so the marginal std of Xt
    # must be strictly inflated relative to the original.
    assert np.std(Xt[:, 0]) > np.std(X[:, 0])


def test_generate__noise_zero_recompute(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=1,
        noise_std=0.0,
    )
    # noise_std=0 -> no injected noise: the perturbed column is written
    # back unchanged (bit-identical). The DESCENDANTS are recomputed with
    # FRESH noise: statistically equivalent to the originals, not identical
    # (Pearl surgery, not delta propagation).
    np.testing.assert_array_equal(Xt[:, 0], X_orig[:, 0])
    dY = Xt[:, 1] - X_orig[:, 1]
    dZ = Xt[:, 2] - X_orig[:, 2]
    assert np.std(dY) > 0  # fresh noise, not a no-op
    np.testing.assert_allclose(np.mean(dY), 0.0, atol=0.1)
    np.testing.assert_allclose(np.mean(dZ), 0.0, atol=0.1)
    assert np.std(dY) < 0.5  # residual scale, not a structural shift


def test_generate__noise_propagates_recompute(chain_data, make_gcm_dgp):
    # Noise injected at X: descendants are RECOMPUTED from the noisy
    # parents with fresh noise (Pearl surgery), so the deterministic
    # component comes through exactly in expectation, not row-wise. The
    # oracle goes through the FITTED mechanism itself, so fitted-beta
    # noise cancels in expectation.
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(7))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=1,
        noise_std=0.5,
    )
    dX = Xt[:, 0] - X_orig[:, 0]
    dB = Xt[:, 1] - X_orig[:, 1]
    dC = Xt[:, 2] - X_orig[:, 2]
    dY_det = (
        gen.dgp.response("Y", Xt[:, [0]])
        - gen.dgp.response("Y", X_orig[:, [0]])
    )
    dZ_det = (
        gen.dgp.response("Z", Xt[:, [1]])
        - gen.dgp.response("Z", X_orig[:, [1]])
    )
    # Fresh-noise residual is zero-mean: the deterministic shift comes
    # through exactly in expectation.
    np.testing.assert_allclose(np.mean(dB - dY_det), 0.0, atol=0.1)
    np.testing.assert_allclose(np.mean(dC - dZ_det), 0.0, atol=0.1)
    # The fresh residual is small relative to the structural shift.
    assert np.std(dB - dY_det) < 0.3
    assert np.std(dC - dZ_det) < 0.3
    # Loose sanity: the fitted betas must be near the true coefficients.
    np.testing.assert_allclose(np.mean(dB), 2.0 * np.mean(dX), rtol=0.1)
    np.testing.assert_allclose(np.mean(dC), 0.5 * np.mean(dB), rtol=0.1)


def test_generate__noise_per_column(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X", "Y"], n_copies=1,
        noise_std=np.array([0.1, 1.0]),
    )
    # Y (col 1) gets more noise than X (col 0)
    assert np.std(Xt[:, 1]) > np.std(Xt[:, 0])


def test_generate__validation_errors(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator()
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(X, perturbation_nodes=["X"])
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    with pytest.raises(ValueError, match="n_copies"):
        gen.generate(X, perturbation_nodes=["X"], n_copies=0)


def test_generate__nan_input_does_not_poison(chain_data, make_gcm_dgp):
    # G9b: the noise scale is computed from the stage input. A NaN-bearing
    # perturbation column must not turn the WHOLE column NaN (NaN scale ->
    # NaN offset). NaN semantics: NaN rows stay NaN; the delta propagates
    # to descendants (Z becomes NaN where the Y delta was NaN) -- the same
    # NaN-cascade semantics as the missingness generator.
    X, graph, column_mapping = chain_data
    world_src = CausalNoiseGenerator(random_generator=np.random.default_rng(1))
    world_src.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:50, 1] = np.nan  # stage 1 (e.g. missingness) wrote NaNs into Y
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    n_nan_y = int(np.isnan(Xn[:, 1]).sum())
    Xt = gen.generate(Xn, perturbation_nodes=["Y"],
                         n_copies=1, noise_std=0.5)
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


def test_generate__all_nan_column_does_not_crash(chain_data, make_gcm_dgp):
    # G9b edge: a fully NaN column has no finite scale; the generator must
    # fall back to a finite scale (1.0) instead of producing NaN noise
    # (which would silently poison the column).
    X, graph, column_mapping = chain_data
    world_src = CausalNoiseGenerator(random_generator=np.random.default_rng(1))
    world_src.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(2))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:, 1] = np.nan
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    Xt = gen.generate(Xn, perturbation_nodes=["Y"],
                         n_copies=1, noise_std=0.5)
    # No crash; no NEW NaNs in the perturbed or upstream columns.
    assert int(np.isnan(Xt[:, 1]).sum()) == before.shape[0]
    np.testing.assert_array_equal(np.isnan(Xt[:, 0]), np.isnan(before[:, 0]))
    # Descendant Z is NaN exactly where the Y delta was NaN.
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), np.isnan(before[:, 1]))


from badgers.core.selectors import BernoulliSelector, TopQuantileSelector


def test_generate__selector_gates_expanded_rows(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalNoiseGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    # Bernoulli on the ORIGINAL rows: each seed row's copies either all
    # fire or all stay clean (masks replicated to the expansion)
    Xt = gen.generate(
        X, perturbation_nodes=["X"], n_copies=3, noise_std=0.5,
        selector=BernoulliSelector("X", rate=0.5, rng=gen.random_generator),
    )
    assert Xt.shape == (X.shape[0] * 3, 3)
    fired = gen.get_noise_labels()
    # groups of n_copies share the same state
    groups = fired.reshape(X.shape[0], 3)
    assert np.all(groups[:, 0] == groups[:, 1]) and np.all(groups[:, 1] == groups[:, 2])
    assert 0.2 < groups.mean() < 0.8
