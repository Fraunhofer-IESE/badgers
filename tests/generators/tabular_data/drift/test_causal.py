"""Tests for CausalDriftGenerator (delta propagation semantics).

Drift semantics: a single systematic shift (drawn once, applied to every
affected row) at the perturbation node, propagated to the descendants as a
delta: descendant_new = descendant_orig + beta @ delta_parents. Each node's
own exogenous noise is preserved exactly.
"""

import numpy as np
import pytest

from badgers.generators.tabular_data.drift.causal import CausalDriftGenerator


def test_generate__drift_systematic_shift(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        drift_std=1.0,
    )
    # in-place: same shape
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))
    labels = gen.get_drift_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__drift_shift_is_constant_across_rows(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        drift_std=1.0,
    )
    # Drift is systematic: the shift is drawn ONCE and applied identically to
    # every affected row. With the default selector all rows are affected, so
    # the per-row change at the perturbation node must be a single constant.
    shift = Xt[:, 0] - X_orig[:, 0]
    assert np.allclose(shift, shift[0], atol=1e-8)


def test_generate__drift_zero_shift_recompute(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        drift_std=0.0,
    )
    # drift_std=0 -> shift of exactly 0: the perturbed column is written
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


def test_generate__drift_propagates_recompute(chain_data, make_gcm_dgp):
    # Known-coefficients chain: Y = 2 X + eps_Y, Z = 0.5 Y + eps_Z.
    # Shifting X by a constant c must shift Y by beta_XY * c in expectation
    # and Z by beta_YZ * beta_XY * c: the descendants are RECOMPUTED with
    # fresh noise, so the deterministic shift comes through exactly on the
    # mean, not row-wise. The oracle goes through the FITTED mechanism
    # itself, so fitted-beta noise cancels in expectation.
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(3))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        drift_std=1.0,
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
    # Loose sanity: the fitted betas must be near the true coefficients
    # (chain_data noise std is 0.1 -> SE(beta) ~ 0.01).
    np.testing.assert_allclose(np.mean(dB), 2.0 * np.mean(dX), rtol=0.1)
    np.testing.assert_allclose(np.mean(dC), 0.5 * np.mean(dB), rtol=0.1)


def test_generate__drift_magnitude(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        drift_std=2.0,
    )
    # The systematic shift is drawn once (std ~ drift_std * col_std); its
    # sampled value can be small, so assert on a generous bound (a tighter
    # bound would be seed-flaky).
    shift = abs(Xt[0, 0] - X_orig[0, 0])
    assert shift > 0.05 * (2.0 * np.std(X_orig[:, 0]))


def test_generate__drift_closed_form_exact(make_gcm_dgp):
    # Noise-free structural equations: least squares recovers beta exactly,
    # so the closed-form oracle holds to machine precision.
    import networkx as nx
    rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = rng.normal(size=100)
    data = np.column_stack([X, 2.0 * X, 0.5 * (2.0 * X)])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    data_orig = data.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(3))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(data))
    Xt = gen.generate(
        data, perturbation_nodes=["X"],
        drift_std=1.0,
    )
    np.testing.assert_allclose(
        Xt[:, 1] - data_orig[:, 1], 2.0 * (Xt[:, 0] - data_orig[:, 0]), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        Xt[:, 2] - data_orig[:, 2], 1.0 * (Xt[:, 0] - data_orig[:, 0]), rtol=1e-10, atol=1e-12
    )


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator()
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(X, perturbation_nodes=["X"])


def test_generate__nan_input_does_not_poison(chain_data, make_gcm_dgp):
    # G9b: the drift scale is computed from the stage input. A NaN-bearing
    # perturbation column must not turn the WHOLE column NaN (NaN scale ->
    # NaN shift). NaN semantics: NaN rows stay NaN; the delta propagates to
    # descendants (Z becomes NaN where the Y delta was NaN) -- the same
    # NaN-cascade semantics as the missingness generator.
    X, graph, column_mapping = chain_data
    world_src = CausalDriftGenerator(random_generator=np.random.default_rng(2))
    world_src.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(1))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:50, 1] = np.nan  # stage 1 (e.g. missingness) wrote NaNs into Y
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    n_nan_y = int(np.isnan(Xn[:, 1]).sum())
    Xt = gen.generate(Xn, perturbation_nodes=["Y"],
                         drift_std=0.5)
    # The perturbed column gains NO new NaN cells (finite scale).
    assert int(np.isnan(Xt[:, 1]).sum()) == n_nan_y
    # Column X (untouched upstream) gains no NaNs either.
    np.testing.assert_array_equal(np.isnan(Xt[:, 0]), np.isnan(before[:, 0]))
    # Finite Y rows received the shift.
    finite = np.isfinite(Xt[:, 1])
    assert finite.any()
    assert not np.allclose(Xt[finite, 1], before[finite, 1])
    # Descendant Z: NaN exactly where the Y delta was NaN (Y was NaN) or
    # Z was already NaN -- no OTHER new NaNs.
    expected_z_nan = np.isnan(before[:, 2]) | np.isnan(before[:, 1])
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), expected_z_nan)


def test_generate__all_nan_column_does_not_crash(chain_data, make_gcm_dgp):
    # G9b edge: a fully NaN column has no finite scale; the generator must
    # fall back to a finite scale (1.0) instead of producing a NaN shift
    # (which would silently poison the column).
    X, graph, column_mapping = chain_data
    world_src = CausalDriftGenerator(random_generator=np.random.default_rng(2))
    world_src.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(3))
    gen.use_dgp(world_src.dgp)
    Xn = X.copy()
    Xn[:, 1] = np.nan
    before = Xn.copy()  # snapshot BEFORE generate (in-place generator)
    Xt = gen.generate(Xn, perturbation_nodes=["Y"],
                         drift_std=0.5)
    # No crash; no NEW NaNs in the perturbed or upstream columns.
    assert int(np.isnan(Xt[:, 1]).sum()) == before.shape[0]
    np.testing.assert_array_equal(np.isnan(Xt[:, 0]), np.isnan(before[:, 0]))
    # Descendant Z is NaN exactly where the Y delta was NaN.
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), np.isnan(before[:, 1]))
