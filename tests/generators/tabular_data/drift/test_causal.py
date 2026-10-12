"""Tests for CausalDriftGenerator (delta propagation semantics).

Drift semantics: a single systematic shift (drawn once, applied to every
affected row) at the perturbation node, propagated to the descendants as a
delta: descendant_new = descendant_orig + beta @ delta_parents. Each node's
own exogenous noise is preserved exactly.
"""

import numpy as np
import pytest

from badgers.generators.tabular_data.drift.causal import CausalDriftGenerator


def test_generate__drift_systematic_shift(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
    )
    # in-place: same shape
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))
    assert list(yt) == ["drift"] * X.shape[0]


def test_generate__drift_shift_is_constant_across_rows(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
    )
    # Drift is systematic: the shift is drawn ONCE and applied identically to
    # every affected row. With mechanism="always" all rows are affected, so
    # the per-row change at the perturbation node must be a single constant.
    shift = Xt[:, 0] - X_orig[:, 0]
    assert np.allclose(shift, shift[0], atol=1e-8)


def test_generate__drift_zero_shift_leaves_data_untouched(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=0.0,
    )
    # drift_std=0 -> shift of exactly 0 -> delta propagation adds exactly 0:
    # the output must equal the original data (exogenous noise preserved).
    np.testing.assert_allclose(Xt, X_orig, atol=1e-9)


def test_generate__drift_propagates_exact_delta(chain_data):
    # Known-coefficients chain: Y = 2 X + eps_Y, Z = 0.5 Y + eps_Z.
    # Shifting X by a constant c must change Y by exactly beta_XY * c on
    # every row, and Z by exactly beta_YZ * (beta_XY * c) -- the propagated
    # delta, not freshly drawn values. The oracle goes through the FITTED
    # mechanism itself (do()'s delta = response(parents_new) -
    # response(parents_orig)), so fitted-beta noise cancels exactly.
    X, graph, column_mapping = chain_data
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(3))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
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
    # Loose sanity: the fitted betas must be near the true coefficients
    # (chain_data noise std is 0.1 -> SE(beta) ~ 0.01).
    np.testing.assert_allclose(dB, 2.0 * dX, rtol=0.05, atol=1e-12)
    np.testing.assert_allclose(dC, 0.5 * dB, rtol=0.05, atol=1e-12)


def test_generate__drift_magnitude(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=2.0,
    )
    # The systematic shift is drawn once (std ~ drift_std * col_std); its
    # sampled value can be small, so assert on a generous bound (a tighter
    # bound would be seed-flaky).
    shift = abs(Xt[0, 0] - X_orig[0, 0])
    assert shift > 0.05 * (2.0 * np.std(X_orig[:, 0]))


def test_generate__drift_closed_form_exact():
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
    gen.fit(data, graph, column_mapping)
    Xt, _ = gen.generate(
        data, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
    )
    dX = Xt[:, 0] - data_orig[:, 0]
    np.testing.assert_allclose(
        Xt[:, 1] - data_orig[:, 1], 2.0 * dX, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        Xt[:, 2] - data_orig[:, 2], 1.0 * dX, rtol=1e-10, atol=1e-12
    )


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator()
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
