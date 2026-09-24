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
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
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
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
    )
    # Drift is systematic: the shift is drawn ONCE and applied identically to
    # every affected row. With mechanism="always" all rows are affected, so
    # the per-row change at the perturbation node must be a single constant.
    shift = Xt[:, 0] - X[:, 0]
    assert np.allclose(shift, shift[0], atol=1e-8)


def test_generate__drift_zero_shift_leaves_data_untouched(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=0.0,
    )
    # drift_std=0 -> shift of exactly 0 -> delta propagation adds exactly 0:
    # the output must equal the original data (exogenous noise preserved).
    np.testing.assert_allclose(Xt, X, atol=1e-9)


def test_generate__drift_propagates_exact_delta(chain_data):
    # Known-coefficients chain: Y = 2 X + eps_Y, Z = 0.5 Y + eps_Z.
    # Shifting X by a constant c must change Y by exactly beta_XY * c on
    # every row, and Z by exactly beta_YZ * (beta_XY * c) -- the propagated
    # delta, not freshly drawn values.
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(3))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=200)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=1.0,
    )
    dX = Xt[:, 0] - X[:, 0]
    dB = Xt[:, 1] - X[:, 1]
    dC = Xt[:, 2] - X[:, 2]
    # delta Y = beta_XY * delta X exactly (constant per row)
    beta_xy = gen.dgp._beta_mean["Y"][0]
    np.testing.assert_allclose(dB, beta_xy * dX, atol=1e-8)
    # delta Z = beta_YZ * delta Y exactly (constant per row)
    beta_yz = gen.dgp._beta_mean["Z"][0]
    np.testing.assert_allclose(dC, beta_yz * dB, atol=1e-8)


def test_generate__drift_magnitude(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=2.0,
    )
    # The systematic shift is drawn once (std ~ drift_std * col_std); its
    # sampled value can be small, so assert on a generous bound (a tighter
    # bound would be seed-flaky).
    shift = abs(Xt[0, 0] - X_orig[0, 0])
    assert shift > 0.05 * (2.0 * np.std(X_orig[:, 0]))


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator()
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
