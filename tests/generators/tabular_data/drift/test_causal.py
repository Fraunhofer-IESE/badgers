"""Tests for CausalDriftGenerator."""

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


def test_generate__drift_no_per_row_residual(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=0.0,
    )
    # drift_std=0 -> no shift. Y = beta * X (deterministic forward pass, no
    # residual). So Y's std should be ~ beta * X's std (no extra noise).
    x_std = np.std(Xt[:, 0])
    y_std = np.std(Xt[:, 1])
    # beta_XY ~ 2.0, so y_std ~ 2 * x_std (within tolerance)
    assert 1.0 * x_std < y_std < 3.5 * x_std


def test_generate__drift_magnitude(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalDriftGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", drift_std=2.0,
    )
    # The systematic shift is drawn once (std ~ drift_std * col_std) and its
    # sample mean has SE = shift_std/sqrt(n); with n=100 a loose 0.1* bound
    # is reliable, while a tighter bound would be seed-flaky.
    shift = abs(np.mean(Xt[:, 0]) - np.mean(X_orig[:, 0]))
    assert shift > 0.1 * (2.0 * np.std(X_orig[:, 0]))


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalDriftGenerator()
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
