"""Tests for CausalMissingnessGenerator (observation layer, chooser-only).

Semantics: NaN is written ONLY at the fired cells of the perturbation
nodes. No descendant cascade -- missingness is a property of the
measurement, not of the world.
"""

import numpy as np
import pytest

from badgers.generators.tabular_data.missingness.causal import (
    CausalMissingnessGenerator,
)


def test_generate__always_fires_everywhere(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always",
    )
    # in-place: same shape
    assert Xt.shape == X.shape
    # X always missing
    assert np.all(np.isnan(Xt[:, 0]))
    # NO cascade: descendants keep their observed values
    assert np.all(np.isfinite(Xt[:, 1]))
    assert np.all(np.isfinite(Xt[:, 2]))
    # labels
    assert list(yt) == ["missingness"] * X.shape[0]


def test_generate__mcar_mask_independent_of_values(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="random", mechanism_rate=0.5,
    )
    mask = np.isnan(Xt[:, 0])
    # ~50% missing
    assert 0.4 < mask.mean() < 0.6
    # Missingness independent of X's (latent) value: the observed (non-missing)
    # X values should have a mean near the original mean.
    observed_x = Xt[:, 0][~mask]
    assert abs(np.mean(observed_x) - np.mean(X_orig[:, 0])) < 0.5
    # Only the perturbed column is touched.
    np.testing.assert_array_equal(np.isnan(Xt[:, 1]), np.zeros(X.shape[0], bool))
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), np.zeros(X.shape[0], bool))


def test_generate__self_mnar_mask_depends_on_own_value(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="self", mechanism_rate=0.3,
    )
    mask = np.isnan(Xt[:, 0])
    # MNAR: high X values are more likely missing.
    # Check that the observed (non-missing) X mean is shifted DOWN (high values
    # removed).
    observed_x = Xt[:, 0][~mask]
    assert np.mean(observed_x) < np.mean(X_orig[:, 0]) - 0.2
    # Only the perturbed column is touched.
    assert np.all(np.isfinite(Xt[:, 1]))
    assert np.all(np.isfinite(Xt[:, 2]))


def test_generate__observed_mar_depends_on_cause(chain_data):
    X, graph, column_mapping = chain_data
    # cause = X (col 0, an ancestor of Y, fully observed); perturb Y.
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["Y"],
        mechanism="observed", mechanism_cause="X", mechanism_rate=0.5,
    )
    mask = np.isnan(Xt[:, 1])
    # Missingness should correlate with X (the cause).
    x = Xt[:, 0]
    miss_high_x = mask[x >= np.quantile(x, 0.75)].mean()
    miss_low_x = mask[x <= np.quantile(x, 0.25)].mean()
    assert miss_high_x > miss_low_x
    # Only the perturbed column is touched.
    assert np.all(np.isfinite(Xt[:, 0]))
    assert np.all(np.isfinite(Xt[:, 2]))


def test_generate__multi_node_only_fired_cells(chain_data):
    # Perturb TWO nodes: each column gets NaN only where IT fired.
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(3))
    gen.fit(X, graph, column_mapping)
    Xt, _ = gen.generate(
        X, y=None, perturbation_nodes=["X", "Z"],
        mechanism="random", mechanism_rate=0.3,
    )
    mx = np.isnan(Xt[:, 0])
    mz = np.isnan(Xt[:, 2])
    assert 0.1 < mx.mean() < 0.5
    assert 0.1 < mz.mean() < 0.5
    # Y untouched.
    assert np.all(np.isfinite(Xt[:, 1]))


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    # not fitted yet
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
    # unknown perturbation node
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="not found in graph"):
        gen.generate(X, y=None, perturbation_nodes=["NOPE"])


def test_generate__mar_requires_cause(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="mechanism_cause"):
        gen.generate(
            X, y=None, perturbation_nodes=["X"],
            mechanism="observed", mechanism_cause=None,
        )


def test_generate__unknown_mechanism_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="mechanism"):
        gen.generate(X, y=None, perturbation_nodes=["X"], mechanism="nope")


def test_generate__labeled_y_passes_through(chain_data):
    X, graph, column_mapping = chain_data
    y = np.array(["a", "b"] * (X.shape[0] // 2))
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=y, perturbation_nodes=["X"], mechanism="always",
    )
    # y unchanged (defect does not change targets)
    np.testing.assert_array_equal(yt, y)


def test_generate__reproducible_with_explicit_rng(chain_data):
    X, graph, column_mapping = chain_data
    outs = []
    for _ in range(2):
        gen = CausalMissingnessGenerator(
            random_generator=np.random.default_rng(11)
        )
        gen.fit(X, graph, column_mapping)
        Xt, _ = gen.generate(
            X.copy(), y=None, perturbation_nodes=["X"],
            mechanism="random", mechanism_rate=0.4,
        )
        outs.append(Xt)
    np.testing.assert_array_equal(outs[0], outs[1])


def test_generate__all_nan_input_edge(chain_data):
    # A fully NaN input column: firing writes NaN again (idempotent), no
    # crash, no other column touched.
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(2))
    gen.fit(X, graph, column_mapping)
    Xn = X.copy()
    Xn[:, 1] = np.nan
    Xt, _ = gen.generate(Xn, y=None, perturbation_nodes=["Y"],
                         mechanism="always")
    assert np.all(np.isnan(Xt[:, 1]))
    np.testing.assert_array_equal(Xt[:, 0], Xn[:, 0])
    np.testing.assert_array_equal(Xt[:, 2], Xn[:, 2])
