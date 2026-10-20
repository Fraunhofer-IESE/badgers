"""Tests for CausalImbalanceGenerator."""

import numpy as np
import pytest

from badgers.core.selectors import BernoulliSelector, TopQuantileSelector
from badgers.generators.tabular_data.imbalance.causal import (
    CausalImbalanceGenerator,
)


def test_generate__quantile_range_respected(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        quantile_range=(0.5, 1.0),
    )
    # in-place subset: fewer rows, all within the top 50% quantile of X
    assert Xt.shape[0] < X.shape[0]
    assert Xt.shape[1] == X.shape[1]
    lo = np.quantile(X[:, 0], 0.5)
    assert np.all(Xt[:, 0] >= lo)
    labels = gen.get_imbalance_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__descendants_skewed(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        quantile_range=(0.5, 1.0),
    )
    # Y (descendant of X) marginal is skewed (mean shifted up)
    assert np.mean(Xt[:, 1]) > np.mean(X[:, 1])


def test_generate__descendants_are_pure_subset(chain_data, make_gcm_dgp):
    # Imbalance only filters rows; since the kept rows' parents are
    # unchanged, the propagated delta is 0 and every value (including the
    # descendants') must be bit-identical to the filtered original.
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        quantile_range=(0.5, 1.0),
    )
    lo = np.quantile(X[:, 0], 0.5)
    keep = X[:, 0] >= lo
    np.testing.assert_array_equal(Xt, X[keep])


def test_generate__multi_node_intersection(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X", "Y"],
        quantile_range=(0.5, 1.0),
    )
    # Both X and Y kept only in the top 50% -> both means shifted up
    assert np.mean(Xt[:, 0]) > np.mean(X[:, 0])
    assert np.mean(Xt[:, 1]) > np.mean(X[:, 1])


def test_generate__validation_errors(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator()
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    with pytest.raises(ValueError, match="quantile_range"):
        gen.generate(
            X, y=None, perturbation_nodes=["X"], quantile_range=(0.9, 0.1),
        )


def test_generate__explicit_selector_drops_fired_rows(chain_data, make_gcm_dgp):
    # With an explicit selector, the FIRED rows are dropped (subsampling
    # polarity) and perturbation_nodes/quantile_range are ignored.
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))

    sel = TopQuantileSelector(
        variables="X", column_mapping=column_mapping, quantile=0.8
    )
    Xt, yt = gen.generate(X, y=None, selector=sel)

    fired = sel(X, None)["X"]
    np.testing.assert_array_equal(Xt, X[~fired])
    assert Xt.shape[0] < X.shape[0]
    labels = gen.get_imbalance_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__explicit_selector_ignores_quantile_params(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))

    sel = TopQuantileSelector(
        variables="Y", column_mapping=column_mapping, quantile=0.7
    )
    Xt, _ = gen.generate(
        X, y=None, selector=sel, quantile_range=(0.9, 0.1),  # invalid, ignored
    )
    fired = sel(X, None)["Y"]
    np.testing.assert_array_equal(Xt, X[~fired])
