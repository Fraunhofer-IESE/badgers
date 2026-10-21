"""Tests for CausalImbalanceGenerator."""

import numpy as np
import pytest

from badgers.core.selectors import TopQuantileSelector
from badgers.generators.tabular_data.imbalance.causal import (
    CausalImbalanceGenerator,
)


def _fitted_generator(X, graph, column_mapping, make_gcm_dgp):
    gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    return gen


def test_generate__quantile_range_respected(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)

    Xt = gen.generate(
        X, perturbation_nodes=["X"], quantile_range=(0.5, 1.0)
    )

    assert Xt.shape[0] < X.shape[0]
    assert Xt.shape[1] == X.shape[1]
    lo = np.quantile(X[:, 0], 0.5)
    assert np.all(Xt[:, 0] >= lo)
    labels = gen.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == Xt.shape[0]
    assert labels.all()


def test_generate__descendants_skewed(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)

    Xt = gen.generate(
        X, perturbation_nodes=["X"], quantile_range=(0.5, 1.0)
    )

    assert np.mean(Xt[:, 1]) > np.mean(X[:, 1])


def test_generate__descendants_are_pure_subset(chain_data, make_gcm_dgp):
    """Kept X rows, including descendants, are bit-identical to the input."""
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)

    Xt = gen.generate(
        X, perturbation_nodes=["X"], quantile_range=(0.5, 1.0)
    )

    lo = np.quantile(X[:, 0], 0.5)
    keep = X[:, 0] >= lo
    np.testing.assert_array_equal(Xt, X[keep])


def test_generate__multi_node_intersection(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)

    Xt = gen.generate(
        X, perturbation_nodes=["X", "Y"], quantile_range=(0.5, 1.0)
    )

    assert np.mean(Xt[:, 0]) > np.mean(X[:, 0])
    assert np.mean(Xt[:, 1]) > np.mean(X[:, 1])


def test_generate__validation_errors(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalImbalanceGenerator()
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(X, perturbation_nodes=["X"])
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    with pytest.raises(ValueError, match="quantile_range"):
        gen.generate(
            X, perturbation_nodes=["X"], quantile_range=(0.9, 0.1)
        )


def test_generate__explicit_selector_drops_fired_rows(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)
    selector = TopQuantileSelector(
        variables="X", column_mapping=column_mapping, quantile=0.8
    )

    Xt = gen.generate(X, selector=selector)

    fired = selector(X)["X"]
    np.testing.assert_array_equal(Xt, X[~fired])
    assert Xt.shape[0] < X.shape[0]
    labels = gen.get_imbalance_labels()
    assert labels.dtype == bool
    assert len(labels) == Xt.shape[0]
    assert labels.all()


def test_generate__explicit_selector_ignores_quantile_params(
    chain_data, make_gcm_dgp
):
    X, graph, column_mapping = chain_data
    gen = _fitted_generator(X, graph, column_mapping, make_gcm_dgp)
    selector = TopQuantileSelector(
        variables="Y", column_mapping=column_mapping, quantile=0.7
    )

    Xt = gen.generate(
        X, selector=selector, quantile_range=(0.9, 0.1)
    )

    fired = selector(X)["Y"]
    np.testing.assert_array_equal(Xt, X[~fired])
