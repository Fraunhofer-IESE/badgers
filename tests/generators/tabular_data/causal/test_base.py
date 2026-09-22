"""Tests for the CausalGenerator base class (validation + mechanism gating)."""

import networkx as nx
import numpy as np
import pytest

from badgers.generators.tabular_data.causal._base import CausalGenerator


class _Probe(CausalGenerator):
    """Concrete subclass exposing the base-class plumbing for testing."""

    deficit_name = "probe"

    def generate(self, X, y, **params):
        raise NotImplementedError


def test_validate__returns_sets(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping, fit_dgp=False)
    order, node_to_col, descendants, exogenous, perturb_cols = gen._validate(
        X, ["X"]
    )
    assert set(order) == {"X", "Y", "Z"}
    assert descendants == {"Y", "Z"}
    assert exogenous == set()  # X has no ancestors
    assert perturb_cols == [0]


def test_validate__missing_graph_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    # not fitted -> self.graph is None
    with pytest.raises(ValueError, match="graph"):
        gen._validate(X, ["X"])


def test_validate__missing_column_mapping_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, None, fit_dgp=False)
    with pytest.raises(ValueError, match="column_mapping"):
        gen._validate(X, ["X"])


def test_validate__unknown_perturbation_node_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping, fit_dgp=False)
    with pytest.raises(ValueError, match="not found in graph"):
        gen._validate(X, ["NOPE"])


def test_should_defect__always(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    row = X[0]
    mask = gen._should_defect(
        row, "always", None, 0.5, ["X"], column_mapping, X
    )
    assert mask.tolist() == [True]


def test_should_defect__random_rate(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
    rows = X
    masks = [
        gen._should_defect(r, "random", None, 0.5, ["X", "Y"], column_mapping, X)
        for r in rows
    ]
    # ~50% of the cells fire
    n_cells = len(rows) * 2
    total = sum(m.sum() for m in masks)
    assert 0.35 * n_cells < total < 0.65 * n_cells


def test_should_defect__observed_depends_on_cause(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    # cause = X (col 0); perturb Y. High X -> defect fires.
    high_x = X[X[:, 0] > np.quantile(X[:, 0], 0.9)]
    low_x = X[X[:, 0] < np.quantile(X[:, 0], 0.1)]
    mask_high = gen._should_defect(
        high_x[0], "observed", "X", 0.2, ["Y"], column_mapping, X
    )
    mask_low = gen._should_defect(
        low_x[0], "observed", "X", 0.2, ["Y"], column_mapping, X
    )
    assert mask_high.tolist() == [True]
    assert mask_low.tolist() == [False]


def test_should_defect__self_depends_on_own_value(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    # perturb X; high X -> defect fires
    high_x = X[X[:, 0] > np.quantile(X[:, 0], 0.9)][0]
    low_x = X[X[:, 0] < np.quantile(X[:, 0], 0.1)][0]
    mask_high = gen._should_defect(
        high_x, "self", None, 0.2, ["X"], column_mapping, X
    )
    mask_low = gen._should_defect(
        low_x, "self", None, 0.2, ["X"], column_mapping, X
    )
    assert mask_high.tolist() == [True]
    assert mask_low.tolist() == [False]


def test_should_defect__unknown_mechanism_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    with pytest.raises(ValueError, match="Unknown mechanism"):
        gen._should_defect(X[0], "bogus", None, 0.5, ["X"], column_mapping, X)


def test_labels_inplace__y_none(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    affected = np.array([True, False, True])
    yt = gen._labels_inplace(None, affected, "missingness")
    assert list(yt) == ["missingness", "original", "missingness"]


def test_labels_inplace__y_passthrough(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    y = np.array([0, 1, 0])
    yt = gen._labels_inplace(y, np.array([True, True, True]), "noise")
    assert list(yt) == [0, 1, 0]


def test_labels_expanded__y_none(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    affected = np.array([True, True, False, False])
    yt = gen._labels_expanded(None, affected, "noise")
    assert list(yt) == ["noise", "noise", "original", "original"]
