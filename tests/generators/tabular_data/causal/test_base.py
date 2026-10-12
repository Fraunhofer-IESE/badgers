"""Tests for the CausalGenerator base class (validation + mechanism gating + delta propagation)."""

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
    gen.fit(X, graph, column_mapping)
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
    gen.fit(X, graph, None)
    with pytest.raises(ValueError, match="column_mapping"):
        gen._validate(X, ["X"])


def test_validate__unknown_perturbation_node_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping)
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


class _FakeDGP:
    """Minimal stand-in for a DGP backend: deterministic linear mechanisms."""

    def __init__(self, graph, column_mapping, beta, noise_std=0.0):
        self.graph = graph
        self.column_mapping = column_mapping
        self._beta = beta          # node -> {parent: coef}
        self._noise_std = noise_std

    def response(self, node, parent_values):
        parents = list(self.graph.predecessors(node))
        coefs = np.array([self._beta[node][p] for p in parents])
        return parent_values @ coefs

    def draw_noise(self, node, rng, n):
        return rng.normal(0, self._noise_std, size=n)


def test_propagate__delta_requires_rows_original(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping)
    gen.dgp = _FakeDGP(graph, column_mapping, {})
    with pytest.raises(ValueError, match="rows_original"):
        gen.propagate(X.copy(), ["X"], np.random.default_rng(0))


def test_propagate__delta_linear_chain(chain_data):
    # Delta semantics: descendant_new = descendant_orig + beta @ delta_parents,
    # for ALL descendants in topological order (multi-hop chains compose).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping)
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    rng = np.random.default_rng(0)
    rows = X.copy()
    rows[:, 0] += 3.0  # perturb the root in place: +3 on X
    rows_original = X.copy()
    out = gen.propagate(rows, ["X"], rng, rows_original=rows_original)
    np.testing.assert_allclose(out[:, 1], X[:, 1] + 2.0 * 3.0)  # Y += beta_XY * dX
    np.testing.assert_allclose(out[:, 2], X[:, 2] + 0.5 * (2.0 * 3.0))  # Z += beta_YZ * dY
    # Root column untouched by propagation
    np.testing.assert_allclose(out[:, 0], X[:, 0] + 3.0)


def test_propagate__zero_parent_delta_is_identity(chain_data):
    # A descendant whose parents did not change must keep its original value
    # exactly (exogenous shock preserved, no recompute noise).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping)
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    rng = np.random.default_rng(0)
    rows = X.copy()
    rows_original = X.copy()
    out = gen.propagate(rows, ["X"], rng, rows_original=rows_original)
    np.testing.assert_array_equal(out, X)


def test_propagate__recompute_semantics(chain_data):
    # mode="recompute" -> descendants = f(child, parents_new) + fresh noise.
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.fit(X, graph, column_mapping)
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}}, noise_std=0.3)
    rows = X.copy()
    rows[:, 0] += 3.0  # perturb the root in place
    out = gen.propagate(rows, ["X"], np.random.default_rng(0),
                        mode="recompute")
    # descendants are recomputed from the perturbed parents WITH noise:
    # not the noise-free delta result.
    deterministic_y = 2.0 * (X[:, 0] + 3.0)
    assert np.std(out[:, 1] - deterministic_y) > 0.15
