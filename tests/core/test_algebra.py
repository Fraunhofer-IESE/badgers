"""Tests for the value-level intervention (Pearl surgery on given rows).

Uses a hand-built deterministic DGP so the expected values are exact.
``intervene`` semantics: fix the nodes at ``values``, recompute ALL their
descendants with fresh noise; upstream and non-descendant columns and rows
untouched; intervened nodes stay pinned even if one is a descendant of
another.
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import DGPMixin, ordered_descendants


class _FakeDGP(DGPMixin):
    """Deterministic linear DGP: v = sum coef[p] * p + N(0, noise_std)."""

    def __init__(self, graph, column_mapping, beta, mu=None, noise_std=0.0):
        self.graph = graph
        self.column_mapping = column_mapping
        self._beta = beta          # node -> {parent: coef}
        self._mu = mu or {}
        self._noise_std = noise_std

    def response(self, node, parent_values):
        parents = list(self.graph.predecessors(node))
        if not parents:
            return np.full(parent_values.shape[0], self._mu[node])
        coefs = np.array([self._beta[node][p] for p in parents])
        return parent_values @ coefs

    def draw_noise(self, node, rng, n):
        return rng.normal(0, self._noise_std, size=n)


def _chain_dgp(noise_std=0.0):
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    beta = {"Y": {"X": 2.0}, "Z": {"Y": 0.5}}
    rng = np.random.default_rng(0)
    X = rng.normal(size=100)
    rows = np.column_stack([X, 2.0 * X, 0.5 * (2.0 * X)])
    return _FakeDGP(graph, column_mapping, beta, noise_std=noise_std), rows


def test_ordered_descendants_topological_order():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    assert ordered_descendants(graph, ["A"]) == ["B", "C", "D"]
    # a leaf has no descendants
    assert ordered_descendants(graph, ["D"]) == []


def test_intervene_accepts_bare_node():
    dgp, _ = _chain_dgp()
    assert ordered_descendants(dgp.graph, "X") == ["Y", "Z"]


def test_intervene_requires_rng():
    dgp, rows = _chain_dgp()
    with pytest.raises(ValueError, match="rng"):
        dgp.intervene(rows.copy(), ["X"])


def test_intervene_noise_free_is_deterministic():
    # With a noise-free mechanism the recomputation is exact: the whole
    # downstream chain is re-derived from the intervened value.
    dgp, rows = _chain_dgp()
    before = rows.copy()
    out = dgp.intervene(rows, "X", values=[rows[:, 0] + 3.0],
                        rng=np.random.default_rng(0))
    assert out is rows  # in-place contract: the same array is returned
    np.testing.assert_allclose(out[:, 0], before[:, 0] + 3.0)   # pinned
    np.testing.assert_allclose(out[:, 1], 2.0 * (before[:, 0] + 3.0))
    np.testing.assert_allclose(out[:, 2], 0.5 * (2.0 * (before[:, 0] + 3.0)))


def test_intervene_values_none_keeps_pinned_columns():
    # values=None: the current column values ARE the intervention (callers
    # that pre-write their perturbation); the pinned nodes are not
    # recomputed, the descendants are.
    dgp, rows = _chain_dgp()
    before = rows.copy()
    rows[:, 0] += 3.0
    out = dgp.intervene(rows, ["X"], rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 0], before[:, 0] + 3.0)
    np.testing.assert_allclose(out[:, 1], 2.0 * (before[:, 0] + 3.0))
    np.testing.assert_allclose(out[:, 2], 0.5 * (2.0 * (before[:, 0] + 3.0)))


def test_intervene_pinned_descendant_not_recomputed():
    # X -> Y -> Z, intervene on BOTH X and Y: Y is pinned (its own value is
    # the intervention) even though it is a descendant of X; only Z is
    # recomputed, FROM Y's (intervened) value.
    dgp, rows = _chain_dgp()
    y_new = rows[:, 1] + 10.0
    out = dgp.intervene(rows, ["X", "Y"], rng=np.random.default_rng(0),
                        values=[rows[:, 0] + 3.0, y_new])
    np.testing.assert_allclose(out[:, 1], y_new)  # pinned, not recomputed
    np.testing.assert_allclose(out[:, 2], 0.5 * y_new)  # from pinned Y


def test_intervene_recompute_redraws_descendant_noise():
    dgp, rows = _chain_dgp(noise_std=0.3)
    before = rows.copy()
    out = dgp.intervene(rows, ["X"], values=[rows[:, 0] + 3.0],
                        rng=np.random.default_rng(1))
    # descendants are recomputed from the perturbed parents with FRESH noise
    deterministic_y = 2.0 * (before[:, 0] + 3.0)
    residual = out[:, 1] - deterministic_y
    assert np.std(residual) > 0.15  # fitted noise std is 0.3
    np.testing.assert_allclose(np.mean(residual), 0.0, atol=0.1)


def test_intervene_nondescendant_parent_contributes():
    # X -> Y, W -> Y: intervening on X keeps W's contribution (only X's
    # influence is re-derived).
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("W", "Y")])
    column_mapping = {"W": 0, "X": 1, "Y": 2}
    beta = {"Y": {"X": 2.0, "W": 0.5}}
    dgp = _FakeDGP(graph, column_mapping, beta)
    rows = np.array([[5.0, 1.0, 2.0 * 1.0 + 0.5 * 5.0]])
    out = dgp.intervene(rows, ["X"], values=[np.array([4.0])],
                        rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 0], [5.0])   # W untouched
    np.testing.assert_allclose(out[:, 2], [2.0 * 4.0 + 0.5 * 5.0])


def test_intervene_unknown_node_raises():
    dgp, rows = _chain_dgp()
    with pytest.raises(KeyError):
        dgp.intervene(rows, ["NOPE"], values=[rows[:, 0]],
                      rng=np.random.default_rng(0))
