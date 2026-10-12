"""Tests for the generic deficit algebra (do-operations), backend-blind.

Uses a hand-built deterministic DGP so the expected values are exact.
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import do, ordered_descendants


class _FakeDGP:
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


def test_do_single_node_without_list():
    dgp, _ = _chain_dgp()
    assert ordered_descendants(dgp.graph, "X") == ["Y", "Z"]


def test_do_delta_requires_rows_original():
    dgp, rows = _chain_dgp()
    with pytest.raises(ValueError, match="rows_original"):
        do(dgp, rows.copy(), ["X"])


def test_do_delta_linear_chain_composes():
    dgp, rows = _chain_dgp()
    rows_original = rows.copy()
    out = do(dgp, rows, ["X"], values=[rows[:, 0] + 3.0],
             rows_original=rows_original)
    # do() mutates rows in place: the oracles must compare against the
    # pre-perturbation snapshot (rows_original), not the mutated rows.
    assert out is rows  # in-place contract: the same array is returned
    np.testing.assert_allclose(out[:, 1], rows_original[:, 1] + 2.0 * 3.0)
    np.testing.assert_allclose(out[:, 2], rows_original[:, 2] + 0.5 * (2.0 * 3.0))
    np.testing.assert_allclose(out[:, 0], rows_original[:, 0] + 3.0)


def test_do_delta_zero_delta_is_identity():
    dgp, rows = _chain_dgp()
    rows_original = rows.copy()
    out = do(dgp, rows, ["X"], values=[rows[:, 0]],
             rows_original=rows_original)
    np.testing.assert_array_equal(out, rows_original)


def test_do_recompute_redraws_descendant_noise():
    dgp, rows = _chain_dgp(noise_std=0.3)
    rows_original = rows.copy()  # do() mutates rows in place: capture BEFORE
    out = do(dgp, rows, ["X"], values=[rows[:, 0] + 3.0], mode="recompute",
             rng=np.random.default_rng(1))
    # descendants are recomputed from the perturbed parents with FRESH noise
    deterministic_y = 2.0 * (rows_original[:, 0] + 3.0)
    residual = out[:, 1] - deterministic_y
    assert np.std(residual) > 0.15  # fitted noise std is 0.3
    np.testing.assert_allclose(np.mean(residual), 0.0, atol=0.1)


def test_do_recompute_requires_rng():
    dgp, rows = _chain_dgp(noise_std=0.3)
    with pytest.raises(ValueError, match="rng"):
        do(dgp, rows, ["X"], mode="recompute")


def test_do_unknown_mode_raises():
    dgp, rows = _chain_dgp()
    rows_original = rows.copy()
    with pytest.raises(ValueError, match="Unknown propagation mode"):
        do(dgp, rows, ["X"], values=[rows[:, 0]], mode="bogus",
           rows_original=rows_original)


def test_do_requires_fitted_dgp_nodes():
    # a node whose parent columns are not filled is the caller's contract;
    # here we only verify that an unknown node raises a KeyError-style error
    dgp, rows = _chain_dgp()
    rows_original = rows.copy()
    with pytest.raises(KeyError):
        do(dgp, rows, ["NOPE"], values=[rows[:, 0]],
           rows_original=rows_original)
