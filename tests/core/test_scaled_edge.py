"""Tests for the scaled-edge view (edge reweighting without beta access)."""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import OlsDGP, ScaledEdgeDGP


def _fork_data(n=500, seed=0):
    """Y = 0.3 * X + 0.5 * W + N(0, 1.0) — multi-parent child."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("W", "Y")])
    column_mapping = {"W": 0, "X": 1, "Y": 2}  # sorted-name mapping
    X = rng.normal(size=n)
    W = rng.normal(size=n)
    Y = 0.3 * X + 0.5 * W + rng.normal(0, 1.0, size=n)
    return np.column_stack([W, X, Y]), graph, column_mapping


def test_scaled_edge_response__multi_parent_closed_form():
    data, graph, column_mapping = _fork_data()
    base = OlsDGP(graph, column_mapping).fit(data)
    scaled = ScaledEdgeDGP(base, "Y", "X", 2.0)
    parents = np.array([[3.0, -2.0]])  # structural order: predecessors(Y) = [X, W]
    # f'(P) = f(P) + (k-1) * contribution(X), with the contribution derived
    # from the base's OWN fitted response -- exact to machine precision
    # (a true-coef closed form would be unsatisfiable at atol=0.05 because
    # the fitted betas carry estimation noise at fork noise std 1.0)
    base_pred = base.response("Y", parents)
    zeroed = parents.copy()
    zeroed[:, 0] = 0.0  # zero the X column (predecessors(Y)[0] = X)
    contribution = base_pred - base.response("Y", zeroed)
    expected = base_pred + (2.0 - 1.0) * contribution
    np.testing.assert_allclose(scaled.response("Y", parents), expected)
    # sibling edge untouched: the W part of the response is unchanged
    # (scaled f' minus 2*contribution == base f minus contribution)
    np.testing.assert_allclose(
        scaled.response("Y", parents) - 2.0 * contribution,
        base_pred - contribution,
    )


def test_scaled_edge_calibrate__infeasible_raises():
    data, graph, column_mapping = _fork_data()
    base = OlsDGP(graph, column_mapping).fit(data)
    scaled = ScaledEdgeDGP(base, "Y", "X", 100.0)
    with pytest.raises(ValueError, match="infeasible"):
        scaled.calibrate(data)


def test_scaled_edge_marginal_variance_preserved():
    data, graph, column_mapping = _fork_data()
    base = OlsDGP(graph, column_mapping).fit(data)
    scaled = ScaledEdgeDGP(base, "Y", "X", 2.0)
    scaled.calibrate(data)
    # fresh parent draws -> child resampled through the scaled mechanism
    rng = np.random.default_rng(3)
    fresh = np.column_stack([rng.normal(size=10000), rng.normal(size=10000),
                             np.zeros(10000)])
    parents = fresh[:, [column_mapping["X"], column_mapping["W"]]]
    child = (scaled.response("Y", parents)
             + scaled.draw_noise("Y", rng, 10000))
    var_orig = np.var(data[:, column_mapping["Y"]])
    assert abs(np.var(child) - var_orig) < 0.1 * var_orig


def test_scaled_edge_delegates_graph_mapping_and_sample():
    data, graph, column_mapping = _fork_data()
    base = OlsDGP(graph, column_mapping).fit(data)
    scaled = ScaledEdgeDGP(base, "Y", "X", 2.0).fit(data)
    assert scaled.graph is base.graph
    assert scaled.column_mapping is base.column_mapping
    np.testing.assert_allclose(
        scaled.sample(np.random.default_rng(0), 20),
        base.sample(np.random.default_rng(0), 20),
    )


def test_scaled_edge_unknown_edge_raises():
    data, graph, column_mapping = _fork_data()
    base = OlsDGP(graph, column_mapping).fit(data)
    with pytest.raises(ValueError, match="not in graph"):
        ScaledEdgeDGP(base, "X", "Y", 2.0)  # X is a root: Y is not its parent
