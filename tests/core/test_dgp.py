"""Backend contract tests for the default OLS DGP.

The OLS backend is deterministic, so response recovery is exact (no MCMC
tolerance). These tests define the contract every backend must satisfy.
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import OlsDGP


def _make_chain_data(n=100, seed=0):
    """X -> Y -> Z with known coefficients (beta_XY=2.0, beta_YZ=0.5)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    Y = 2.0 * X + rng.normal(0, 0.1, size=n)
    Z = 0.5 * Y + rng.normal(0, 0.1, size=n)
    data = np.column_stack([X, Y, Z])
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    return data, graph, column_mapping


def test_fit__response_recovers_true_mechanism():
    data, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping).fit(data)
    # deterministic least squares: response(parents) recovers the TRUE
    # mechanism (beta = 2.0 / 0.5) up to estimation error; the raw child
    # column only matches up to its residual (noise std = 0.1)
    np.testing.assert_allclose(
        dgp.response("Y", data[:, [0]]), 2.0 * data[:, 0], atol=0.05
    )
    np.testing.assert_allclose(
        dgp.response("Z", data[:, [1]]), 0.5 * data[:, 1], atol=0.05
    )


def test_response__root_returns_fitted_mean():
    data, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping).fit(data)
    mean = dgp.response("X", np.zeros((7, 0)))
    assert mean.shape == (7,)
    np.testing.assert_allclose(mean, np.mean(data[:, 0]), atol=0.05)


def test_response__before_fit_raises():
    _, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping)
    with pytest.raises(RuntimeError, match="before fit"):
        dgp.response("X", np.zeros((3, 0)))


def test_draw_noise__before_fit_raises():
    _, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping)
    with pytest.raises(RuntimeError, match="before fit"):
        dgp.draw_noise("Z", np.random.default_rng(0), 3)


def test_draw_noise__shape_and_spread():
    data, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping).fit(data)
    shocks = dgp.draw_noise("Z", np.random.default_rng(0), 5000)
    assert shocks.shape == (5000,)
    # fitted residual std ~ 0.1
    assert 0.05 < np.std(shocks) < 0.2


def test_fit_node__isolation():
    data, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping).fit(data)
    z_before = dgp.response("Z", data[:, [1]]).copy()
    # refit ONLY node Y on different data: Z's fitted mechanism is untouched
    other = np.column_stack(
        [np.random.default_rng(9).normal(size=len(data)),
         np.full(len(data), 7.0), np.zeros(len(data))]
    )
    dgp.fit_node("Y", ["X"], other)
    np.testing.assert_allclose(dgp.response("Z", data[:, [1]]), z_before)


def test_fit__loop_equals_per_node_fits():
    data, graph, column_mapping = _make_chain_data()
    a = OlsDGP(graph, column_mapping).fit(data)
    b = OlsDGP(graph, column_mapping)
    for node in nx.topological_sort(graph):
        b.fit_node(node, list(graph.predecessors(node)), data)
    np.testing.assert_allclose(
        a.response("Y", data[:, [0]]), b.response("Y", data[:, [0]])
    )
    np.testing.assert_allclose(
        a.response("Z", data[:, [1]]), b.response("Z", data[:, [1]])
    )


def test_sample__shape_and_root_mean():
    data, graph, column_mapping = _make_chain_data()
    dgp = OlsDGP(graph, column_mapping).fit(data)
    X_sampled = dgp.sample(np.random.default_rng(0), 50)
    assert X_sampled.shape == (50, 3)
    X_big = dgp.sample(np.random.default_rng(0), 2000)
    assert abs(np.mean(X_big[:, 0]) - np.mean(data[:, 0])) < 0.5


# -- G9: complete-case fit on NaN-contaminated data --------------------------

def test_fit__nan_rows_complete_case():
    # G9: a NaN-contaminated matrix (e.g. the output of a missingness
    # stage) must NOT crash the fit with a cryptic LinAlgError: NaN rows
    # are dropped for the ESTIMATION (complete case) only.
    data, graph, column_mapping = _make_chain_data()
    dirty = data.copy()
    dirty[:50, 0] = np.nan  # contaminates X, hence Y and Z rows too
    dgp = OlsDGP(graph, column_mapping).fit(dirty)
    clean = OlsDGP(graph, column_mapping).fit(data)
    # beta estimated from the 50 complete-case rows ~= beta from all rows
    np.testing.assert_allclose(
        dgp._beta["Y"], clean._beta["Y"], atol=0.05
    )
    np.testing.assert_allclose(
        dgp._beta["Z"], clean._beta["Z"], atol=0.1
    )


def test_fit__fully_nan_column_raises():
    # A fully NaN column cannot identify its equation -> a CLEAR ValueError
    # (not the cryptic SVD LinAlgError).
    data, graph, column_mapping = _make_chain_data()
    dirty = data.copy()
    dirty[:, 1] = np.nan
    dgp = OlsDGP(graph, column_mapping)
    with pytest.raises(ValueError, match="complete-case|too many NaNs"):
        dgp.fit(dirty)
