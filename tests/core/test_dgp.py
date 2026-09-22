"""Unit tests for the PyMC-based DGP (PyMCDGP)."""

import networkx as nx
import numpy as np

from badgers.core.dgp import PyMCDGP


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


def test_fit__chain_beta_close_to_true():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    assert dgp._fitted
    # beta for Y (parent X) close to 2.0
    assert abs(dgp._beta_mean["Y"][0] - 2.0) < 0.5
    # beta for Z (parent Y) close to 0.5
    assert abs(dgp._beta_mean["Z"][0] - 0.5) < 0.5


def test_forward_pass__chain_descendants_filled():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    rows = np.full((5, 3), np.nan)
    rows[:, 0] = 10.0  # perturb X
    dgp.forward_pass(rows, ["Y", "Z"], np.random.default_rng(0),
                     add_residual=False)
    # Y and Z (descendants) are filled
    assert np.all(np.isfinite(rows[:, 1]))
    assert np.all(np.isfinite(rows[:, 2]))
    # Y shifted well above its original mean (X=10 -> Y ~ 2*10)
    assert np.mean(rows[:, 1]) > np.mean(data[:, 1]) + 5.0


def test_forward_pass__empty_nodes_noop():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    rows = np.full((5, 3), np.nan)
    rows[:, 2] = 10.0  # perturb Z (leaf)
    dgp.forward_pass(rows, [], np.random.default_rng(0), add_residual=False)
    # Ancestors X, Y remain NaN (no backward propagation)
    assert np.all(np.isnan(rows[:, 0]))
    assert np.all(np.isnan(rows[:, 1]))


def test_forward_pass__add_residual_varies_rows():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    rows = np.full((50, 3), np.nan)
    rows[:, 0] = 10.0
    dgp.forward_pass(rows, ["Y", "Z"], np.random.default_rng(0),
                     add_residual=True)
    # With residual, Y values are not all identical
    assert np.std(rows[:, 1]) > 0.0


def test_sample__shape():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    X_sampled = dgp.sample(np.random.default_rng(0), 50)
    assert X_sampled.shape == (50, 3)


def test_sample__marginal_mean_close():
    data, graph, column_mapping = _make_chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_samples_mcmc=50)
    dgp.fit(data)
    X_sampled = dgp.sample(np.random.default_rng(0), 2000)
    # Root X mean close to original
    assert abs(np.mean(X_sampled[:, 0]) - np.mean(data[:, 0])) < 0.5
