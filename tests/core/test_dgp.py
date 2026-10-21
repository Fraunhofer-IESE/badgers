"""Backend contract tests for the SCM-adapter DGP (``GcmSCMDGP``, dowhy extra).

The adapter delegates mechanism fitting to ``gcm.fit``; these tests pin the
BEHAVIORAL contract the SCM-adapter backends must satisfy. They degrade to
a skip without the ``dowhy`` extra.
"""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("dowhy.gcm")


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


def test_fit__response_recovers_true_mechanism(make_gcm_dgp):
    data, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping).fit(data)
    # deterministic least squares: response(parents) recovers the TRUE
    # mechanism (beta = 2.0 / 0.5) up to estimation error; the raw child
    # column only matches up to its residual (noise std = 0.1)
    np.testing.assert_allclose(
        dgp.response("Y", data[:, [0]]), 2.0 * data[:, 0], atol=0.05
    )
    np.testing.assert_allclose(
        dgp.response("Z", data[:, [1]]), 0.5 * data[:, 1], atol=0.05
    )


def test_response__root_zero_and_noise_reconstructs_marginal(make_gcm_dgp):
    data, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping).fit(data)
    # empirical root: the deterministic part contributes zeros; the MARGINAL
    # comes entirely from draw_noise resampling the stored observations
    mean = dgp.response("X", np.zeros((7, 0)))
    assert mean.shape == (7,)
    np.testing.assert_allclose(mean, np.zeros(7))
    draws = dgp.draw_noise("X", np.random.default_rng(0), 5000)
    assert np.mean(draws) == pytest.approx(np.mean(data[:, 0]), abs=0.05)
    assert np.std(draws) == pytest.approx(np.std(data[:, 0]), rel=0.1)


def test_response__before_fit_raises(make_gcm_dgp):
    _, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping)
    # the adapter has no mechanism table before fit -> KeyError
    with pytest.raises(KeyError):
        dgp.response("X", np.zeros((3, 0)))


def test_draw_noise__before_fit_raises(make_gcm_dgp):
    _, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping)
    with pytest.raises(KeyError):
        dgp.draw_noise("Z", np.random.default_rng(0), 3)


def test_draw_noise__shape_and_spread(make_gcm_dgp):
    data, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping).fit(data)
    shocks = dgp.draw_noise("Z", np.random.default_rng(0), 5000)
    assert shocks.shape == (5000,)
    # fitted residual std ~ 0.1
    assert 0.05 < np.std(shocks) < 0.2


# NOTE: the adapter delegates the WHOLE-GRAPH fit to gcm.fit — there is no
# per-node refit semantics to isolate (``fit_node`` only validates/caches
# the mechanism translation), so the former per-node isolation/loop tests
# do not apply. Likewise the complete-case (G9) fit tests were OlsDGP
# internals: NaN handling here is delegated to gcm's per-mechanism
# estimators (no complete-case guarantee); see tests/core/test_gcm_scm.py.


def test_sample__shape_and_root_mean(make_gcm_dgp):
    data, graph, column_mapping = _make_chain_data()
    dgp = make_gcm_dgp(graph, column_mapping).fit(data)
    X_sampled = dgp.sample(np.random.default_rng(0), 50)
    assert X_sampled.shape == (50, 3)
    X_big = dgp.sample(np.random.default_rng(0), 2000)
    assert abs(np.mean(X_big[:, 0]) - np.mean(data[:, 0])) < 0.5


