"""Conformance tests for the ``DowhyGcmDGP`` backend (dowhy extra).

Per the structural-modifications resolution (D2): the backend implements
only the per-node protocol core and inherits the generic
``fit``/``sample``/``intervene``/edit surface from ``DGPMixin``; these tests
pin that the inherited machinery behaves identically to the reference
``OlsDGP`` path (within tolerance â€” GCM's linear regressor carries an
intercept and its noise is empirical, not parametric).
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import OlsDGP
from badgers.core.dgp import edits

pytest.importorskip("dowhy.gcm")

from badgers.core.dgp.backends import DowhyGcmDGP  # noqa: E402


def _chain_data(n=300, seed=0, noise=0.1):
    """X -> Y -> Z; Y = 2X + e, Z = 0.5Y + e."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    X = rng.normal(size=n)
    Y = 2.0 * X + rng.normal(0, noise, size=n)
    Z = 0.5 * Y + rng.normal(0, noise, size=n)
    return np.column_stack([X, Y, Z]), graph, column_mapping


def test_fit_node_response_matches_ols_reference():
    # Both backends fit least squares on the same data: the deterministic
    # part must agree within intercept tolerance.
    data, graph, column_mapping = _chain_data()
    ours = DowhyGcmDGP(graph, column_mapping).fit(data)
    ref = OlsDGP(graph, column_mapping).fit(data)
    parents = np.array([[1.0], [-2.0], [3.0]])
    # tolerance: GCM's default linear regressor is slightly regularized
    np.testing.assert_allclose(
        ours.response("Y", parents), ref.response("Y", parents), atol=1e-2
    )
    np.testing.assert_allclose(
        ours.response("Z", parents), ref.response("Z", parents), atol=1e-2
    )


def test_draw_noise_is_empirical_residual_resampling():
    # The shock distribution is the FITTED RESIDUALS (not a parametric
    # normal): its std must match the residual std of the data.
    data, graph, column_mapping = _chain_data()
    dgp = DowhyGcmDGP(graph, column_mapping).fit(data)
    noise = dgp.draw_noise("Z", np.random.default_rng(0), 5000)
    assert np.std(noise) == pytest.approx(
        np.std(data[:, 2] - dgp.response("Z", data[:, [1]])), rel=0.15
    )
    # draw_noise depends on the caller's rng (reproducibility contract).
    n2 = dgp.draw_noise("Z", np.random.default_rng(0), 5)
    n3 = dgp.draw_noise("Z", np.random.default_rng(0), 5)
    np.testing.assert_array_equal(n2, n3)


def test_root_node_is_empirical_distribution():
    data, graph, column_mapping = _chain_data()
    dgp = DowhyGcmDGP(graph, column_mapping).fit(data)
    # response = fitted mean; noise = resampled centered observations
    assert dgp.response("X", np.zeros((4, 0))) == pytest.approx(
        np.mean(data[:, 0])
    )
    shocks = dgp.draw_noise("X", np.random.default_rng(0), 5000)
    assert np.std(shocks) == pytest.approx(np.std(data[:, 0]), rel=0.1)
    np.testing.assert_allclose(np.mean(shocks), 0.0, atol=0.05)


def test_inherited_intervene_semantics():
    # Pearl surgery through the inherited mixin path: descendants recomputed
    # with fresh (empirical) noise, upstream columns untouched.
    data, graph, column_mapping = _chain_data()
    dgp = DowhyGcmDGP(graph, column_mapping).fit(data)
    before = data.copy()
    out = dgp.intervene(data.copy(), ["X"], values=[data[:, 0] + 3.0],
                        rng=np.random.default_rng(1))
    np.testing.assert_allclose(out[:, 0], before[:, 0] + 3.0)
    dY = out[:, 1] - before[:, 1]
    dY_det = (
        dgp.response("Y", out[:, [0]]) - dgp.response("Y", before[:, [0]])
    )
    np.testing.assert_allclose(np.mean(dY - dY_det), 0.0, atol=0.1)


def test_inherited_edits_work():
    # scale_edge (weaken the edge â€” feasible within the variance budget) /
    # exogenize views compose over the dowhy backend.
    data, graph, column_mapping = _chain_data()
    dgp = DowhyGcmDGP(graph, column_mapping).fit(data)
    scaled = edits.scale_edge(dgp, "Y", "X", 0.8).calibrate(data)
    assert scaled.column_mapping is dgp.column_mapping
    exo = edits.exogenize(dgp, "Y")
    assert list(exo.graph.predecessors("Y")) == []
    assert dgp.graph is graph  # base untouched


def test_sample_shape_and_finite():
    data, graph, column_mapping = _chain_data()
    dgp = DowhyGcmDGP(graph, column_mapping).fit(data)
    s = dgp.sample(np.random.default_rng(0), 50)
    assert s.shape == (50, 3)
    assert np.all(np.isfinite(s))


def test_complete_case_fit_too_few_rows_raises():
    data, graph, column_mapping = _chain_data(n=10)
    data[:, 1] = np.nan
    dgp = DowhyGcmDGP(graph, column_mapping)
    with pytest.raises(ValueError, match="complete-case"):
        dgp.fit(data)


