"""Conformance tests for the ``PyMCDGP`` backend (pymc extra).

Per the structural-modifications resolution (D2): the backend implements
only the per-node protocol core (its own tiny MCMC per node) plus
only the per-node protocol core (its own tiny MCMC per node) plus the
``pick_parameters`` hook and inherits the generic ``fit``/``sample``/
``intervene`` surface from ``DGPMixin``. The FULL posterior is kept —
NO mean collapse: ``response``/``draw_noise`` on the carrier evaluate
draw 0 (a canonical parameter set); each ``sample``/``intervene`` call
pins ONE draw internally via the hook. Oracles are necessarily LOOSE
(short-MCMC beta noise is a real effect).

Gotchas pinned here (session lessons): distinct seeds for the data and
the test rng (shared seeds make drawn residuals correlates of the data).
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import edits

pytest.importorskip("pymc")

from badgers.core.dgp.backends import PyMCDGP  # noqa: E402
from badgers.core.dgp.backends.pymc import _ParameterDrawView  # noqa: E402

N_MCMC = 50  # short draws: keep the whole module's latency acceptable


def _chain_data(n=200, seed=0, noise=0.1):
    """X -> Y -> Z; Y = 2X + e, Z = 0.5Y + e."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    X = rng.normal(size=n)
    Y = 2.0 * X + rng.normal(0, noise, size=n)
    Z = 0.5 * Y + rng.normal(0, noise, size=n)
    return np.column_stack([X, Y, Z]), graph, column_mapping


@pytest.fixture(scope="module")
def fitted_chain():
    data, graph, column_mapping = _chain_data()
    dgp = PyMCDGP(graph, column_mapping, n_draws=N_MCMC).fit(data)
    return data, graph, column_mapping, dgp


def test_carrier_response_is_canonical_draw(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    parents = np.array([[1.0], [-2.0]])
    # carrier == canonical parameter set (draw 0); the pinned-index-0 view
    # must agree EXACTLY with the carrier's response (no independent draw)
    zero = _ParameterDrawView(dgp, 0)
    np.testing.assert_allclose(
        zero.response("Y", parents), dgp.response("Y", parents)
    )


def test_response_mean_agrees_with_true_slope(fitted_chain):
    # Averaging over MANY pinned parameter draws recovers the posterior
    # mean, which approximates the TRUE structural slope (2.0; loose rtol:
    # short MCMC + prior shrinkage).
    data, graph, column_mapping, dgp = fitted_chain
    parents = np.array([[1.0]])
    preds = np.array([
        dgp.pick_parameters(np.random.default_rng(s)).response("Y", parents)
        for s in range(40)
    ])
    np.testing.assert_allclose(np.mean(preds), 2.0, rtol=0.2)


def test_draw_noise_shapes_and_reproducibility(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    n1 = dgp.draw_noise("Z", np.random.default_rng(0), 100)
    n2 = dgp.draw_noise("Z", np.random.default_rng(0), 100)
    n3 = dgp.draw_noise("Z", np.random.default_rng(1), 100)
    np.testing.assert_array_equal(n1, n2)
    assert n1.shape == (100,)
    assert not np.allclose(n1, n3)
    # carrier noise = draw-0 sigma (small residual for the noisy fixture)
    assert np.std(n1) < 1.0


def test_pick_parameters_pins_one_draw(fitted_chain):
    # Two pinned draws DIFFER (posterior spread is real), and each draw is
    # internally coherent: repeated response calls on ONE pin are identical
    # (deterministic given the draw — protocol constraint).
    data, graph, column_mapping, dgp = fitted_chain
    p = np.array([[1.0]])
    w1 = dgp.pick_parameters(np.random.default_rng(0))
    w2 = dgp.pick_parameters(np.random.default_rng(1))
    assert not np.isclose(w1.response("Y", p)[0], w2.response("Y", p)[0],
                          atol=1e-12)
    np.testing.assert_array_equal(
        w1.response("Y", p), w1.response("Y", p)
    )
    # reproducible draw selection under a fixed rng
    w1b = dgp.pick_parameters(np.random.default_rng(0))
    np.testing.assert_allclose(w1.response("Y", p), w1b.response("Y", p))
    # idempotent: a pinned view re-picks itself
    assert w1.pick_parameters(np.random.default_rng(5)) is w1


def test_root_node_distribution(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    shocks = dgp.pick_parameters(
        np.random.default_rng(3)
    ).draw_noise("X", np.random.default_rng(0), 500)
    assert 0.5 < np.std(shocks) < 2.0  # ~ N(0,1) root (drawn sigma)


def test_inherited_intervene_mean_shift(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    before = data.copy()
    out = dgp.intervene(data.copy(), ["X"], values=[data[:, 0] + 3.0],
                        rng=np.random.default_rng(42))  # DISTINCT seed
    np.testing.assert_allclose(out[:, 0], before[:, 0] + 3.0)
    dY = out[:, 1] - before[:, 1]
    # carrier world (draw 0) beta ~ 2.0: mean shift ~ 6 within tolerance
    np.testing.assert_allclose(np.mean(dY), 6.0, atol=1.0)
    dZ = out[:, 2] - before[:, 2]
    np.testing.assert_allclose(np.mean(dZ), 3.0, atol=0.8)


def test_intervene_pins_one_draw_and_is_coherent(fitted_chain):
    # The generic cascade pins ONE parameter draw at the top (via the
    # hook): pinned node, untouched upstream, descendants recomputed
    # coherently with THAT draw's beta.
    data, graph, column_mapping, dgp = fitted_chain
    out = dgp.intervene(data.copy(), ["X"], values=[data[:, 0] + 3.0],
                        rng=np.random.default_rng(7))
    np.testing.assert_allclose(out[:, 0], data[:, 0] + 3.0)
    dY = out[:, 1] - data[:, 1]
    # pinned beta ~ 2 -> mean shift ~6 within loose tolerance
    np.testing.assert_allclose(np.mean(dY), 6.0, atol=1.0)
    # coherent: Y column consistent with SOME single beta (~2 +- wobble);
    # residual spread must be small relative to the shift itself
    resid = out[:, 1] - data[:, 0] * 2.0
    assert np.std(resid) < 0.5
    # edited DGP: the pin still applies (delegating hook)
    scaled = edits.scale_edge(dgp, "Y", "X", 0.8).calibrate(data)
    out2 = scaled.intervene(data.copy(), ["X"],
                            values=[data[:, 0] + 3.0],
                            rng=np.random.default_rng(8))
    dY2 = out2[:, 1] - data[:, 1]
    np.testing.assert_allclose(np.mean(dY2), 0.8 * 6.0, atol=1.5)


def test_inherited_edits_work(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    # weakening the edge is feasible within the variance budget (k > 1 on
    # a saturated single-parent chain would always be infeasible)
    scaled = edits.scale_edge(dgp, "Y", "X", 0.8).calibrate(data)
    assert scaled.column_mapping is dgp.column_mapping
    exo = edits.exogenize(dgp, "Y")
    assert list(exo.graph.predecessors("Y")) == []


def test_sample_shape_and_finite(fitted_chain):
    data, graph, column_mapping, dgp = fitted_chain
    s = dgp.sample(np.random.default_rng(0), 20)
    assert s.shape == (20, 3)
    assert np.all(np.isfinite(s))


def test_requires_fitted_for_response():
    data, graph, column_mapping = _chain_data(n=10)
    dgp = PyMCDGP(graph, column_mapping)
    with pytest.raises(RuntimeError, match="before fit_node"):
        dgp.response("Y", np.zeros((2, 1)))


