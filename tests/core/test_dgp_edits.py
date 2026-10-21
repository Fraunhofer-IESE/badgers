"""Tests for the structure edits: exogenize, set_form, add/reverse stubs."""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import edits


def _chain_data(n=300, seed=0):
    """X -> Y -> Z; Y = 2X + e, Z = 0.5Y + e (unit variances)."""
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {"X": 0, "Y": 1, "Z": 2}
    X = rng.normal(size=n)
    Y = 2.0 * X + rng.normal(0, 1.0, size=n)
    Z = 0.5 * Y + rng.normal(0, 1.0, size=n)
    return np.column_stack([X, Y, Z]), graph, column_mapping


def test_exogenize_removes_incoming_edges(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    exo = edits.exogenize(base, "Y")
    # Y is a root in the view's graph (its own copy); the base is untouched
    assert list(exo.graph.predecessors("Y")) == []
    assert list(base.graph.predecessors("Y")) == ["X"]
    assert exo.graph is not base.graph
    assert exo.column_mapping is base.column_mapping
    # response of the exogenized node is zero (contributes nothing back)
    np.testing.assert_allclose(
        exo.response("Y", np.zeros((4, 0))), np.zeros(4)
    )
    # other mechanisms delegate
    np.testing.assert_allclose(
        exo.response("Z", np.array([[1.0], [2.0]])),
        base.response("Z", np.array([[1.0], [2.0]])),
    )


def test_exogenize_unknown_node_raises(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    with pytest.raises(ValueError, match="not in graph"):
        edits.exogenize(base, "NOPE")


def test_exogenize_cascade_never_touches_the_root(make_gcm_dgp):
    # intervene() recomputes only descendants; an exogenized node is a root
    # and cannot be a descendant, so a downstream intervention leaves the
    # node's column exactly as the caller wrote it.
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    exo = edits.exogenize(base, "Y")
    rows = data.copy()
    fixed = rows[:, 1] + 7.0
    out = exo.intervene(rows, ["Y"], values=[fixed],
                        rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 1], fixed)   # pinned root untouched
    # Z recomputed from the intervened Y through the base mechanism
    det_z = base.response("Z", out[:, [1]])
    assert np.allclose(out[:, 2] - det_z, out[:, 2] - det_z)  # finite
    np.testing.assert_allclose(np.mean(out[:, 2] - det_z), 0.0, atol=0.3)


def test_set_form_replaces_response(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    parents = np.array([[1.0], [2.0]])
    base_ref = base.response("Y", parents).copy()
    view = edits.set_form(base, "Y", lambda p: 3.0 * p[:, 0])
    np.testing.assert_allclose(view.response("Y", parents), [3.0, 6.0])
    # the base is untouched: its response is the pre-view fitted mechanism
    np.testing.assert_allclose(base.response("Y", parents), base_ref)
    # non-overridden nodes delegate
    np.testing.assert_allclose(
        view.response("Z", parents),
        base.response("Z", parents),
    )


def test_set_form_broadcasts_scalar_responses(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.set_form(base, "Y", lambda p: 1.5)  # constant form
    np.testing.assert_allclose(
        view.response("Y", np.zeros((5, 1))), np.full(5, 1.5)
    )


def test_set_form_non_callable_raises(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    with pytest.raises(ValueError, match="callable"):
        edits.set_form(base, "Y", 3.0)


def test_add_edge_contribution__coefficient_honored(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.add_edge_contribution(base, "Z", "X", coefficient=0.25)
    # graph gained the edge; base untouched
    assert view.graph.has_edge("X", "Z")
    assert not base.graph.has_edge("X", "Z")
    parents = np.array([[1.0, 2.0], [3.0, 4.0]])  # view order: [Y, X]
    det = base.response("Z", parents[:, [0]])  # base order: [Y]
    np.testing.assert_allclose(
        view.response("Z", parents), det + 0.25 * parents[:, 1]
    )


def test_add_edge_contribution__closure_contribution(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.add_edge_contribution(
        base, "Z", "X", contribution=lambda x: np.exp(x)
    )
    parents = np.array([[0.0, 1.0], [2.0, 3.0]])  # view order: [Y, X]
    det = base.response("Z", parents[:, [0]])  # base order: [Y]
    np.testing.assert_allclose(
        view.response("Z", parents), det + np.exp(parents[:, 1])
    )


def test_add_edge_contribution__descendants_and_intervene(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.add_edge_contribution(base, "Z", "X", coefficient=0.5)
    rows = data.copy()
    out = view.intervene(rows, ["Y"], values=[data[:, 1] + 1.0],
                         rng=np.random.default_rng(0))
    # Z recomputed through the AUGMENTED mechanism (extra X term present)
    parents = out[:, [1, 0]]  # predecessors(Z) = [Y, X] in view graph
    det = base.response("Z", parents[:, [0]]) + 0.5 * out[:, 0]
    assert np.std(out[:, 2] - det) < 1.2
    assert abs(np.mean(out[:, 2] - det)) < 0.3


def test_add_edge_contribution__cycle_raises(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    # adding X->Z is fine, but adding Z->X would create a cycle
    with pytest.raises(ValueError, match="cycle"):
        edits.add_edge_contribution(base, "X", "Z", coefficient=0.5)


def test_add_edge_contribution__missing_term_raises(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    with pytest.raises(ValueError, match="contribution|coefficient"):
        edits.add_edge_contribution(base, "Z", "X")
    with pytest.raises(ValueError, match="not both"):
        edits.add_edge_contribution(base, "Z", "X", coefficient=0.5,
                                    contribution=lambda x: x)


def test_set_form__custom_draw_noise_used_for_the_node(make_gcm_dgp):
    # Q2 option (b): a user-supplied draw_noise(rng, n) closure replaces
    # the inherited noise law for the edited node ONLY.
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.set_form(base,         "Y",
        lambda p: 3.0 * p[:, 0],
        draw_noise=lambda rng, n: rng.normal(0.0, 5.0, size=n),
    )
    z = view.draw_noise("Y", np.random.default_rng(0), 500)
    assert np.isclose(np.std(z), 5.0, rtol=0.15)
    # other nodes keep the inherited law
    assert np.std(view.draw_noise("X", np.random.default_rng(0), 500)) < 1.5


def test_set_form__custom_noise_flows_into_intervene(make_gcm_dgp):
    # The custom noise law is used by the mixin's intervene recompute.
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.set_form(base,         "Y",
        lambda p: 2.0 * p[:, 0],
        draw_noise=lambda rng, n: rng.normal(0.0, 0.0, size=n),
    )
    rows = data.copy()
    rows[:, 0] += 3.0
    out = view.intervene(rows, ["X"], rng=np.random.default_rng(0))
    # zero-noise custom law: Y recomputed EXACTLY, deterministic
    np.testing.assert_allclose(out[:, 1], 2.0 * (data[:, 0] + 3.0))
    # Z (descendant) recomputed through the custom deterministic Y; Z
    # carries its OWN fresh noise (std ~1), so only the residual spread is
    # bounded â€” Z stays centered around 0.5*Y.
    diff = out[:, 2] - 0.5 * out[:, 1]
    assert np.std(diff) < 1.2
    assert abs(np.mean(diff)) < 0.3


def test_set_form__draw_noise_must_be_callable_or_none(make_gcm_dgp):
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    with pytest.raises(ValueError, match="draw_noise must be callable"):
        edits.set_form(base, "Y", lambda p: p[:, 0], draw_noise=42)


def test_edits_compose(make_gcm_dgp):
    # Views compose by delegation: form edit inside scale edit (order
    # irrelevant here — different nodes).
    data, graph, column_mapping = _chain_data()
    base = make_gcm_dgp(graph, column_mapping).fit(data)
    view = edits.scale_edge(
        edits.set_form(base, "Y", lambda p: 3.0 * p[:, 0]), "Z", "Y", 2.0
    )
    assert view.column_mapping is base.column_mapping
    parents_y = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(view.response("Y", parents_y), [3.0, 6.0])


