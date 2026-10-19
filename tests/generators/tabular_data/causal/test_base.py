"""Tests for the CausalGenerator base class (validation + selector gating + intervene propagation)."""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import DGPMixin, OlsDGP
from badgers.core.dgp import edits
from badgers.generators.tabular_data.causal._base import CausalGenerator


class _Probe(CausalGenerator):
    """Concrete subclass exposing the base-class plumbing for testing."""

    deficit_name = "probe"

    def generate(self, X, y, **params):
        raise NotImplementedError


def test_validate__returns_sets(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=_FakeDGP(graph, column_mapping, {}))
    order, node_to_col, descendants, exogenous, perturb_cols = gen._validate(
        X, ["X"]
    )
    assert set(order) == {"X", "Y", "Z"}
    assert descendants == {"Y", "Z"}
    assert exogenous == set()  # X has no ancestors
    assert perturb_cols == [0]


def test_validate__missing_dgp_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    # no DGP configured -> the generator has no structure at all
    with pytest.raises(ValueError, match="DGP"):
        gen._validate(X, ["X"])


def test_validate__unknown_perturbation_node_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=_FakeDGP(graph, column_mapping, {}))
    with pytest.raises(ValueError, match="not found in graph"):
        gen._validate(X, ["NOPE"])


def test_fire_masks__none_selector_fires_all_rows(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=_FakeDGP(graph, column_mapping, {}))
    fires = gen._fire_masks(X, ["X"], column_mapping, selector=None)
    assert set(fires) == {"X"}
    assert fires["X"].all()


def test_fire_masks__delegates_to_selector(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=_FakeDGP(graph, column_mapping, {}))

    def selector(X, y=None):
        return {"X": np.zeros(X.shape[0], dtype=bool)}

    fires = gen._fire_masks(X, ["X"], column_mapping, selector=selector)
    assert not fires["X"].any()


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


class _FakeDGP(DGPMixin):
    """Minimal deterministic DGP backend: linear mechanisms."""

    def __init__(self, graph, column_mapping, beta, noise_std=0.0):
        self.graph = graph
        self.column_mapping = column_mapping
        self._beta = beta          # node -> {parent: coef}
        self._noise_std = noise_std
        self._fit_called = False

    def fit(self, X):
        self._fit_called = True
        return self

    def is_fitted(self):
        return self._fit_called

    def response(self, node, parent_values):
        parents = list(self.graph.predecessors(node))
        coefs = np.array([self._beta[node][p] for p in parents])
        return parent_values @ coefs

    def draw_noise(self, node, rng, n):
        return rng.normal(0, self._noise_std, size=n)


def test_intervene__linear_chain_recompute(chain_data):
    # Recompute semantics: descendant_new = f(child, parents_new) + fresh
    # noise, for ALL descendants in topological order (multi-hop chains).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    rows = X.copy()
    rows[:, 0] += 3.0  # perturb the root in place
    out = gen.dgp.intervene(rows, ["X"], rng=np.random.default_rng(0))
    # Root column untouched by propagation; descendants re-derived exactly
    # (noise-free fake mechanism).
    np.testing.assert_allclose(out[:, 0], X[:, 0] + 3.0)
    np.testing.assert_allclose(out[:, 1], 2.0 * (X[:, 0] + 3.0))
    np.testing.assert_allclose(out[:, 2], 0.5 * (2.0 * (X[:, 0] + 3.0)))


def test_intervene__zero_perturbation_is_deterministic_recompute(chain_data):
    # With a noise-free mechanism, a zero perturbation recomputes the
    # descendants to their deterministic values (the fixture's Y/Z carry
    # noise: the recompute is NOT bit-identical to the input — fresh noise
    # is part of the recompute contract even for a zero shift).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    out = gen.dgp.intervene(X.copy(), ["X"], rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 0], X[:, 0])
    np.testing.assert_allclose(out[:, 1], 2.0 * X[:, 0])
    np.testing.assert_allclose(out[:, 2], 0.5 * (2.0 * X[:, 0]))


def test_intervene__recompute_redraws_descendant_noise(chain_data):
    # Recompute: descendants = f(child, parents_new) + fresh noise (not a
    # delta on the original values).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator()
    gen.dgp = _FakeDGP(graph, column_mapping,
                       {"Y": {"X": 2.0}, "Z": {"Y": 0.5}}, noise_std=0.3)
    rows = X.copy()
    rows[:, 0] += 3.0  # perturb the root in place
    out = gen.dgp.intervene(rows, ["X"], rng=np.random.default_rng(0))
    # descendants are recomputed from the perturbed parents WITH noise:
    deterministic_y = 2.0 * (X[:, 0] + 3.0)
    assert np.std(out[:, 1] - deterministic_y) > 0.15


def test_lazy_fit__unfitted_constructor_dgp_is_fitted_on_generate(chain_data):
    # New contract: an UNFITTED DGP passed to the constructor is fitted
    # lazily inside _require_dgp on the generate-time data (no public
    # generator.fit exists).
    X, graph, column_mapping = chain_data
    fake = _FakeDGP(graph, column_mapping, {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    gen = CausalGenerator(dgp=fake)
    assert fake._fit_called is False
    assert gen.graph is graph
    gen._require_dgp(X)
    assert gen.dgp is fake
    assert fake._fit_called is True
    assert fake.is_fitted()


def test_lazy_fit__fitted_constructor_dgp_is_untouched(chain_data):
    # A fitted DGP is never refitted: is_fitted() short-circuits the lazy
    # fit (the whole point of the explicit-fit pattern).
    X, graph, column_mapping = chain_data
    fake = _FakeDGP(graph, column_mapping, {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    fake.fit(X)
    gen = CausalGenerator(dgp=fake)
    gen._require_dgp(X)
    assert fake._fit_called is True  # from the explicit fit, not again
    assert gen.dgp is fake


def test_use_dgp__attaches_fitted_dgp(chain_data):
    # use_dgp(dgp) attaches an already-fitted backend and syncs the
    # structure from it (shared-across-a-pipeline contract).
    X, graph, column_mapping = chain_data
    fake = _FakeDGP(graph, column_mapping, {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    fake.fit(X)
    gen = CausalGenerator()
    gen.use_dgp(fake)
    assert gen.dgp is fake
    assert gen.graph is fake.graph
    assert gen.column_mapping is fake.column_mapping
    gen._require_fitted()  # structure came from the DGP: no fit needed
    rows = X.copy()
    rows[:, 0] += 3.0
    out = gen.dgp.intervene(rows, ["X"], rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 1], 2.0 * (X[:, 0] + 3.0))


def test_rebuild_with_graph__ols_rebuilds_on_modified_graph(chain_data):
    # A3 option (b) refit route: a fresh OlsDGP is built for the modified
    # graph and fitted on the clean entry data (no training-data retention).
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=OlsDGP(graph, column_mapping))
    g2 = graph.copy()
    g2.add_edge("X", "Z")  # 2 parents for Z
    rebuilt = gen.rebuild_with_graph(X, g2)
    assert type(rebuilt).__name__ == "OlsDGP"
    assert rebuilt.graph is g2
    assert rebuilt.column_mapping is column_mapping
    # fitted: response() works for Z with both parents
    vals = rebuilt.response("Z", np.array([[1.0, 1.0]]))
    assert vals.shape == (1,)
    # adopting the rebuilt world keeps the generator consistent
    gen.use_dgp(rebuilt)
    assert gen.graph is g2


def test_rebuild_with_graph__preserves_backend_class(chain_data):
    # The rebuilt world uses the SAME backend class as the current DGP
    # (object-injection contract: dowhy stays dowhy).
    pytest.importorskip("dowhy.gcm")
    from badgers.core.dgp.backends import DowhyGcmDGP
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(dgp=DowhyGcmDGP(graph, column_mapping))
    g2 = graph.copy()
    g2.add_edge("X", "Z")
    rebuilt = gen.rebuild_with_graph(X, g2)
    assert type(rebuilt) is DowhyGcmDGP
    assert rebuilt.graph is g2


def test_add_edge_contribution__guides_to_rebuild_for_estimation(chain_data):
    # A3 decision (b): an ESTIMATED new parent needs a refit, which is not
    # possible without training data — the error points at the rebuild
    # route (edits.rebuild / rebuild_with_graph).
    X, graph, column_mapping = chain_data
    fake = _FakeDGP(graph, column_mapping, {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    fake.fit(X)
    with pytest.raises(ValueError, match="(?i)rebuild"):
        edits.add_edge_contribution(fake, "Z", "X")
