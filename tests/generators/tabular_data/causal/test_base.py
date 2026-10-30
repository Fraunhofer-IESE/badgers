"""Tests for the CausalGenerator base class (validation + selector selection + intervene propagation)."""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import DGPMixin
from badgers.generators.tabular_data.causal._base import CausalGenerator


class _Probe(CausalGenerator):
    """Concrete subclass exposing the base-class plumbing for testing."""

    deficit_name = "probe"

    def generate(self, X, **params):
        raise NotImplementedError


def test_validate_structure__ok(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=_FakeDGP(graph, column_mapping, {}))
    assert gen._validate_structure(X) is None
    assert gen._validate_perturbation_nodes(["X"]) is None


def test_validate_structure__missing_dgp_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
    # no DGP configured -> the generator has no structure at all
    with pytest.raises(ValueError, match="DGP"):
        gen._validate_structure(X)


def test_validate_structure__bad_column_count_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=_FakeDGP(graph, column_mapping, {}))
    with pytest.raises(ValueError, match="columns but graph"):
        gen._validate_structure(X[:, :2])


def test_validate_perturbation_nodes__missing_dgp_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="DGP"):
        gen._validate_perturbation_nodes(["X"])


def test_validate_perturbation_nodes__unknown_node_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=_FakeDGP(graph, column_mapping, {}))
    with pytest.raises(ValueError, match="not found in graph"):
        gen._validate_perturbation_nodes(["NOPE"])


def test_validate_perturbation_nodes__not_a_list_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=_FakeDGP(graph, column_mapping, {}))
    with pytest.raises(ValueError, match="must be a list"):
        gen._validate_perturbation_nodes("X")


def test_validate_perturbation_nodes__empty_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=_FakeDGP(graph, column_mapping, {}))
    with pytest.raises(ValueError, match="not be empty"):
        gen._validate_perturbation_nodes([])


def test_deficit_labels__start_as_none(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
    assert gen.deficit_labels_ is None


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
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
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
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
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
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
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
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=fake)
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
    gen = CausalGenerator(random_generator=np.random.default_rng(0), dgp=fake)
    gen._require_dgp(X)
    assert fake._fit_called is True  # from the explicit fit, not again
    assert gen.dgp is fake


def test_use_dgp__attaches_fitted_dgp(chain_data):
    # use_dgp(dgp) attaches an already-fitted backend and syncs the
    # structure from it (shared-across-a-pipeline contract).
    X, graph, column_mapping = chain_data
    fake = _FakeDGP(graph, column_mapping, {"Y": {"X": 2.0}, "Z": {"Y": 0.5}})
    fake.fit(X)
    gen = CausalGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(fake)
    assert gen.dgp is fake
    assert gen.graph is fake.graph
    assert gen.column_mapping is fake.column_mapping
    gen._require_fitted()  # structure came from the DGP: no fit needed
    rows = X.copy()
    rows[:, 0] += 3.0
    out = gen.dgp.intervene(rows, ["X"], rng=np.random.default_rng(0))
    np.testing.assert_allclose(out[:, 1], 2.0 * (X[:, 0] + 3.0))



