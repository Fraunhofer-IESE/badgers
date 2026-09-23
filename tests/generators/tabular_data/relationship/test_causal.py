"""Tests for CausalEdgePerturbationGenerator."""

import networkx as nx
import numpy as np
import pytest

from badgers.generators.tabular_data.relationship.causal import (
    CausalEdgePerturbationGenerator,
)


def _weak_edge_data(n=100, rng=None):
    """Chain X -> Y -> Z where the X->Y edge is WEAK (large residual).

    Y = 0.3 * X + N(0, 1.0)  ->  R^2(X->Y) ~ 0.08, so doubling the edge
    (beta' ~ 0.6) stays comfortably feasible while preserving Y's marginal,
    even with a noisy short-MCMC estimate of beta.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    column_mapping = {n: i for i, n in enumerate(sorted(graph.nodes))}
    X = rng.normal(0, 1, size=n)
    Y = 0.3 * X + rng.normal(0, 1.0, size=n)
    Z = 0.5 * Y + rng.normal(0, 0.5, size=n)
    return np.column_stack([X, Y, Z]), graph, column_mapping


def test_generate__edge_relationship_strengthened():
    X, graph, column_mapping = _weak_edge_data(rng=np.random.default_rng(0))
    # Distinct seed from the data rng: the residual drawn during propagate
    # must not be a scaled copy of X (a shared seed would inflate the result).
    gen = CausalEdgePerturbationGenerator(random_generator=np.random.default_rng(42))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    base_corr = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    base_var_y = np.var(X[:, 1])
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["Y"],
        mechanism="always", edge=("X", "Y"), weight_factor=2.0,
    )
    # in-place: same shape, all finite
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))
    assert list(yt) == ["relationship"] * X.shape[0]
    # The X->Y relationship is doubled: Y's dependence on X strengthens,
    # while Y's own marginal variance is preserved.
    new_corr = abs(np.corrcoef(Xt[:, 0], Xt[:, 1])[0, 1])
    assert new_corr > base_corr
    assert abs(np.var(Xt[:, 1]) - base_var_y) < 0.5 * base_var_y


def test_generate__marginal_preserved():
    X, graph, column_mapping = _weak_edge_data(rng=np.random.default_rng(0))
    # Distinct seed from the data rng: the residual drawn during propagate
    # must not be a scaled copy of X (a shared seed would inflate the result).
    gen = CausalEdgePerturbationGenerator(random_generator=np.random.default_rng(42))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    base_var_y = np.var(X[:, 1])
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["Y"],
        mechanism="always", edge=("X", "Y"), weight_factor=2.0,
    )
    # Marginal-preserving: Y's variance is preserved (within tolerance)
    assert abs(np.var(Xt[:, 1]) - base_var_y) < 0.5 * base_var_y


def test_generate__infeasible_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalEdgePerturbationGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    # weight_factor huge -> beta'^2 * var(X) > var(Y) -> infeasible
    with pytest.raises(ValueError, match="infeasible"):
        gen.generate(
            X, y=None, perturbation_nodes=["Y"],
            mechanism="always", edge=("X", "Y"), weight_factor=100.0,
        )


def test_generate__perturbation_nodes_must_be_child(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalEdgePerturbationGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    with pytest.raises(ValueError, match="perturbation_nodes"):
        gen.generate(
            X, y=None, perturbation_nodes=["X"],
            mechanism="always", edge=("X", "Y"), weight_factor=2.0,
        )


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalEdgePerturbationGenerator()
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(
            X, y=None, perturbation_nodes=["Y"], edge=("X", "Y"),
        )
    gen.fit(X, graph, column_mapping, n_samples_mcmc=50)
    with pytest.raises(ValueError, match="edge"):
        gen.generate(
            X, y=None, perturbation_nodes=["Y"], edge=None,
        )
