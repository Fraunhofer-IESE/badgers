"""Shared fixtures and helpers for the causal defect generator tests."""

import networkx as nx
import numpy as np
import pytest


def _generate_linear_data(graph, n_samples, rng, coefficients=None,
                          column_mapping=None):
    """Generate data following linear structural equations.

    Each node v = sum_{p in parents(v)} coef[p->v] * p + N(0, 0.1).
    Root nodes ~ N(0, 1).
    """
    if coefficients is None:
        coefficients = {(u, v): 1.0 for u, v in graph.edges}
    if column_mapping is None:
        column_mapping = {n: i for i, n in enumerate(sorted(graph.nodes))}
    order = list(nx.topological_sort(graph))
    d = len(graph.nodes)
    X = np.zeros((n_samples, d))
    for node in order:
        col = column_mapping[node]
        parents = list(graph.predecessors(node))
        if parents:
            parent_cols = [column_mapping[p] for p in parents]
            coefs = np.array([coefficients.get((p, node), 1.0) for p in parents])
            X[:, col] = (
                X[:, parent_cols] @ coefs
                + rng.normal(0, 0.1, size=n_samples)
            )
        else:
            X[:, col] = rng.normal(0, 1.0, size=n_samples)
    return X, column_mapping


@pytest.fixture
def chain_data(rng):
    """X -> Y -> Z linear data with known coefficients."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X, column_mapping = _generate_linear_data(
        graph, n_samples=100, rng=rng, coefficients=coefs
    )
    return X, graph, column_mapping
