import networkx as nx
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data(request, rng):
    """Parametrized fixture yielding X (features only) for each input type."""
    if request.param == "numpy_1D":
        return rng.normal(size=100).reshape(-1, 1)
    elif request.param == "numpy_2D":
        return rng.normal(size=(100, 10))
    elif request.param == "pandas_1D":
        return pd.DataFrame(rng.normal(size=100).reshape(-1, 1), columns=["col"])
    elif request.param == "pandas_2D":
        return pd.DataFrame(
            rng.normal(size=(100, 10)),
            columns=[f"col{i}" for i in range(10)],
        )


@pytest.fixture(params=["numpy_1D", "numpy_2D", "pandas_1D", "pandas_2D"])
def tabular_data_labeled(request, rng):
    """Parametrized fixture yielding a labeled matrix X whose LAST column holds
    5-class classification values (added as column "cls" for pandas inputs)."""
    if request.param == "numpy_1D":
        X = rng.normal(size=100).reshape(-1, 1)
        y = np.array([i % 5 for i in range(100)])
        return np.column_stack([X, y]).astype(float)
    elif request.param == "numpy_2D":
        X = rng.normal(size=(100, 10))
        y = np.array([0, 1, 2, 3, 4] * 20)
        return np.column_stack([X, y]).astype(float)
    elif request.param == "pandas_1D":
        X = pd.DataFrame(rng.normal(size=100).reshape(-1, 1), columns=["col"])
        return X.assign(cls=[i % 5 for i in range(100)])
    elif request.param == "pandas_2D":
        X = pd.DataFrame(
            rng.normal(size=(100, 10)),
            columns=[f"col{i}" for i in range(10)],
        )
        return X.assign(cls=[0, 1, 2, 3, 4] * 20)


@pytest.fixture
def chain_graph():
    """X -> Y -> Z"""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Y", "Z")])
    return g


@pytest.fixture
def fork_graph():
    """C -> X, C -> Y"""
    g = nx.DiGraph()
    g.add_edges_from([("C", "X"), ("C", "Y")])
    return g


@pytest.fixture
def collider_graph():
    """X -> Z <- Y"""
    g = nx.DiGraph()
    g.add_edges_from([("X", "Z"), ("Y", "Z")])
    return g


@pytest.fixture
def diamond_graph():
    """A -> B, A -> C, B -> D, C -> D"""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return g


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
