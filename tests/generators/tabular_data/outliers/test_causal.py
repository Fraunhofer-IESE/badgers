import networkx as nx
import numpy as np
import pytest

from badgers.generators.tabular_data.outliers.causal import CausalOutlierPropagationGenerator


# --- Helper: generate linear data from a graph ---

def _generate_linear_data(graph, n_samples, rng, coefficients=None):
    """
    Generate data following linear structural equations.

    Each node v = sum_{p in parents(v)} coef[p->v] * p + noise
    where noise ~ N(0, 0.1).
    """
    if coefficients is None:
        coefficients = {}
        for u, v in graph.edges:
            coefficients[(u, v)] = 1.0

    order = list(nx.topological_sort(graph))
    node_to_idx = {n: i for i, n in enumerate(sorted(graph.nodes))}
    d = len(graph.nodes)
    X = np.zeros((n_samples, d))

    for node in order:
        col = node_to_idx[node]
        parents = list(graph.predecessors(node))
        if parents:
            parent_cols = [node_to_idx[p] for p in parents]
            coefs = np.array([coefficients.get((p, node), 1.0) for p in parents])
            X[:, col] = X[:, parent_cols] @ coefs + rng.normal(0, 0.1, size=n_samples)
        else:
            X[:, col] = rng.normal(0, 1.0, size=n_samples)

    return X


# --- Tests ---

def test_generate__chain_x_to_y(rng):
    """Perturb X in X->Y->Z: Y and Z should shift proportionally."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="X",
        outlier_magnitude=3.0, sample_index=0,
    )

    # X should be perturbed
    delta_x = Xt[0, 0] - X[0, 0]
    assert abs(delta_x) > 0

    # Y should shift by coef * delta_x (approximately, noise in fitting)
    delta_y = Xt[0, 1] - X[0, 1]
    assert abs(delta_y) > 0
    # Y shift should be roughly 2.0 * delta_x
    assert np.sign(delta_y) == np.sign(delta_x)

    # Z should shift (Y's child)
    delta_z = Xt[0, 2] - X[0, 2]
    assert abs(delta_z) > 0

    # Only row 0 should be modified
    assert np.allclose(Xt[1:], X[1:])


def test_generate__chain_y_no_propagation(rng):
    """Perturb Y in X->Y->Z: X unchanged, Z shifts."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="Y",
        outlier_magnitude=3.0, sample_index=0,
    )

    # X should NOT change (Y is not an ancestor of X)
    assert Xt[0, 0] == X[0, 0]

    # Y should be perturbed
    assert Xt[0, 1] != X[0, 1]

    # Z should shift (Y's child)
    assert Xt[0, 2] != X[0, 2]


def test_generate__fork(rng):
    """Perturb C in C->X, C->Y: both X and Y should shift."""
    graph = nx.DiGraph()
    graph.add_edges_from([("C", "X"), ("C", "Y")])
    coefs = {("C", "X"): 1.5, ("C", "Y"): 0.8}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="C",
        outlier_magnitude=3.0, sample_index=0,
    )

    # C should be perturbed
    assert Xt[0, 0] != X[0, 0]

    # Both X and Y should shift
    assert Xt[0, 1] != X[0, 1]
    assert Xt[0, 2] != X[0, 2]


def test_generate__collider(rng):
    """Perturb X in X->Z<-Y: Z shifts, Y does not."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Z"), ("Y", "Z")])
    coefs = {("X", "Z"): 1.0, ("Y", "Z"): 1.0}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="X",
        outlier_magnitude=3.0, sample_index=0,
    )

    # X should be perturbed
    assert Xt[0, 0] != X[0, 0]

    # Y should NOT change (not a descendant of X)
    assert Xt[0, 1] == X[0, 1]

    # Z should shift (child of X)
    assert Xt[0, 2] != X[0, 2]


def test_generate__diamond(rng):
    """Perturb A in diamond: all descendants B, C, D should shift."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    coefs = {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "D"): 0.5, ("C", "D"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="A",
        outlier_magnitude=3.0, sample_index=0,
    )

    # A should be perturbed
    assert Xt[0, 0] != X[0, 0]

    # B, C, D should all shift
    assert Xt[0, 1] != X[0, 1]  # B
    assert Xt[0, 2] != X[0, 2]  # C
    assert Xt[0, 3] != X[0, 3]  # D


def test_generate__no_graph_raises(rng):
    """Should raise ValueError if graph is not provided."""
    X = rng.normal(size=(100, 3))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="graph"):
        generator.generate(X, y=None, target_node="X")


def test_generate__invalid_target_raises(rng):
    """Should raise ValueError if target_node is not in the graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(100, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="not found"):
        generator.generate(X, y=None, graph=graph, target_node="Z")


def test_generate__cycle_raises(rng):
    """Should raise ValueError on a cyclic graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "X")])
    X = rng.normal(size=(100, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        generator.generate(X, y=None, graph=graph, target_node="X")


def test_generate__y_unchanged(rng):
    """y should be returned unchanged."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)
    y = np.array([0, 1] * 50)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=y, graph=graph, target_node="X",
        outlier_magnitude=3.0, sample_index=0,
    )

    assert yt is y


def test_generate__non_perturbed_rows_unchanged(rng):
    """Only the target row should be modified."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="X",
        outlier_magnitude=3.0, sample_index=5,
    )

    # Row 5 should differ
    assert not np.allclose(Xt[5], X[5])
    # All other rows should be identical
    assert np.allclose(Xt[:5], X[:5])
    assert np.allclose(Xt[6:], X[6:])


def test_generate__magnitude_zero(rng):
    """outlier_magnitude=0 should produce no change."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node="X",
        outlier_magnitude=0.0, sample_index=0,
    )

    assert np.allclose(Xt, X)


def test_generate__sample_index_out_of_bounds(rng):
    """Should raise IndexError if sample_index >= n_samples."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(IndexError):
        generator.generate(
            X, y=None, graph=graph, target_node="X",
            outlier_magnitude=3.0, sample_index=10,
        )


def test_generate__int_nodes(rng):
    """Should work with integer node labels."""
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    coefs = {(0, 1): 2.0, (1, 2): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, target_node=0,
        outlier_magnitude=3.0, sample_index=0,
    )

    assert Xt[0, 0] != X[0, 0]
    assert Xt[0, 1] != X[0, 1]
    assert Xt[0, 2] != X[0, 2]
