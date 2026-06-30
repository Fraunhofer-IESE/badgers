import networkx as nx
import numpy as np
import pytest

from badgers.generators.tabular_data.outliers.causal import CausalOutlierPropagationGenerator


# --- Helper: generate linear data from a graph ---

def _generate_linear_data(graph, n_samples, rng, coefficients=None,
                          column_mapping=None):
    """
    Generate data following linear structural equations.

    Each node v = sum_{p in parents(v)} coef[p->v] * p + noise
    where noise ~ N(0, 0.1).

    Parameters
    ----------
    column_mapping : dict of str -> int, optional
        Mapping from node names to column indices. If None, defaults to
        alphabetical sort order (for backward compatibility in tests that
        don't yet use explicit mapping).
    """
    if coefficients is None:
        coefficients = {}
        for u, v in graph.edges:
            coefficients[(u, v)] = 1.0

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
            X[:, col] = X[:, parent_cols] @ coefs + rng.normal(0, 0.1, size=n_samples)
        else:
            X[:, col] = rng.normal(0, 1.0, size=n_samples)

    return X, column_mapping


# --- Tests ---

def test_generate__chain_x_to_y(rng):
    """Perturb X in X->Y->Z: Y and Z should shift proportionally."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    # Original data should be unchanged
    assert np.allclose(Xt[:100], X)

    # Outlier rows appended
    assert Xt.shape == (105, 3)

    # yt labels
    assert list(yt[:100]) == ["original"] * 100
    assert list(yt[100:]) == ["outliers"] * 5

    # Outlier rows should differ from the mean
    mean = np.mean(X, axis=0)
    for i in range(100, 105):
        assert not np.allclose(Xt[i], mean)


def test_generate__chain_y_do_intervention(rng):
    """Perturb Y in X->Y->Z with do()-style: X unchanged, Y perturbed, Z propagates."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["Y"],
        outlier_magnitude=3.0, n_outliers=3,
    )

    # Original data unchanged
    assert np.allclose(Xt[:100], X)
    assert Xt.shape == (103, 3)

    # Outlier rows: X is exogenous (ancestor of Y), sampled from its
    # distribution but NOT perturbed. Y is perturbed directly (do-style).
    # Z propagates from both X and Y.
    outlier_rows = Xt[100:]
    mean_X = np.mean(X[:, 0])
    mean_Y = np.mean(X[:, 1])
    # X should be near its original mean (sampled from distribution, not perturbed)
    assert np.all(np.abs(outlier_rows[:, 0] - mean_X) < 5.0)
    # Y should be far from the mean (perturbed)
    assert np.all(np.abs(outlier_rows[:, 1] - mean_Y) > 0)


def test_generate__fork(rng):
    """Perturb C in C->X, C->Y: both X and Y should shift."""
    graph = nx.DiGraph()
    graph.add_edges_from([("C", "X"), ("C", "Y")])
    coefs = {("C", "X"): 1.5, ("C", "Y"): 0.8}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["C"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)

    # Outlier rows: C perturbed, X and Y should shift
    outlier_rows = Xt[100:]
    mean_C = np.mean(X[:, 0])
    mean_X = np.mean(X[:, 1])
    mean_Y = np.mean(X[:, 2])
    assert np.all(np.abs(outlier_rows[:, 0] - mean_C) > 0)
    assert np.all(np.abs(outlier_rows[:, 1] - mean_X) > 0)
    assert np.all(np.abs(outlier_rows[:, 2] - mean_Y) > 0)


def test_generate__collider(rng):
    """Perturb X in X->Z<-Y: Z shifts, Y does not (Y is root, gets noise only)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Z"), ("Y", "Z")])
    coefs = {("X", "Z"): 1.0, ("Y", "Z"): 1.0}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)

    # Outlier rows: X perturbed, Y is exogenous (ancestor of X? No — Y has
    # no path to X, so Y is NOT an ancestor of X). With do(X), Y is
    # unaffected and stays at zero (no causal path from X to Y).
    # Z propagates from X (and Y, but Y=0).
    outlier_rows = Xt[100:]
    mean_X = np.mean(X[:, 0])
    # X should be far from mean (perturbed)
    assert np.all(np.abs(outlier_rows[:, 0] - mean_X) > 0)
    # Z should shift (child of X)
    mean_Z = np.mean(X[:, 2])
    assert np.all(np.abs(outlier_rows[:, 2] - mean_Z) > 0)


def test_generate__diamond(rng):
    """Perturb A in diamond: all descendants B, C, D should shift."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    coefs = {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "D"): 0.5, ("C", "D"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["A"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 4)
    assert np.allclose(Xt[:100], X)

    # All outlier columns should differ from their means
    outlier_rows = Xt[100:]
    for col in range(4):
        mean_col = np.mean(X[:, col])
        assert np.all(np.abs(outlier_rows[:, col] - mean_col) > 0)


def test_generate__no_graph_raises(rng):
    """Should raise ValueError if graph is not provided."""
    X = rng.normal(size=(100, 3))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="graph"):
        generator.generate(X, y=None, perturbation_nodes=["X"])


def test_generate__invalid_perturbation_raises(rng):
    """Should raise ValueError if a perturbation node is not in the graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(100, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="not found"):
        generator.generate(X, y=None, graph=graph, perturbation_nodes=["Z"])


def test_generate__cycle_raises(rng):
    """Should raise ValueError on a cyclic graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "X")])
    X = rng.normal(size=(100, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        generator.generate(X, y=None, graph=graph, perturbation_nodes=["X"])


def test_generate__y_with_labels(rng):
    """When y is provided, outlier labels should be appended."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)
    y = np.array([0, 1] * 50)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=y, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
    )

    assert Xt.shape == (103, 2)
    # np.append with string labels converts ints to strings
    assert list(yt[:100]) == ["0", "1"] * 50
    assert list(yt[100:]) == ["outliers"] * 3


def test_generate__y_none_creates_labels(rng):
    """When y is None, labels should be created."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
    )

    assert Xt.shape == (103, 2)
    assert list(yt[:100]) == ["original"] * 100
    assert list(yt[100:]) == ["outliers"] * 3


def test_generate__original_data_unchanged(rng):
    """Original rows should be identical to input."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert np.allclose(Xt[:100], X)


def test_generate__magnitude_zero(rng):
    """outlier_magnitude=0: outliers should be structurally consistent
    with the causal model (no extra perturbation beyond sampling noise)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=0.0, n_outliers=3,
    )

    assert Xt.shape == (103, 3)
    assert np.allclose(Xt[:100], X)

    # With zero magnitude and do(X), X is sampled + 0 perturbation.
    # Y and Z are descendants, computed via forward pass.
    # Fit the coefficients from the data to verify structure.
    from numpy.linalg import lstsq
    beta_xy = lstsq(X[:, :1], X[:, 1])[0][0]
    beta_yz = lstsq(X[:, 1:2], X[:, 2])[0][0]

    outliers = Xt[100:]
    # Y should be approximately beta_xy * X
    assert np.allclose(outliers[:, 1], beta_xy * outliers[:, 0], atol=0.5)
    # Z should be approximately beta_yz * Y
    assert np.allclose(outliers[:, 2], beta_yz * outliers[:, 1], atol=0.5)


def test_generate__n_outliers_zero_raises(rng):
    """Should raise ValueError if n_outliers <= 0."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="n_outliers must be positive"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=["X"],
            outlier_magnitude=3.0, n_outliers=0,
        )


def test_generate__int_nodes(rng):
    """Should work with integer node labels."""
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    coefs = {(0, 1): 2.0, (1, 2): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=[0],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)

    # Outlier rows should differ from means
    outlier_rows = Xt[100:]
    for col in range(3):
        mean_col = np.mean(X[:, col])
        assert np.all(np.abs(outlier_rows[:, col] - mean_col) > 0)


# --- New tests for perturbation_nodes (list) and sampler ---

def test_generate__multiple_perturbation_nodes(rng):
    """Multiple perturbation nodes: all should be perturbed simultaneously."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X", "Y"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)

    # Both X and Y should be far from their means
    outlier_rows = Xt[100:]
    mean_X = np.mean(X[:, 0])
    mean_Y = np.mean(X[:, 1])
    assert np.all(np.abs(outlier_rows[:, 0] - mean_X) > 0)
    assert np.all(np.abs(outlier_rows[:, 1] - mean_Y) > 0)


def test_generate__non_root_perturbation_do_style(rng):
    """do(Y) in C->Y->T: C is exogenous (sampled normally), Y perturbed,
    T propagates. C should NOT be perturbed."""
    graph = nx.DiGraph()
    graph.add_edges_from([("C", "Y"), ("Y", "T")])
    coefs = {("C", "Y"): 1.5, ("Y", "T"): 0.8}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["Y"],
        outlier_magnitude=3.0, n_outliers=5,
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)

    outlier_rows = Xt[100:]
    mean_C = np.mean(X[:, 0])
    mean_Y = np.mean(X[:, 1])
    mean_T = np.mean(X[:, 2])

    # C is exogenous (ancestor of Y), sampled from distribution — NOT perturbed
    assert np.all(np.abs(outlier_rows[:, 0] - mean_C) < 5.0)
    # Y is perturbed
    assert np.all(np.abs(outlier_rows[:, 1] - mean_Y) > 0)
    # T is a descendant, propagates
    assert np.all(np.abs(outlier_rows[:, 2] - mean_T) > 0)


def test_generate__out_of_distribution_sampler_string(rng):
    """out_of_distribution_sampler as a string should be resolved via create_out_of_distribution_sampler."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
        out_of_distribution_sampler="hypersphere",
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__out_of_distribution_sampler_instance(rng):
    """out_of_distribution_sampler as an OutOfDistributionSampler instance should be used directly."""
    from badgers.core.sampling import ZScoreSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = ZScoreSampler(scale=2.0)
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
        out_of_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__perturbation_nodes_not_list_raises(rng):
    """Should raise ValueError if perturbation_nodes is not a list."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="must be a list"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes="X",
            outlier_magnitude=3.0, n_outliers=3,
        )


def test_generate__perturbation_nodes_empty_raises(rng):
    """Should raise ValueError if perturbation_nodes is empty."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="must not be empty"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=[],
            outlier_magnitude=3.0, n_outliers=3,
        )


def test_generate__perturbation_nodes_none_raises(rng):
    """Should raise ValueError if perturbation_nodes is None."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="perturbation_nodes"):
        generator.generate(
            X, y=None, graph=graph,
            outlier_magnitude=3.0, n_outliers=3,
        )


def test_generate__invalid_out_of_distribution_sampler_type_raises(rng):
    """Should raise ValueError if out_of_distribution_sampler is not an OutOfDistributionSampler or str."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="out_of_distribution_sampler must be"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=["X"],
            outlier_magnitude=3.0, n_outliers=3,
            out_of_distribution_sampler=42,
        )


def test_generate__within_distribution_sampler_string(rng):
    """within_distribution_sampler as a string should be resolved via create_within_distribution_sampler."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
        within_distribution_sampler="uniform",
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__within_distribution_sampler_instance(rng):
    """within_distribution_sampler as a WithinDistributionSampler instance should be used directly."""
    from badgers.core.sampling import UniformSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = UniformSampler()
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
        within_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__invalid_within_distribution_sampler_type_raises(rng):
    """Should raise ValueError if within_distribution_sampler is not a WithinDistributionSampler or str."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="within_distribution_sampler must be"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=["X"],
            outlier_magnitude=3.0, n_outliers=3,
            within_distribution_sampler=42,
        )


def test_generate__both_samplers_specified(rng):
    """Both within_distribution_sampler and out_of_distribution_sampler can be specified."""
    from badgers.core.sampling import UniformSampler, ZScoreSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("C", "Y"), ("Y", "T")])
    coefs = {("C", "Y"): 1.5, ("Y", "T"): 0.8}
    X = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["Y"],
        outlier_magnitude=3.0, n_outliers=5,
        within_distribution_sampler=UniformSampler(),
        out_of_distribution_sampler=ZScoreSampler(scale=2.0),
    )

    assert Xt.shape == (105, 3)
    assert np.allclose(Xt[:100], X)


def test_generate__default_samplers(rng):
    """Default samplers should be NormalSampler (within) and ZScoreSampler (out-of)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__uniform_out_of_distribution_sampler(rng):
    """UniformOutOfDistributionSampler should work as out_of_distribution_sampler."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = UniformOutOfDistributionSampler(expansion=0.5)
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    Xt, yt = generator.generate(
        X, y=None, graph=graph, perturbation_nodes=["X"],
        outlier_magnitude=3.0, n_outliers=3,
        out_of_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__uniform_out_of_distribution_no_expansion_raises(rng):
    """UniformOutOfDistributionSampler with expansion=0 should raise ValueError."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    with pytest.raises(ValueError, match="expansion must be > 0"):
        UniformOutOfDistributionSampler(expansion=0)


def test_generate__uniform_out_of_distribution_negative_expansion_raises(rng):
    """UniformOutOfDistributionSampler with negative expansion should raise ValueError."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    with pytest.raises(ValueError, match="expansion must be > 0"):
        UniformOutOfDistributionSampler(expansion=-0.5)


def test_generate__within_distribution_sampler_rejects_out_of_distribution(rng):
    """Passing an OutOfDistributionSampler as within_distribution_sampler should raise."""
    from badgers.core.sampling import ZScoreSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="within_distribution_sampler must be"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=["X"],
            outlier_magnitude=3.0, n_outliers=3,
            within_distribution_sampler=ZScoreSampler(),
        )


def test_generate__out_of_distribution_sampler_rejects_within_distribution(rng):
    """Passing a WithinDistributionSampler as out_of_distribution_sampler should raise."""
    from badgers.core.sampling import UniformSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="out_of_distribution_sampler must be"):
        generator.generate(
            X, y=None, graph=graph, perturbation_nodes=["X"],
            outlier_magnitude=3.0, n_outliers=3,
            out_of_distribution_sampler=UniformSampler(),
        )
