from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp.backends.gcm_scm import GcmSCMDGP
from badgers.core.selectors import (
    AlwaysSelector,
    CountSelector,
    TopQuantileSelector,
)
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


def _world(graph, X, column_mapping=None):
    """Build a fitted ``GcmSCMDGP`` world for a test graph (linear ANM SCM
    via the shared conftest helper)."""
    pytest.importorskip("dowhy.gcm")
    from tests.conftest import make_gcm_scm

    if column_mapping is None:
        column_mapping = {n: i for i, n in enumerate(sorted(graph.nodes))}
    return GcmSCMDGP(make_gcm_scm(graph), column_mapping).fit(X)


def _dgp_shell(graph, column_mapping=None):
    """An UNFITTED ``GcmSCMDGP`` over a bare graph carrier: validation-error
    tests never fit it, so no mechanisms are needed. An explicitly given
    column_mapping is attached AFTER construction (GcmSCMDGP validates a
    mapping it is handed, which would preempt the generator-level mapping
    validation these tests pin)."""
    dgp = GcmSCMDGP(SimpleNamespace(graph=graph))
    if column_mapping is not None:
        dgp.column_mapping = column_mapping
    return dgp


def test_generate__chain_x_to_y(rng):
    """Perturb X in X->Y->Z: Y and Z should shift proportionally."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=5, rng=rng),
    )

    # Original data should be unchanged
    assert np.allclose(Xt[:100], X)

    # Outlier rows appended
    assert Xt.shape == (105, 3)

    # yt labels
    labels = generator.get_outliers_labels()
    assert not labels[:100].any()
    assert labels[100:].all()
    mean = np.mean(X, axis=0)
    for i in range(100, 105):
        assert not np.allclose(Xt[i], mean)


def test_generate__chain_y_do_intervention(rng):
    """Perturb Y in X->Y->Z with do()-style: X unchanged, Y perturbed, Z propagates."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([1], count=3, rng=rng),
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
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=5, rng=rng),
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
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=5, rng=rng),
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
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=5, rng=rng),
    )

    assert Xt.shape == (105, 4)
    assert np.allclose(Xt[:100], X)

    # All outlier columns should differ from their means
    outlier_rows = Xt[100:]
    for col in range(4):
        mean_col = np.mean(X[:, col])
        assert np.all(np.abs(outlier_rows[:, col] - mean_col) > 0)


def test_generate__no_dgp_raises(rng):
    """Should raise ValueError if no DGP is configured."""
    X = rng.normal(size=(100, 3))
    generator = CausalOutlierPropagationGenerator(random_generator=rng)
    with pytest.raises(ValueError, match="DGP configured"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__cycle_raises(rng):
    """Should raise ValueError on a cyclic graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "X")])
    X = rng.normal(size=(100, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1})
    )
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__labels_without_y_output(rng):
    """Causal outlier labels are available separately from generated features."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
    )

    assert Xt.shape == (103, 2)
    labels = generator.get_outliers_labels()
    assert not labels[:100].any()
    assert labels[100:].all()


def test_generate__original_data_unchanged(rng):
    """Original rows should be identical to input."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=5, rng=rng),
    )

    assert np.allclose(Xt[:100], X)


def test_generate__magnitude_zero(rng):
    """Default ZScoreSampler: outliers should be structurally consistent with the causal model."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
    )

    assert Xt.shape == (103, 3)
    assert np.allclose(Xt[:100], X)

    # With do(X), X is perturbed. Y and Z are descendants, computed via forward pass.
    # Fit the coefficients from the data to verify structure.
    from numpy.linalg import lstsq
    beta_xy = lstsq(X[:, :1], X[:, 1])[0][0]
    beta_yz = lstsq(X[:, 1:2], X[:, 2])[0][0]

    outliers = Xt[100:]
    # Y should be approximately beta_xy * X
    assert np.allclose(outliers[:, 1], beta_xy * outliers[:, 0], atol=0.5)
    # Z should be approximately beta_yz * Y
    assert np.allclose(outliers[:, 2], beta_yz * outliers[:, 1], atol=0.5)


# --- New tests for the selector-owned targets and samplers ---

def test_generate__multiple_perturbation_nodes(rng):
    """Multiple targets via one selector: all should be perturbed simultaneously."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    coefs = {("X", "Y"): 2.0, ("Y", "Z"): 0.5}
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0, 1], count=5, rng=rng),
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
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng, coefficients=coefs)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([1], count=5, rng=rng),
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
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
        out_of_distribution_sampler="hypersphere",
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__out_of_distribution_sampler_instance(rng):
    """out_of_distribution_sampler as an OutOfDistributionSampler instance should be used directly."""
    from badgers.core.sampling import ZScoreSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = ZScoreSampler(scale=2.0)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
        out_of_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__selector_none_raises(rng):
    """Should raise ValueError if no selector is given."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1})
    )
    with pytest.raises(ValueError, match="selector is required"):
        generator.generate(X, None)


def test_generate__per_cell_pinning_partial_selection(rng):
    """Rows selected only on one target: that cell is OOD, the other target
    is recomputed through the SCM surgery (per-cell pinning); labels are
    True for every appended row (no pure-bootstrap rows exist)."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    coefs = {("X", "Y"): 0.0}
    X, column_mapping = _generate_linear_data(
        graph, n_samples=100, rng=rng, coefficients=coefs)
    sel_x = np.zeros(100, dtype=bool)
    sel_x[[0, 1, 2]] = True
    sel_y = np.zeros(100, dtype=bool)
    sel_y[[0, 2, 3]] = True
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(X, lambda X_, targets=None: {0: sel_x, 1: sel_y})
    assert Xt.shape == (104, 2)
    labels = generator.get_outliers_labels()
    np.testing.assert_array_equal(labels[:100], np.zeros(100, dtype=bool))
    assert labels[100:].all()  # every appended row selected somewhere
    stds = X.std(axis=0)
    # appended rows are the seeds in increasing X-row order: 0, 1, 2, 3
    # row 101 selects only on X -> its Y cell is recomputed, not pinned
    only_x = 101
    assert abs(Xt[only_x, 0] - X[:, 0].mean()) > 3.0 * stds[0]
    pred_y = float(
        generator.dgp.response("Y", Xt[only_x:only_x + 1, [0]])[0])
    assert abs(Xt[only_x, 1] - pred_y) < 0.5
    assert abs(Xt[only_x, 1] - X[:, 1].mean()) < 3.0 * stds[1]


def test_generate__invalid_out_of_distribution_sampler_type_raises(rng):
    """Should raise ValueError if out_of_distribution_sampler is not an OutOfDistributionSampler or str."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1})
    )
    with pytest.raises(ValueError, match="out_of_distribution_sampler must be"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
            out_of_distribution_sampler=42,
        )


def test_generate__default_samplers(rng):
    """The default out-of-distribution sampler should be ZScoreSampler."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(X, CountSelector([0], count=3, rng=rng))
    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__uniform_out_of_distribution_sampler(rng):
    """UniformOutOfDistributionSampler should work as out_of_distribution_sampler."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = UniformOutOfDistributionSampler(expansion=0.5)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
        out_of_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__uniform_out_of_distribution_zero_expansion_allowed(rng):
    """UniformOutOfDistributionSampler accepts expansion=0 (exact bounding box)."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)

    sampler = UniformOutOfDistributionSampler(expansion=0)
    generator = CausalOutlierPropagationGenerator(random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(
        X,
        CountSelector([0], count=3, rng=rng),
        out_of_distribution_sampler=sampler,
    )

    assert Xt.shape == (103, 2)
    assert np.allclose(Xt[:100], X)


def test_generate__uniform_out_of_distribution_negative_expansion_raises():
    """UniformOutOfDistributionSampler with negative expansion should raise ValueError."""
    from badgers.core.sampling import UniformOutOfDistributionSampler

    with pytest.raises(ValueError, match="expansion must be >= 0"):
        UniformOutOfDistributionSampler(expansion=-0.5)


def test_generate__out_of_distribution_sampler_rejects_within_distribution(rng):
    """Passing a WithinDistributionSampler as out_of_distribution_sampler should raise."""
    from badgers.core.sampling import UniformSampler

    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1})
    )
    with pytest.raises(ValueError, match="out_of_distribution_sampler must be"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
            out_of_distribution_sampler=UniformSampler(),
        )


def test_generate__column_mapping_not_dict_raises(rng):
    """Should raise ValueError if the DGP carries a non-dict column_mapping."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, [("X", 0), ("Y", 1)])
    )
    with pytest.raises(ValueError, match="column_mapping must be a dict"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_missing_node_raises(rng):
    """Should raise ValueError if a graph node is missing from column_mapping."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = rng.normal(size=(10, 3))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1})
    )
    with pytest.raises(ValueError, match="column_mapping missing node"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_extra_node_raises(rng):
    """Should raise ValueError if column_mapping has nodes not in the graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1, "Z": 2})
    )
    with pytest.raises(ValueError, match="column_mapping has unknown node"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_non_int_value_raises(rng):
    """Should raise ValueError if a column_mapping value is not an int."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": "col1"})
    )
    with pytest.raises(ValueError, match="column_mapping values must be int"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_out_of_range_raises(rng):
    """Should raise ValueError if a column index is out of range."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 5})
    )
    with pytest.raises(ValueError, match="out of range"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_duplicate_index_raises(rng):
    """Should raise ValueError if column_mapping has duplicate indices."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Y", "Z")])
    X = rng.normal(size=(10, 3))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {"X": 0, "Y": 1, "Z": 1})
    )
    with pytest.raises(ValueError, match="duplicate column index"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__column_mapping_int_nodes_raises(rng):
    """Should raise ValueError if graph has integer nodes."""
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1)])
    X = rng.normal(size=(10, 2))
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_dgp_shell(graph, {0: 0, 1: 1})
    )
    with pytest.raises(ValueError, match="graph nodes must be strings"):
        generator.generate(
            X,
            selector=AlwaysSelector([0]),
        )


def test_generate__count_zero_generates_nothing(rng):
    """An empty selection seeds nothing: X is returned unchanged."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(X, CountSelector([0], count=0, rng=rng))
    np.testing.assert_array_equal(Xt, X)
    labels = generator.get_outliers_labels()
    assert not labels.any()


def test_generate__seeds_are_copies_of_selected_rows(rng):
    """Unselected non-descendant cells keep the seed row's own values."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B")])
    coefs = {("A", "B"): 0.0}
    X, column_mapping = _generate_linear_data(
        graph, n_samples=100, rng=rng, coefficients=coefs)
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(X, CountSelector([0], count=5, rng=rng))
    # A is the selected root; B recomputes (descendant); the appended rows
    # are 5 distinct real rows' copies
    assert Xt.shape == (105, 2)
    assert np.allclose(Xt[:100], X)
    # the 5 appended rows must be distinct rows of X (without replacement)
    assert np.unique(Xt[100:, 0]).size == 5


def test_generate__value_driven_selector_reads_X(rng):
    """A value-driven selector seeds the rows it selects on X."""
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y")])
    X, column_mapping = _generate_linear_data(graph, n_samples=100, rng=rng)
    generator = CausalOutlierPropagationGenerator(
        random_generator=rng, dgp=_world(graph, X, column_mapping))
    Xt = generator.generate(X, TopQuantileSelector([0], column=0, quantile=0.9))
    n_seeds = int((X[:, 0] >= np.quantile(X[:, 0], 0.9)).sum())
    assert Xt.shape == (100 + n_seeds, 2)
    labels = generator.get_outliers_labels()
    assert labels[100:].all()
    # the pinned cells are beyond 3 sigma of the column
    stds = X.std(axis=0)
    assert np.all(np.abs(Xt[100:, 0] - X[:, 0].mean()) > 3.0 * stds[0])


