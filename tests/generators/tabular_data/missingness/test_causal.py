"""Tests for CausalMissingnessGenerator."""

import numpy as np
import pytest

from badgers.generators.tabular_data.missingness.causal import (
    CausalMissingnessGenerator,
)


def test_generate__any_rule_propagates(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", missingness_rule="any",
    )
    # in-place: same shape
    assert Xt.shape == X.shape
    # X always missing
    assert np.all(np.isnan(Xt[:, 0]))
    # any-rule: Y and Z missing (descendants of missing X)
    assert np.all(np.isnan(Xt[:, 1]))
    assert np.all(np.isnan(Xt[:, 2]))
    # labels
    assert list(yt) == ["missingness"] * X.shape[0]


def test_generate__all_rule_recoverable_child_computed(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", missingness_rule="all",
    )
    # X missing
    assert np.all(np.isnan(Xt[:, 0]))
    # all-rule: Y has only parent X (missing) -> Y missing
    assert np.all(np.isnan(Xt[:, 1]))
    # Z has parent Y (missing) -> Z missing
    assert np.all(np.isnan(Xt[:, 2]))


def test_generate__all_rule_diamond_recoverable(chain_data):
    # A -> B, A -> C, B -> D, C -> D ; perturb A, rule=all.
    # D has parents B and C (both missing) -> D missing.
    import networkx as nx
    from tests.generators.tabular_data.conftest import (
        _generate_linear_data,
    )
    rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    X, column_mapping = _generate_linear_data(graph, 100, rng)
    gen = CausalMissingnessGenerator(random_generator=rng)
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["A"],
        mechanism="always", missingness_rule="all",
    )
    assert np.all(np.isnan(Xt[:, column_mapping["A"]]))
    assert np.all(np.isnan(Xt[:, column_mapping["D"]]))


def test_generate__all_rule_recoverable_child_computed_from_observed_parent():
    # X -> Y, Z -> Y (fork into Y). Perturb X, rule=all.
    # Y has parents X (missing) and Z (observed) -> Y is RECOVERABLE:
    # computed from the observed parent Z via the structural equation.
    import networkx as nx
    from tests.generators.tabular_data.conftest import (
        _generate_linear_data,
    )
    rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Z", "Y")])
    coefs = {("X", "Y"): 1.0, ("Z", "Y"): 1.0}
    X, column_mapping = _generate_linear_data(
        graph, 100, rng, coefficients=coefs
    )
    gen = CausalMissingnessGenerator(random_generator=rng)
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="always", missingness_rule="all",
    )
    # X missing
    assert np.all(np.isnan(Xt[:, column_mapping["X"]]))
    # Z observed (background, not a perturbation node)
    assert np.all(np.isfinite(Xt[:, column_mapping["Z"]]))
    # Y recoverable: computed from observed parent Z -> finite
    assert np.all(np.isfinite(Xt[:, column_mapping["Y"]]))
    # Y should track Z (beta_ZY ~ 1.0): correlation is high
    y_vals = Xt[:, column_mapping["Y"]]
    z_vals = Xt[:, column_mapping["Z"]]
    assert np.corrcoef(y_vals, z_vals)[0, 1] > 0.5


def test_generate__mcar_mask_independent_of_values(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="random", mechanism_rate=0.5, missingness_rule="any",
    )
    mask = np.isnan(Xt[:, 0])
    # ~50% missing
    assert 0.4 < mask.mean() < 0.6
    # Missingness independent of X's (latent) value: the observed (non-missing)
    # X values should have a mean near the original mean.
    observed_x = Xt[:, 0][~mask]
    assert abs(np.mean(observed_x) - np.mean(X_orig[:, 0])) < 0.5


def test_generate__self_mnar_mask_depends_on_own_value(chain_data):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["X"],
        mechanism="self", mechanism_rate=0.3, missingness_rule="any",
    )
    mask = np.isnan(Xt[:, 0])
    # MNAR: high X values are more likely missing.
    # Check that the observed (non-missing) X mean is shifted DOWN (high values
    # removed).
    observed_x = Xt[:, 0][~mask]
    assert np.mean(observed_x) < np.mean(X_orig[:, 0]) - 0.2


def test_generate__observed_mar_depends_on_cause(chain_data):
    X, graph, column_mapping = chain_data
    # cause = X (col 0, an ancestor of Y, fully observed); perturb Y.
    # (The cause must not be a descendant of the perturbation node, or the
    # cause column itself would be NaN-propagated.)
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping)
    Xt, yt = gen.generate(
        X, y=None, perturbation_nodes=["Y"],
        mechanism="observed", mechanism_cause="X", mechanism_rate=0.5,
        missingness_rule="any",
    )
    mask = np.isnan(Xt[:, 1])
    # Missingness should correlate with X (the cause).
    x = Xt[:, 0]
    miss_high_x = mask[x >= np.quantile(x, 0.75)].mean()
    miss_low_x = mask[x <= np.quantile(x, 0.25)].mean()
    assert miss_high_x > miss_low_x


def test_generate__validation_errors(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    # not fitted yet
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.generate(X, y=None, perturbation_nodes=["X"])
    # unknown perturbation node
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="not found in graph"):
        gen.generate(X, y=None, perturbation_nodes=["NOPE"])


def test_generate__mar_requires_cause(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    gen.fit(X, graph, column_mapping)
    with pytest.raises(ValueError, match="mechanism_cause"):
        gen.generate(
            X, y=None, perturbation_nodes=["X"],
            mechanism="observed", mechanism_cause=None,
        )

