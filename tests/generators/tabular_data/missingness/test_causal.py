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


def test_generate__observation_graph_mcar_dropout(chain_data):
    import networkx as nx
    X, graph, column_mapping = chain_data
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    obs.add_edges_from([("X", "Y"), ("Y", "Z")])  # derivation edges (mirror
    # the structural graph); without them Y and Z would stay observed and
    # the assertions below would be unsatisfiable. [controller pre-fix]
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    Xt, yt = gen.generate(X, y=None, mechanism="always", mechanism_rate=1.0)
    # isolated R_X fires on every row -> X missing (dropout preserves values)
    assert np.all(np.isnan(Xt[:, 0]))
    # "any" derivation default: Y and Z missing
    assert np.all(np.isnan(Xt[:, 1]))
    assert np.all(np.isnan(Xt[:, 2]))
    assert list(yt) == ["missingness"] * X.shape[0]
    # dropout: observed values equal the originals
    observed = ~np.isnan(Xt[:, 0])
    np.testing.assert_allclose(Xt[observed, 0], X_orig[observed, 0])


def test_generate__observation_graph_mar_driver(chain_data):
    import networkx as nx
    X, graph, column_mapping = chain_data
    # trigger: Y's reading drives R_X's failure; wiring mirrors the structure
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    obs.add_edges_from([("Y", "R_X"), ("X", "Y"), ("Y", "Z")])
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    X_orig = X.copy()  # capture BEFORE generate: the driver quantiles must be
    # computed on the PRE-propagation Y column (the implementation computes
    # them during the indicator pass, before any NaN is written downstream);
    # after generate, Y carries NaNs and np.quantile would return NaN.
    # [controller pre-fix]
    Xt, yt = gen.generate(X, y=None, mechanism="always", mechanism_rate=0.5)
    mask_x = np.isnan(Xt[:, 0])
    y_orig = X_orig[:, 1]
    miss_high_y = mask_x[y_orig >= np.quantile(y_orig, 0.75)].mean()
    miss_low_y = mask_x[y_orig <= np.quantile(y_orig, 0.25)].mean()
    assert miss_high_y > miss_low_y


def test_generate__observation_graph_mnar_self_driven(chain_data):
    import networkx as nx
    X, graph, column_mapping = chain_data
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    obs.add_edges_from([("X", "R_X"), ("X", "Y"), ("Y", "Z")])
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    X_orig = X.copy()  # capture BEFORE generate (in-place generator)
    Xt, yt = gen.generate(X, y=None, mechanism="always", mechanism_rate=0.3)
    # MNAR: high X values are removed -> observed mean shifts down
    observed_x = Xt[:, 0][~np.isnan(Xt[:, 0])]
    assert np.mean(observed_x) < np.mean(X_orig[:, 0]) - 0.2


def test_generate__observation_graph_rejects_perturbation_nodes(chain_data):
    import networkx as nx
    X, graph, column_mapping = chain_data
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    with pytest.raises(ValueError, match="perturbation_nodes"):
        gen.generate(
            X, y=None, perturbation_nodes=["X"],
            mechanism="always", mechanism_rate=0.5,
        )


def test_validate__observation_graph_non_leaf_indicator(chain_data):
    import networkx as nx
    X, graph, column_mapping = chain_data
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    obs.add_edges_from([("X", "Y"), ("Y", "Z"), ("X", "R_X"), ("R_X", "Z")])
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    with pytest.raises(ValueError, match="must be a leaf"):
        gen.generate(X, y=None, mechanism="always", mechanism_rate=0.5)


def test_generate__observation_graph_rule_all_recovers():
    # Fork X -> Y <- Z with DERIVATION edges listed in reverse structural
    # order: obs sources for Y are [Z, X] but the structural predecessors
    # are [X, Z]. The recoverable recompute must call the mechanism in
    # STRUCTURAL order, so the prediction is beta_XY * 0 + beta_ZY * z
    # = 0.5 * z (a wrong-order call would predict 2.0 * z and fail).
    # Also exercises the needs_dgp pre-fit guard (no DGP is passed to fit).
    # [controller pre-fix: this path is entirely absent from the plan]
    import networkx as nx
    from tests.generators.tabular_data.conftest import (
        _generate_linear_data,
    )
    rng = np.random.default_rng(0)
    graph = nx.DiGraph()
    graph.add_edges_from([("X", "Y"), ("Z", "Y")])
    coefs = {("X", "Y"): 2.0, ("Z", "Y"): 0.5}
    X, column_mapping = _generate_linear_data(
        graph, 200, rng, coefficients=coefs
    )
    obs = nx.DiGraph()
    obs.add_nodes_from(["X", "Y", "Z", "R_X"])
    obs.add_edges_from([("Z", "Y"), ("X", "Y")])  # reversed vs structural
    obs.nodes["Y"]["derivation_rule"] = "all"
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.fit(X, graph, column_mapping, observation_graph=obs)
    Xt, yt = gen.generate(X, y=None, mechanism_rate=0.5)
    mask_x = np.isnan(Xt[:, column_mapping["X"]])
    # MCAR R_X fires on ~50% of the rows
    assert 0.2 < mask_x.mean() < 0.8
    # derivation_rule="all": Y is missing only if ALL sources are missing;
    # Z is never missing -> Y never NaN (and Z, fed by Y, never NaN either).
    assert not np.isnan(Xt[:, column_mapping["Y"]]).any()
    assert not np.isnan(Xt[:, column_mapping["Z"]]).any()
    # Recovered Y on X-missing rows: predicted from the observed sources
    # (missing X zeroed) in STRUCTURAL order, plus fresh noise (std ~ 0.1).
    y_miss = Xt[mask_x, column_mapping["Y"]]
    z_miss = Xt[mask_x, column_mapping["Z"]]
    predicted = gen.dgp.response("Y", np.column_stack([
        np.zeros(int(mask_x.sum())), z_miss,
    ]))
    np.testing.assert_allclose(y_miss, predicted, atol=0.5)
    # Loose sanity: recovered Y still tracks Z (beta_ZY ~ 0.5).
    assert np.corrcoef(y_miss, z_miss)[0, 1] > 0.8
