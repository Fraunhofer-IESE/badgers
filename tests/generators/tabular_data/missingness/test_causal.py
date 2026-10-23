"""Tests for CausalMissingnessGenerator (observation layer, chooser-only).

Semantics: NaN is written ONLY at the fired cells of the perturbation
nodes. No descendant cascade -- missingness is a property of the
measurement, not of the world.
"""

import networkx as nx
import numpy as np
import pytest

from badgers.core.dgp import DGPMixin
from badgers.core.selectors import (
    AlwaysSelector,
    BernoulliSelector,
    TopQuantileSelector,
)
from badgers.generators.tabular_data.missingness.causal import (
    CausalMissingnessGenerator,
)


class _FakeDGP(DGPMixin):
    """Minimal DGP shell: graph + column mapping only (no mechanisms)."""

    def __init__(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("X", "Y"), ("Y", "Z")])
        self.graph = graph
        self.column_mapping = {"X": 0, "Y": 1, "Z": 2}


def test_generate__always_fires_everywhere(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        selector=AlwaysSelector([0]),
    )
    # in-place: same shape
    assert Xt.shape == X.shape
    # X always missing
    assert np.all(np.isnan(Xt[:, 0]))
    # NO cascade: descendants keep their observed values
    assert np.all(np.isfinite(Xt[:, 1]))
    assert np.all(np.isfinite(Xt[:, 2]))
    # labels
    labels = gen.get_missing_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__selector_receives_rows_and_labels_match_result(
    chain_data, make_gcm_dgp
):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    selector_rows = []

    def selector(rows, targets=None):
        selector_rows.append(rows)
        return {col: np.arange(rows.shape[0]) % 2 == 0 for col in targets}

    Xt = gen.generate(X, perturbation_nodes=["X"], selector=selector)

    assert Xt is X
    assert selector_rows == [X]
    np.testing.assert_array_equal(
        gen.get_missing_labels(), np.arange(X.shape[0]) % 2 == 0
    )
    assert gen.get_missing_labels().shape[0] == Xt.shape[0]


def test_fire_masks__selector_receives_only_working_rows():
    rows = np.arange(6).reshape(3, 2)
    gen = CausalMissingnessGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_FakeDGP(),
    )
    selector_calls = []

    def selector(working_rows, targets=None):
        selector_calls.append(working_rows)
        return {col: np.array([True, False, True]) for col in targets}

    masks = gen._fire_masks(
        rows, ["X"], selector=selector
    )

    assert selector_calls == [rows]
    np.testing.assert_array_equal(masks["X"], [True, False, True])


def test_generate__mcar_mask_independent_of_values(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        selector=BernoulliSelector([0], rate=0.5,
                                   rng=gen.random_generator),
    )
    mask = np.isnan(Xt[:, 0])
    # ~50% missing
    assert 0.4 < mask.mean() < 0.6
    # Missingness independent of X's (latent) value: the observed (non-missing)
    # X values should have a mean near the original mean.
    observed_x = Xt[:, 0][~mask]
    assert abs(np.mean(observed_x) - np.mean(X_orig[:, 0])) < 0.5
    # Only the perturbed column is touched.
    np.testing.assert_array_equal(np.isnan(Xt[:, 1]), np.zeros(X.shape[0], bool))
    np.testing.assert_array_equal(np.isnan(Xt[:, 2]), np.zeros(X.shape[0], bool))


def test_generate__self_mnar_mask_depends_on_own_value(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X"],
        selector=TopQuantileSelector([0], column=0, quantile=0.7),
    )
    mask = np.isnan(Xt[:, 0])
    # MNAR: high X values are more likely missing.
    # Check that the observed (non-missing) X mean is shifted DOWN (high values
    # removed).
    observed_x = Xt[:, 0][~mask]
    assert np.mean(observed_x) < np.mean(X_orig[:, 0]) - 0.2
    # Only the perturbed column is touched.
    assert np.all(np.isfinite(Xt[:, 1]))
    assert np.all(np.isfinite(Xt[:, 2]))


def test_generate__observed_mar_depends_on_cause(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    # cause = X (col 0, an ancestor of Y, fully observed); perturb Y.
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(1))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["Y"],
        selector=TopQuantileSelector(
            [1], column=1, quantile=0.5, driver=0
        ),
    )
    mask = np.isnan(Xt[:, 1])
    # Missingness should correlate with X (the cause).
    x = Xt[:, 0]
    miss_high_x = mask[x >= np.quantile(x, 0.75)].mean()
    miss_low_x = mask[x <= np.quantile(x, 0.25)].mean()
    assert miss_high_x > miss_low_x
    # Only the perturbed column is touched.
    assert np.all(np.isfinite(Xt[:, 0]))
    assert np.all(np.isfinite(Xt[:, 2]))


def test_generate__multi_node_only_fired_cells(chain_data, make_gcm_dgp):
    # Perturb TWO nodes: each column gets NaN only where IT fired.
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(3))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xt = gen.generate(
        X, perturbation_nodes=["X", "Z"],
        selector=BernoulliSelector(
            [0, 2], rate=0.3, rng=gen.random_generator
        ),
    )
    mx = np.isnan(Xt[:, 0])
    mz = np.isnan(Xt[:, 2])
    assert 0.1 < mx.mean() < 0.5
    assert 0.1 < mz.mean() < 0.5
    # Y untouched.
    assert np.all(np.isfinite(Xt[:, 1]))


def test_generate__validation_errors(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator()
    # not fitted yet
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(X, perturbation_nodes=["X"])
    # unknown perturbation node
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    with pytest.raises(ValueError, match="not found in graph"):
        gen.generate(X, perturbation_nodes=["NOPE"])


def test_generate__labels_before_generate_raises(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="only available after calling"):
        gen.get_missing_labels()


def test_generate__reproducible_with_explicit_rng(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    outs = []
    for _ in range(2):
        gen = CausalMissingnessGenerator(
            random_generator=np.random.default_rng(11)
        )
        gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
        Xt = gen.generate(
            X.copy(), perturbation_nodes=["X"],
            selector=BernoulliSelector([0], rate=0.4,
                                       rng=gen.random_generator),
        )
        outs.append(Xt)
    np.testing.assert_array_equal(outs[0], outs[1])


def test_generate__all_nan_input_edge(chain_data, make_gcm_dgp):
    # A fully NaN input column: firing writes NaN again (idempotent), no
    # crash, no other column touched.
    X, graph, column_mapping = chain_data
    gen = CausalMissingnessGenerator(random_generator=np.random.default_rng(2))
    gen.use_dgp(make_gcm_dgp(graph, column_mapping).fit(X))
    Xn = X.copy()
    Xn[:, 1] = np.nan
    Xt = gen.generate(Xn, perturbation_nodes=["Y"],
                         selector=AlwaysSelector([1]))
    assert np.all(np.isnan(Xt[:, 1]))
    np.testing.assert_array_equal(Xt[:, 0], Xn[:, 0])
    np.testing.assert_array_equal(Xt[:, 2], Xn[:, 2])
