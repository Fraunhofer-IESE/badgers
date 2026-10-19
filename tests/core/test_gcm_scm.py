"""Conformance tests for the ``GcmSCMDGP`` adapter (dowhy extra).

The adapter accepts an EXTERNALLY built ``gcm.StructuralCausalModel``
and delegates mechanism fitting to ``gcm.fit``; sampling/intervention
stay on badgers' generic ``DGPMixin`` path with the caller's rng.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy.gcm")

import dowhy.gcm as gcm  # noqa: E402
from dowhy.gcm.ml import PredictionModel, create_linear_regressor  # noqa: E402

from badgers.core.dgp.backends.gcm_scm import GcmSCMDGP  # noqa: E402


class SinModel(PredictionModel):
    """Deterministic f(A) = sin(A) — a NON-linear custom mechanism."""

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return np.sin(np.asarray(X)).ravel()

    def clone(self):
        return SinModel()


def _build_scm():
    """A -> B -> C; A empirical root, B = sin(A) + e, C = 0.5 B + e (linear ANM)."""
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.EmpiricalDistribution())
    cm.set_causal_mechanism("B", gcm.AdditiveNoiseModel(prediction_model=SinModel()))
    cm.set_causal_mechanism("C", gcm.AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    return cm


def _chain_data(n=500, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 3))
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = np.sin(X[:, 0]) + rng.normal(0, noise, size=n)
    X[:, 2] = 0.5 * X[:, 1] + rng.normal(0, noise, size=n)
    return X


def test_fit_and_sample_shapes_and_sin_response():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    assert dgp.is_fitted()
    sample = dgp.sample(np.random.default_rng(0), 200)
    assert sample.shape == (200, 3)
    # B = sin(A) + small noise: response must be non-linear sin
    parents = np.array([[1.0], [-2.0], [3.0]])
    np.testing.assert_allclose(dgp.response("B", parents), np.sin(parents).ravel(), atol=1e-6)


def test_sample_reconstructs_mechanism_relationships():
    data = _chain_data(n=2000)
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    Xs = dgp.sample(np.random.default_rng(1), 5000)
    # root marginal: mean/std close to training data
    assert np.mean(Xs[:, 0]) == pytest.approx(np.mean(data[:, 0]), abs=0.1)
    assert np.std(Xs[:, 0]) == pytest.approx(np.std(data[:, 0]), rel=0.1)
    # sin link: corr(Xs0, Xs1) high and regression on sin(A) tight
    coef = np.polyfit(np.sin(Xs[:, 0]), Xs[:, 1], 1)
    assert coef[0] == pytest.approx(1.0, abs=0.1)
    assert coef[1] == pytest.approx(0.0, abs=0.05)
    # linear link C ~ 0.5 B
    coef_c = np.polyfit(Xs[:, 1], Xs[:, 2], 1)
    assert coef_c[0] == pytest.approx(0.5, abs=0.1)


def test_draw_noise_is_seeded_empirical_resampling():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    n1 = dgp.draw_noise("C", np.random.default_rng(0), 5)
    n2 = dgp.draw_noise("C", np.random.default_rng(0), 5)
    np.testing.assert_array_equal(n1, n2)  # caller-rng reproducibility
    assert np.std(dgp.draw_noise("C", np.random.default_rng(3), 5000)) == pytest.approx(
        np.std(data[:, 2] - dgp.response("C", data[:, [1]])), rel=0.15
    )


def test_root_draw_reconstructs_marginal():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    draws = dgp.draw_noise("A", np.random.default_rng(0), 5000)
    assert np.std(draws) == pytest.approx(np.std(data[:, 0]), rel=0.1)
    assert np.mean(draws) == pytest.approx(np.mean(data[:, 0]), abs=0.05)


def test_default_column_mapping_follows_graph_order():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    assert dgp.column_mapping == {"A": 0, "B": 1, "C": 2}


# ---- error-path tests --------------------------------------------------


def test_unsupported_mechanism_raises_not_implemented():
    graph = nx.DiGraph([("A", "B")])
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.EmpiricalDistribution())
    cm.set_causal_mechanism("B", object())  # dummy: unsupported mechanism type
    data = _chain_data()[:, :2]
    with pytest.raises(NotImplementedError, match="'B'"):
        GcmSCMDGP(cm).fit(data)


def test_missing_mechanism_raises():
    graph = nx.DiGraph([("A", "B")])
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.EmpiricalDistribution())
    # node B deliberately left without a mechanism
    data = _chain_data()[:, :2]
    with pytest.raises(ValueError, match="'B'"):
        GcmSCMDGP(cm).fit(data)


def test_root_anm_raises():
    graph = nx.DiGraph()
    graph.add_node("A")
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    data = _chain_data()[:, [0]]
    with pytest.raises(ValueError, match="root"):
        GcmSCMDGP(cm).fit(data)


def test_empirical_root_with_parents_raises():
    graph = nx.DiGraph([("A", "B")])
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.EmpiricalDistribution())
    cm.set_causal_mechanism("B", gcm.EmpiricalDistribution())
    data = _chain_data()[:, :2]
    with pytest.raises(ValueError, match="'B'"):
        GcmSCMDGP(cm).fit(data)


def test_explicit_column_mapping_reorder():
    # 2-node chain A -> B with data columns SWAPPED: col 1 = A, col 0 = B
    graph = nx.DiGraph([("A", "B")])
    cm = gcm.StructuralCausalModel(graph)
    cm.set_causal_mechanism("A", gcm.EmpiricalDistribution())
    cm.set_causal_mechanism("B", gcm.AdditiveNoiseModel(prediction_model=SinModel()))
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    b = np.sin(a) + rng.normal(0, 0.1, size=n)
    data = np.zeros((n, 2))
    data[:, 1] = a
    data[:, 0] = b
    dgp = GcmSCMDGP(cm, column_mapping={"A": 1, "B": 0}).fit(data)
    assert dgp.column_mapping == {"A": 1, "B": 0}
    sample = dgp.sample(np.random.default_rng(1), 2000)
    # col 1 is the root A, col 0 is the sin-of-A child B
    assert np.std(sample[:, 1]) == pytest.approx(np.std(a), rel=0.1)
    coef = np.polyfit(np.sin(sample[:, 1]), sample[:, 0], 1)
    assert coef[0] == pytest.approx(1.0, abs=0.1)
    # response for B consumes the parent column per the explicit mapping
    parents = np.array([[2.0], [-1.0]])
    np.testing.assert_allclose(dgp.response("B", parents), np.sin(parents).ravel(), atol=1e-6)


def test_explicit_column_mapping_incomplete_raises():
    cm = _build_scm()
    data = _chain_data()
    with pytest.raises(ValueError, match="column_mapping"):
        GcmSCMDGP(cm, column_mapping={"A": 0, "B": 1})
    with pytest.raises(ValueError, match="column_mapping"):
        GcmSCMDGP(cm, column_mapping={"A": 0, "B": 1, "C": 1})


# ---- intervene-conformance tests (plan Task 3) -------------------------


def test_inherited_intervene_semantics():
    # do(B = 10): B pinned, A untouched, C recomputed toward 0.5 * 10
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    rows = data[:50].copy()
    before = rows.copy()
    rows = dgp.intervene(
        rows, ["B"], [np.full(50, 10.0)], rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(rows[:, 0], before[:, 0])   # A untouched
    np.testing.assert_allclose(rows[:, 1], 10.0)           # B pinned
    assert np.allclose(np.mean(rows[:, 2]), 5.0, atol=0.5)  # C ~ 0.5 * B


def test_intervene_without_values_pins_current_column():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    rows = data[:50].copy()
    before_b = rows[:, 1].copy()
    rows = dgp.intervene(rows, ["B"], None, rng=np.random.default_rng(0))
    np.testing.assert_allclose(rows[:, 1], before_b)       # B stays as-is
    assert np.all(np.isfinite(rows[:, 2]))                 # C recomputed, finite
    assert np.count_nonzero(np.abs(rows[:, 2] - data[:50, 2]) > 1e-9) > 0  # C differs somewhere


def test_sample_reproducible_with_same_seed():
    data = _chain_data()
    dgp = GcmSCMDGP(_build_scm()).fit(data)
    s1 = dgp.sample(np.random.default_rng(7), 100)
    s2 = dgp.sample(np.random.default_rng(7), 100)
    np.testing.assert_array_equal(s1, s2)
