"""Conformance tests for the ``PyMCSCMDGP`` adapter (pymc extra).

The user writes ONE generative ``pm.Model`` — structural nodes as
observed RVs whose location/shape parameters reference parent RVs, free
RVs as Bayesian parameters — and hands it to badgers. Mechanism fitting
is PyMC's job (one joint ``pm.sample``); sampling and interventions run
on badgers' generic ``DGPMixin`` path with the caller's rng and ONE
pinned posterior draw per cascade.
"""

import numpy as np
import pytest

pytest.importorskip("pymc")

import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402

from badgers.core.dgp.backends import PyMCSCMDGP  # noqa: E402


def _sin_chain_model(n=300, seed=0):
    """A -> B -> C in PyMC idiom: B = sin(A) + e, C = 0.5 B + e."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 3))
    data[:, 0] = rng.normal(size=n)
    data[:, 1] = np.sin(data[:, 0]) + rng.normal(0, 0.2, size=n)
    data[:, 2] = 0.5 * data[:, 1] + rng.normal(0, 0.2, size=n)
    with pm.Model() as scm:
        mu_A = pm.Normal("mu_A", 0.0, 2.0)
        sigma_A = pm.HalfNormal("sigma_A", 1.0)
        A = pm.Normal("A", mu=mu_A, sigma=sigma_A, observed=data[:, 0])
        sigma_B = pm.HalfNormal("sigma_B", 1.0)
        B = pm.Normal("B", mu=pt.sin(A), sigma=sigma_B, observed=data[:, 1])
        beta_C = pm.Normal("beta_C", 0.0, 5.0)
        sigma_C = pm.HalfNormal("sigma_C", 1.0)
        C = pm.Normal("C", mu=beta_C * B, sigma=sigma_C, observed=data[:, 2])
    return scm, data, {"A": 0, "B": 1, "C": 2}


def test_graph_is_derived_from_param_expressions():
    scm, data, mapping = _sin_chain_model()
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30)
    assert set(dgp.graph.nodes) == {"A", "B", "C"}
    assert list(dgp.graph.predecessors("B")) == ["A"]
    assert list(dgp.graph.predecessors("C")) == ["B"]
    assert not list(dgp.graph.predecessors("A"))


def test_fit_and_sample_shapes_and_sin_response():
    scm, data, mapping = _sin_chain_model()
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    assert dgp.is_fitted()
    Xs = dgp.sample(np.random.default_rng(0), 100)
    assert Xs.shape == (100, 3)
    # response is the pinned-draw location: sin(A), NOT a linear fit
    parents = np.array([[1.0], [-2.0], [3.0]])
    np.testing.assert_allclose(
        dgp.response("B", parents), np.sin(parents).ravel(), atol=1e-6
    )


def test_sample_reconstructs_mechanism_relationships():
    scm, data, mapping = _sin_chain_model(n=800)
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    Xs = dgp.sample(np.random.default_rng(1), 8000)
    coef_b = np.polyfit(np.sin(Xs[:, 0]), Xs[:, 1], 1)
    assert coef_b[0] == pytest.approx(1.0, abs=0.1)
    coef_c = np.polyfit(Xs[:, 1], Xs[:, 2], 1)
    assert coef_c[0] == pytest.approx(0.5, abs=0.1)


def test_full_posterior_is_kept_not_collapsed():
    scm, data, mapping = _sin_chain_model(n=500)
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    assert dgp.n_draws == 30
    # two cascades with different rngs differ (posterior draws differ)
    s1 = dgp.sample(np.random.default_rng(0), 50)
    s2 = dgp.sample(np.random.default_rng(1), 50)
    assert not np.allclose(s1, s2)
    # same rng: identical (reproducibility contract)
    s3 = dgp.sample(np.random.default_rng(0), 50)
    np.testing.assert_array_equal(s1, s3)


def test_root_location_and_scale_are_inferred():
    scm, data, mapping = _sin_chain_model(n=500)
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    Xs = dgp.sample(np.random.default_rng(2), 5000)
    # root marginal reconstructed from pinned mu_A + Normal(0, sigma_A)
    assert np.mean(Xs[:, 0]) == pytest.approx(np.mean(data[:, 0]), abs=0.2)
    assert np.std(Xs[:, 0]) == pytest.approx(np.std(data[:, 0]), rel=0.25)
