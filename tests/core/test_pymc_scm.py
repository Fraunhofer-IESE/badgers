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


def _mixed_model(n=400, seed=0):
    """A -> B (continuous) -> D (Bernoulli on logit w*B) in one pm.Model."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 3))
    data[:, 0] = rng.normal(size=n)
    data[:, 1] = np.sin(data[:, 0]) + rng.normal(0, 0.2, size=n)
    p = 1.0 / (1.0 + np.exp(-(2.0 * data[:, 1])))
    data[:, 2] = (rng.random(n) < p).astype(float)
    with pm.Model() as scm:
        A = pm.Normal("A", mu=0.0, sigma=1.0, observed=data[:, 0])
        sigma_B = pm.HalfNormal("sigma_B", 1.0)
        B = pm.Normal("B", mu=pt.sin(A), sigma=sigma_B, observed=data[:, 1])
        w = pm.Normal("w", 0.0, 5.0)
        D = pm.Bernoulli("D", logit_p=w * B, observed=data[:, 2])
    return scm, data, {"A": 0, "B": 1, "D": 2}


def test_discrete_node_marginal_and_samples():
    scm, data, mapping = _mixed_model()
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    Xs = dgp.sample(np.random.default_rng(0), 5000)
    assert set(np.unique(Xs[:, 2])) <= {0.0, 1.0}
    # marginal rate close to training rate
    assert np.mean(Xs[:, 2]) == pytest.approx(np.mean(data[:, 2]), abs=0.08)
    # discrete node response contributes zeros; class comes from noise
    np.testing.assert_allclose(dgp.response("D", data[:5, [1]]), 0.0)


def test_poisson_and_binomial_families():
    rng = np.random.default_rng(3)
    n = 400
    Xobs = rng.normal(size=n)
    Y = rng.poisson(np.exp(Xobs))
    Z = rng.binomial(4, 0.3, size=n)
    with pm.Model() as scm:
        X = pm.Normal("X", mu=0.0, sigma=1.0, observed=Xobs)
        lam = pm.HalfNormal("lam", 1.0)
        P = pm.Poisson("P", mu=pt.exp(X) * 0.5, observed=Y)
        Nt = pm.Binomial("Nt", n=4, p=0.3, observed=Z)
    dgp = PyMCSCMDGP(scm, {"X": 0, "P": 1, "Nt": 2}, n_draws=30).fit(
        np.column_stack([Xobs, Y, Z])
    )
    Xs = dgp.sample(np.random.default_rng(1), 3000)
    assert np.all(Xs[:, 1] >= 0)
    assert np.all(Xs[:, 1] == np.floor(Xs[:, 1]))
    assert set(np.unique(Xs[:, 2])) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert np.mean(Xs[:, 2]) == pytest.approx(1.2, abs=0.3)


def test_intervene_continuous_chain():
    # do(B = 10): B pinned, A untouched, C recomputed toward 0.5 * 10
    scm, data, mapping = _sin_chain_model(n=400)
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    rows = data[:50].copy()
    before = rows.copy()
    rows = dgp.intervene(
        rows, ["B"], [np.full(50, 10.0)], rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(rows[:, 0], before[:, 0])
    np.testing.assert_allclose(rows[:, 1], 10.0)
    assert np.mean(rows[:, 2]) == pytest.approx(5.0, abs=0.6)


def test_intervene_discrete_descendant():
    # do(B = 3): D recomputed with high logit (w * 3) -> rate near 1
    scm, data, mapping = _mixed_model(n=500)
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    rows = data[:200].copy()
    rows = dgp.intervene(
        rows, ["B"], [np.full(200, 3.0)], rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(rows[:, 1], 3.0)
    assert np.mean(rows[:, 2]) > 0.9  # logit w*3 with w ~ 2 -> p ~ 0.998


def test_intervene_without_values_pins_current_column():
    scm, data, mapping = _sin_chain_model()
    dgp = PyMCSCMDGP(scm, mapping, n_draws=30).fit(data)
    rows = data[:50].copy()
    before_b = rows[:, 1].copy()
    rows = dgp.intervene(rows, ["B"], None, rng=np.random.default_rng(0))
    np.testing.assert_allclose(rows[:, 1], before_b)
    # C recomputed (finite, differs somewhere)
    assert np.all(np.isfinite(rows[:, 2]))
    assert np.count_nonzero(np.abs(rows[:, 2] - data[:50, 2]) > 1e-9) > 0


def test_error_paths():
    # structural node missing from the model
    with pm.Model() as scm:
        A = pm.Normal("A", mu=0.0, sigma=1.0, observed=np.zeros(10))
        sigma_B = pm.HalfNormal("sigma_B", 1.0)
        B = pm.Normal("B", mu=A, sigma=sigma_B, observed=np.zeros(10))
    with pytest.raises(ValueError, match="not found"):
        PyMCSCMDGP(scm, {"A": 0, "Z": 1})
    # unobserved structural node
    with pm.Model() as scm2:
        A = pm.Normal("A", mu=0.0, sigma=1.0)  # latent, not observed
    with pytest.raises(ValueError, match="observed RV"):
        PyMCSCMDGP(scm2, {"A": 0}).fit(np.zeros((10, 1)))
    # unsupported family
    with pm.Model() as scm3:
        A = pm.Normal("A", mu=0.0, sigma=1.0, observed=np.zeros(10))
        U = pm.Uniform("U", lower=0.0, upper=1.0, observed=np.zeros(10))
    with pytest.raises(NotImplementedError, match="UniformRV"):
        PyMCSCMDGP(scm3, {"A": 0, "U": 1}).fit(np.zeros((10, 2)))
    # structural function referencing an undeclared (non-parent) RV —
    # the unsubstituted-RV scan must catch it before silent corruption
    with pm.Model() as scm4:
        A = pm.Normal("A", mu=0.0, sigma=1.0, observed=np.zeros(10))
        E = pm.Normal("E", mu=0.0, sigma=1.0, observed=np.zeros(10))
        B = pm.Normal("B", mu=A + E, sigma=1.0, observed=np.zeros(10))
    with pytest.raises(ValueError, match="references RV 'E'"):
        PyMCSCMDGP(scm4, {"A": 0, "B": 1}).fit(np.zeros((10, 2)))
