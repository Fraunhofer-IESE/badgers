"""Tests for CausalAdditiveNoiseGenerator."""

import numpy as np
import pytest
import scipy.stats

from badgers.core.selectors import (
    AlwaysSelector,
    BernoulliSelector,
)
from badgers.generators.tabular_data.noise.causal import CausalAdditiveNoiseGenerator


def _world(X, graph, column_mapping):
    """Fitted ``GcmSCMDGP`` world (linear ANM SCM via the conftest helper)."""
    pytest.importorskip("dowhy.gcm")
    from tests.conftest import make_gcm_scm
    from badgers.core.dgp.backends.gcm_scm import GcmSCMDGP
    return GcmSCMDGP(make_gcm_scm(graph), column_mapping).fit(X)


def test_generate__shape_and_labels(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.laplace(scale=0.5),
    )
    # strictly in-place
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))
    # every row fires
    labels = gen.get_noise_labels()
    assert labels.dtype == bool
    assert labels.all()


def test_generate__inplace_returns_only_rows(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.norm(scale=0.1),
    )
    assert Xt.shape == X.shape
    assert gen.get_noise_labels().shape == (Xt.shape[0],)


def test_generate__x_changes(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.laplace(scale=0.5),
    )
    assert not np.allclose(Xt[:, 0], X[:, 0])
    # Laplace sample std = scale * sqrt(2)
    assert np.std(Xt[:, 0] - X[:, 0]) == pytest.approx(0.5 * np.sqrt(2), rel=0.3)


def test_generate__zero_noise_descendants_untouched(chain_data):
    """noise_std=0 -> X unchanged and OLS recomputation leaves rows unchanged."""
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.norm(), noise_std=0.0,
    )
    # zero injected noise: the perturbed root column is unchanged, but its
    # descendants are recomputed with FRESH structural noise (Pearl surgery)
    assert np.allclose(Xt[:, 0], X[:, 0])
    assert not np.allclose(Xt[:, 1], X[:, 1])
    assert not np.allclose(Xt[:, 2], X[:, 2])


def test_generate__propagation(chain_data):
    """Descendant columns move; independent exogenous root does not."""
    X, graph, column_mapping = chain_data
    X_orig = X.copy()
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.laplace(scale=0.5),
    )
    # X changed (injected noise)
    assert not np.allclose(Xt[:, 0], X_orig[:, 0])
    # Y and Z are descendants of X: recomputed, moved
    assert not np.allclose(Xt[:, 1], X_orig[:, 1])
    assert not np.allclose(Xt[:, 2], X_orig[:, 2])
    # correlation with original still substantial (recomputed, not destroyed)
    corr = np.corrcoef(Xt[:, 1], X_orig[:, 1])[0, 1]
    assert corr > 0.7


def test_generate__no_dgp_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(
            X, AlwaysSelector([0]),
            distribution=scipy.stats.norm(scale=0.1),
        )


def test_generate__no_structure_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    gen.dgp = None
    gen.graph = None
    gen.column_mapping = None
    with pytest.raises(RuntimeError, match="no structure"):
        gen.generate(
            X, AlwaysSelector([0]),
            distribution=scipy.stats.norm(scale=0.1),
        )


def test_generate__invalid_distribution_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    with pytest.raises(TypeError, match="frozen scipy"):
        gen.generate(
            X, AlwaysSelector([0]),
            distribution="normal",
        )


def test_generate__negative_noise_std_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    with pytest.raises(ValueError, match="noise_std"):
        gen.generate(
            X, AlwaysSelector([0]),
            distribution=scipy.stats.norm(), noise_std=-1.0,
        )


def test_generate__invalid_perturbation_node_raises(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    with pytest.raises(ValueError, match="out of range"):
        gen.generate(
            X, AlwaysSelector([3]),
            distribution=scipy.stats.norm(),
        )


def test_generate__lazy_fit(chain_data, make_gcm_dgp):
    X, graph, column_mapping = chain_data
    dgp = make_gcm_dgp(graph, column_mapping)
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=dgp,
    )
    assert not dgp.is_fitted()
    Xt = gen.generate(
        X, AlwaysSelector([0]),
        distribution=scipy.stats.norm(scale=0.1),
    )
    assert dgp.is_fitted()
    assert Xt.shape == X.shape


def test_generate__selector_gates_rows(chain_data):
    X, graph, column_mapping = chain_data
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0),
        dgp=_world(X, graph, column_mapping),
    )
    # Bernoulli selector: only ~half the rows fire
    Xt = gen.generate(
        X, BernoulliSelector([0], rate=0.5, rng=gen.random_generator),
        distribution=scipy.stats.norm(scale=0.5),
    )
    assert Xt.shape == X.shape
    fired = gen.get_noise_labels()
    assert fired.any() and not fired.all()
    # unfired rows are untouched
    assert np.array_equal(Xt[~fired], X[~fired])


def test_generate__always_selector_reproducible(chain_data):
    X, graph, column_mapping = chain_data
    dgp = _world(X, graph, column_mapping)
    base = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0), dgp=dgp,
    )
    Xt_first = base.generate(
        X.copy(), AlwaysSelector([0]),
        distribution=scipy.stats.norm(scale=0.5),
    )
    # labels: AlwaysSelector fires EVERY row
    np.testing.assert_array_equal(
        base.get_noise_labels(), np.ones(X.shape[0], dtype=bool)
    )
    # a second run with the same rng seed is bit-identical
    gen = CausalAdditiveNoiseGenerator(
        random_generator=np.random.default_rng(0), dgp=dgp,
    )
    Xt = gen.generate(
        X.copy(), AlwaysSelector([0]),
        distribution=scipy.stats.norm(scale=0.5),
    )
    np.testing.assert_array_equal(
        gen.get_noise_labels(), base.get_noise_labels()
    )
    np.testing.assert_array_equal(Xt, Xt_first)
