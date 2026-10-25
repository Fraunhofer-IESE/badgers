"""Tests for AdditiveNoiseGenerator (non-causal)."""

import numpy as np
import pytest
import scipy.stats

from badgers.core.selectors import AlwaysSelector, BernoulliSelector, ClassSelector, TopQuantileSelector
from badgers.generators.tabular_data.noise.noise import AdditiveNoiseGenerator


@pytest.fixture
def gen():
    return AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))


def _simple_data(rng):
    return rng.normal(0, 1, size=(50, 3))


def test_generate__gaussian_matches_builtin(gen):
    """With scipy normal, additive noise behaves like the classic Gaussian generator."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.norm(scale=0.5), noise_std=1.0)
    assert Xt.shape == X.shape
    assert np.std(Xt - X) == pytest.approx(0.5, rel=0.2)


def test_generate__laplace_heavy_tail(gen):
    """Laplace noise (user-frozen distribution) is added as-is."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.laplace(scale=0.1), noise_std=1.0)
    delta = Xt - X
    # Laplace sample std = scale * sqrt(2)
    assert np.std(delta) == pytest.approx(0.1 * np.sqrt(2), rel=0.3)
    # Kurtosis > 3 (heavier tails than Gaussian)
    excess = scipy.stats.kurtosis(delta, axis=None)
    assert excess > 1.0


def test_generate__noise_std_multiplier(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt1 = gen.generate(X, distribution=scipy.stats.laplace(scale=1), noise_std=0.1)
    Xt2 = gen.generate(X, distribution=scipy.stats.laplace(scale=1), noise_std=0.0)
    assert np.allclose(Xt2, X)
    assert np.std(Xt1 - X) < 0.5


def test_generate__selector_scope_restricts_columns(gen):
    """Only columns in the selector's scope receive noise."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    selector = BernoulliSelector([1], rate=0.5, rng=np.random.default_rng(1))
    Xt = gen.generate(X, distribution=scipy.stats.norm(scale=0.3),
                      selector=selector)
    # only column 1 changed
    assert np.allclose(Xt[:, 0], X[:, 0])
    assert np.allclose(Xt[:, 2], X[:, 2])
    assert not np.allclose(Xt[:, 1], X[:, 1])


def test_generate__invalid_distribution_raises(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    with pytest.raises(TypeError, match="frozen scipy"):
        gen.generate(X, distribution="laplace")
    with pytest.raises(TypeError, match="rvs"):
        gen.generate(X, distribution=np.random.default_rng(0))


def test_generate__negative_noise_std_raises(gen):
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    with pytest.raises(ValueError, match="noise_std"):
        gen.generate(X, distribution=scipy.stats.norm(), noise_std=-0.1)


def test_generate__selector_none_matches_whole_data_noise(gen):
    """selector=None keeps the backward-compatible behavior: every row gets noise."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    Xt = gen.generate(X, distribution=scipy.stats.norm(scale=0.5), selector=None)
    assert Xt.shape == X.shape
    assert np.std(Xt - X) == pytest.approx(0.5, rel=0.2)


def test_generate__selector_none_is_relevant_default_selector():
    """selector=None IS a relevant default selector: AlwaysSelector over
    every column. The explicit and default paths are identical."""
    X = _simple_data(np.random.default_rng(0))
    g1 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    g2 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    Xt_none = g1.generate(X.copy(), distribution=scipy.stats.norm(scale=0.5), selector=None)
    Xt_default = g2.generate(
        X.copy(), distribution=scipy.stats.norm(scale=0.5),
        selector=AlwaysSelector(range(X.shape[1])))
    np.testing.assert_array_equal(Xt_none, Xt_default)


def test_generate__selector_restricts_noisy_rows(gen):
    """Only the columns and rows fired by the selector receive noise."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    selector = TopQuantileSelector([0, 1], 0, 0.8)
    gen2 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    Xt = gen2.generate(X, distribution=scipy.stats.norm(scale=2.0),
                       selector=selector)
    masks = selector(X)
    # column 2 never perturbed
    np.testing.assert_array_equal(Xt[:, 2], X[:, 2])
    # columns 0 and 1: non-fired rows untouched
    np.testing.assert_array_equal(Xt[~masks[0], 0], X[~masks[0], 0])
    np.testing.assert_array_equal(Xt[~masks[1], 1], X[~masks[1], 1])
    # fired rows are noisy
    assert not np.allclose(Xt[masks[0], 0], X[masks[0], 0])


def test_generate__bernoulli_selector_reproducible_with_fresh_streams():
    """A BernoulliSelector is reproducible when re-seeded per generate() call."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    gen2 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    Xt1 = gen2.generate(X, distribution=scipy.stats.norm(scale=2.0),
                        selector=BernoulliSelector([0], rate=0.5, rng=np.random.default_rng(1)))
    gen2.random_generator = np.random.default_rng(0)
    Xt2 = gen2.generate(X, distribution=scipy.stats.norm(scale=2.0),
                        selector=BernoulliSelector([0], rate=0.5, rng=np.random.default_rng(1)))
    np.testing.assert_array_equal(Xt1, Xt2)


def test_generate__class_selector_gates_noise_to_that_class(gen):
    """A ClassSelector over the class column gives noise only to that class's rows."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    y = rng.integers(0, 2, size=len(X))
    Xc = np.column_stack([X, y])
    # selector reads class values from column 3 once; the same mask is
    # returned under every covered column (0, 1, 2)
    selector = ClassSelector([0, 1, 2], class_column=3, proportions={1: 1.0},
                             rng=np.random.default_rng(2))
    gen2 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    Xt = gen2.generate(Xc, distribution=scipy.stats.norm(scale=2.0),
                       selector=selector)
    target = Xc[:, -1] == 1
    assert not np.allclose(Xt[target, :3], Xc[target, :3])
    np.testing.assert_array_equal(Xt[~target], Xc[~target])


def test_generate__selector_scope_and_firing_combine(gen):
    """Noise lands only where a fired row AND a covered column intersect."""
    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    selector = TopQuantileSelector([1], 1, 0.8)
    gen2 = AdditiveNoiseGenerator(random_generator=np.random.default_rng(0))
    Xt = gen2.generate(X, distribution=scipy.stats.norm(scale=2.0),
                       selector=selector)
    fired = selector(X)[1]
    np.testing.assert_array_equal(Xt[:, 0], X[:, 0])
    np.testing.assert_array_equal(Xt[:, 2], X[:, 2])
    np.testing.assert_array_equal(Xt[~fired, 1], X[~fired, 1])
    assert not np.allclose(Xt[fired, 1], X[fired, 1])


def test_generate__masks_computed_on_unmutated_input(gen):
    """The selector sees the original rows (compute-all-masks-first invariant)."""
    seen = []

    class Spy:
        def __call__(self, X, targets=None):
            seen.append(np.array(X, copy=True))
            return {0: np.ones(X.shape[0], dtype=bool)}

    rng = np.random.default_rng(0)
    X = _simple_data(rng)
    gen.generate(X, distribution=scipy.stats.norm(scale=1.0), selector=Spy())
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], X)
