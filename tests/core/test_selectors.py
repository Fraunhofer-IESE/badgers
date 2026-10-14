"""Tests for the Selector protocol and the selector presets."""

import numpy as np
import pytest

from badgers.core.selectors import (
    Selector,
    BernoulliSelector,
    ExactCountSelector,
)


def _X(n=10, d=2):
    return np.arange(n * d, dtype=float).reshape(n, d)


class TestSelectorProtocol:
    def test_bernoulli_satisfies_protocol(self):
        selector = BernoulliSelector(
            0.3, "temperature", rng=np.random.default_rng(0)
        )
        assert isinstance(selector, Selector)

    def test_plain_function_satisfies_protocol(self):
        def selector(X, y=None):
            return {"a": np.ones(len(X), dtype=bool)}

        assert isinstance(selector, Selector)

    def test_non_callable_does_not_satisfy_protocol(self):
        assert not isinstance(object(), Selector)


class TestBernoulliSelector:
    def test_returns_dict_keyed_by_variable(self):
        selector = BernoulliSelector(
            0.3, "temperature", rng=np.random.default_rng(0)
        )
        X = _X()
        masks = selector(X, y=None)
        assert set(masks) == {"temperature"}
        (mask,) = masks.values()
        assert mask.shape == (X.shape[0],)
        assert mask.dtype == bool

    def test_accepts_list_of_variables(self):
        selector = BernoulliSelector(
            0.5, ["pressure", "temperature"], rng=np.random.default_rng(0)
        )
        masks = selector(_X(50), y=None)
        assert set(masks) == {"pressure", "temperature"}
        for mask in masks.values():
            assert mask.shape == (50,)
            assert mask.dtype == bool

    def test_variables_are_drawn_independently(self):
        selector = BernoulliSelector(0.5, ["a", "b"], rng=np.random.default_rng(0))
        masks = selector(_X(200), y=None)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_single_string_is_normalized_to_tuple(self):
        selector = BernoulliSelector(
            0.3, "temperature", rng=np.random.default_rng(0)
        )
        assert selector.variables == ("temperature",)

    def test_duplicate_variables_raise(self):
        with pytest.raises(ValueError, match="unique"):
            BernoulliSelector(0.3, ["a", "a"], rng=np.random.default_rng(0))

    def test_empty_variables_raise(self):
        with pytest.raises(ValueError, match="empty"):
            BernoulliSelector(0.3, [], rng=np.random.default_rng(0))

    def test_non_str_variable_raises(self):
        with pytest.raises(ValueError, match="str"):
            BernoulliSelector(0.3, ["a", 0], rng=np.random.default_rng(0))

    def test_rate_zero_all_false(self):
        selector = BernoulliSelector(0.0, "a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None).values()
        assert not mask.any()

    def test_rate_one_all_true(self):
        selector = BernoulliSelector(1.0, "a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None).values()
        assert mask.all()

    def test_rate_fraction(self):
        selector = BernoulliSelector(0.5, "a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(2000), y=None).values()
        assert 0.45 < mask.mean() < 0.55

    def test_y_is_ignored(self):
        # Two identically-seeded selectors: passing y must not change the draw
        m1 = BernoulliSelector(0.3, "a", rng=np.random.default_rng(7))(
            _X(), y=None
        )
        m2 = BernoulliSelector(0.3, "a", rng=np.random.default_rng(7))(
            _X(), y=np.arange(10)
        )
        assert np.array_equal(m1["a"], m2["a"])

    def test_missing_rng_raises_on_call(self):
        selector = BernoulliSelector(0.3, "a")
        with pytest.raises(ValueError, match="random generator"):
            selector(_X(), y=None)

    def test_rate_below_zero_raises_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector(-0.1, "a", rng=np.random.default_rng(0))

    def test_rate_above_one_raises_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector(1.5, "a", rng=np.random.default_rng(0))

    def test_reproducible_for_same_seed(self):
        s1 = BernoulliSelector(0.3, "a", rng=np.random.default_rng(5))
        s2 = BernoulliSelector(0.3, "a", rng=np.random.default_rng(5))
        assert np.array_equal(s1(_X(), y=None)["a"], s2(_X(), y=None)["a"])

    def test_repr_is_informative(self):
        selector = BernoulliSelector(0.3, "temperature")
        text = repr(selector)
        assert "BernoulliSelector" in text
        assert "0.3" in text
        assert "temperature" in text


class TestExactCountSelector:
    def _selector(self, **kw):
        kw.setdefault("rng", np.random.default_rng(0))
        return ExactCountSelector(0.3, "temperature", **kw)

    def test_satisfies_protocol_and_exact_count(self):
        selector = self._selector()
        assert isinstance(selector, Selector)
        X = _X(50)
        (mask,) = selector(X, y=None).values()
        assert mask.shape == (50,) and mask.sum() == int(0.3 * 50)

    def test_accepts_list_of_variables_independent_masks(self):
        selector = ExactCountSelector(0.3, ["a", "b"],
                                      rng=np.random.default_rng(0))
        masks = selector(_X(30), y=None)
        assert set(masks) == {"a", "b"}
        # counts match but selections are independent
        assert masks["a"].sum() == masks["b"].sum() == int(0.3 * 30)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_weights_callable_receives_X_and_y(self):
        seen = {}

        def weights(X, y=None):
            seen["X"], seen["y"] = X, y
            return np.full((X.shape[0], 1), 1.0 / X.shape[0])

        selector = ExactCountSelector(0.2, "a", weights=weights,
                                      rng=np.random.default_rng(0))
        y = np.arange(10)
        (mask,) = selector(_X(), y=y).values()
        assert seen["y"] is y and mask.sum() == 2

    def test_weights_skew_toward_high_p(self):
        p = np.zeros((10, 1))
        p[:2, 0] = 0.5  # mass only on first two rows
        selector = ExactCountSelector(0.2, "a", weights=p,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None).values()
        assert mask.sum() == 2 and set(np.where(mask)[0]) <= {0, 1}

    def test_weights_2d_shape_mismatch_raises_on_call(self):
        selector = ExactCountSelector(0.3, ["a", "b"],
                                      weights=np.ones((10, 1)),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None)

    def test_weights_negative_raises_on_call(self):
        selector = ExactCountSelector(0.3, "a", weights=np.full((10, 1), -0.5),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None)

    def test_rate_one_selects_all_rows(self):
        selector = ExactCountSelector(1.0, "a", rng=np.random.default_rng(0))
        # rate 1.0 -> count == n_rows: valid. rate > 1 rejected at construction.
        (mask,) = selector(_X(), y=None).values()
        assert mask.all()

    def test_missing_rng_raises_on_call(self):
        with pytest.raises(ValueError, match="random generator"):
            ExactCountSelector(0.3, "a")(_X(), y=None)

    def test_rate_bounds_and_variables_validation_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector(-0.1, "a")
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector(1.5, "a")
        with pytest.raises(ValueError, match="empty"):
            ExactCountSelector(0.3, [])

    def test_reproducible_for_same_seed(self):
        s1 = ExactCountSelector(0.3, "a", rng=np.random.default_rng(5))
        s2 = ExactCountSelector(0.3, "a", rng=np.random.default_rng(5))
        assert np.array_equal(s1(_X(), y=None)["a"], s2(_X(), y=None)["a"])

    def test_repr_is_informative(self):
        text = repr(ExactCountSelector(0.3, "temperature"))
        assert "ExactCountSelector" in text and "0.3" in text
