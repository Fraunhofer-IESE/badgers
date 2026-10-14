"""Tests for the Selector protocol and the selector presets."""

import numpy as np
import pytest

from badgers.core.selectors import (
    Selector,
    BernoulliSelector,
    ExactCountSelector,
    TopQuantileSelector,
)


def _X(n=10, d=2):
    return np.arange(n * d, dtype=float).reshape(n, d)


class TestSelectorProtocol:
    def test_bernoulli_satisfies_protocol(self):
        selector = BernoulliSelector(
            "temperature", rng=np.random.default_rng(0)
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
            "temperature", rng=np.random.default_rng(0)
        )
        X = _X()
        masks = selector(X, y=None, rate=0.3)
        assert set(masks) == {"temperature"}
        (mask,) = masks.values()
        assert mask.shape == (X.shape[0],)
        assert mask.dtype == bool

    def test_accepts_list_of_variables(self):
        selector = BernoulliSelector(
            ["pressure", "temperature"], rng=np.random.default_rng(0)
        )
        masks = selector(_X(50), y=None, rate=0.5)
        assert set(masks) == {"pressure", "temperature"}
        for mask in masks.values():
            assert mask.shape == (50,)
            assert mask.dtype == bool

    def test_variables_are_drawn_independently(self):
        selector = BernoulliSelector(["a", "b"], rng=np.random.default_rng(0))
        masks = selector(_X(200), y=None, rate=0.5)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_single_string_is_normalized_to_tuple(self):
        selector = BernoulliSelector(
            "temperature", rng=np.random.default_rng(0)
        )
        assert selector.variables == ("temperature",)

    def test_duplicate_variables_raise(self):
        with pytest.raises(ValueError, match="unique"):
            BernoulliSelector(["a", "a"], rng=np.random.default_rng(0))

    def test_empty_variables_raise(self):
        with pytest.raises(ValueError, match="empty"):
            BernoulliSelector([], rng=np.random.default_rng(0))

    def test_non_str_variable_raises(self):
        with pytest.raises(ValueError, match="str"):
            BernoulliSelector(["a", 0], rng=np.random.default_rng(0))

    def test_rate_zero_all_false(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None, rate=0.0).values()
        assert not mask.any()

    def test_rate_one_all_true(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None, rate=1.0).values()
        assert mask.all()

    def test_rate_fraction(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        (mask,) = selector(_X(2000), y=None, rate=0.5).values()
        assert 0.45 < mask.mean() < 0.55

    def test_rate_is_swept_on_one_object(self):
        # rate is a CALL argument: the same selector object can be used
        # with different intensities without re-constructing it
        selector = BernoulliSelector("a", rng=np.random.default_rng(7))
        m1 = selector(_X(), y=None, rate=0.1)
        m2 = selector(_X(), y=None, rate=0.9)
        assert m1["a"].sum() < m2["a"].sum()

    def test_y_is_ignored(self):
        # Two identically-seeded selectors: passing y must not change the draw
        m1 = BernoulliSelector("a", rng=np.random.default_rng(7))(
            _X(), y=None, rate=0.3
        )
        m2 = BernoulliSelector("a", rng=np.random.default_rng(7))(
            _X(), y=np.arange(10), rate=0.3
        )
        assert np.array_equal(m1["a"], m2["a"])

    def test_missing_rng_raises_on_call(self):
        selector = BernoulliSelector("a")
        with pytest.raises(ValueError, match="random generator"):
            selector(_X(), y=None, rate=0.3)

    def test_rate_below_zero_raises_on_call(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            selector(_X(), y=None, rate=-0.1)

    def test_rate_above_one_raises_on_call(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            selector(_X(), y=None, rate=1.5)

    def test_missing_rate_raises_on_call(self):
        selector = BernoulliSelector("a", rng=np.random.default_rng(0))
        with pytest.raises(TypeError):
            selector(_X(), y=None)

    def test_reproducible_for_same_seed(self):
        s1 = BernoulliSelector("a", rng=np.random.default_rng(5))
        s2 = BernoulliSelector("a", rng=np.random.default_rng(5))
        m1 = s1(_X(), y=None, rate=0.3)
        m2 = s2(_X(), y=None, rate=0.3)
        assert np.array_equal(m1["a"], m2["a"])

    def test_repr_is_informative(self):
        selector = BernoulliSelector("temperature")
        text = repr(selector)
        assert "BernoulliSelector" in text
        assert "temperature" in text


class TestExactCountSelector:
    def _selector(self, **kw):
        kw.setdefault("rng", np.random.default_rng(0))
        return ExactCountSelector("temperature", **kw)

    def test_satisfies_protocol_and_exact_count(self):
        selector = self._selector()
        assert isinstance(selector, Selector)
        X = _X(50)
        (mask,) = selector(X, y=None, rate=0.3).values()
        assert mask.shape == (50,) and mask.sum() == int(0.3 * 50)

    def test_accepts_list_of_variables_independent_masks(self):
        selector = ExactCountSelector(["a", "b"],
                                      rng=np.random.default_rng(0))
        masks = selector(_X(30), y=None, rate=0.3)
        assert set(masks) == {"a", "b"}
        # counts match but selections are independent
        assert masks["a"].sum() == masks["b"].sum() == int(0.3 * 30)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_weights_callable_receives_X_and_y(self):
        seen = {}

        def weights(X, y=None):
            seen["X"], seen["y"] = X, y
            return np.full((X.shape[0], 1), 1.0 / X.shape[0])

        selector = ExactCountSelector("a", weights=weights,
                                      rng=np.random.default_rng(0))
        y = np.arange(10)
        (mask,) = selector(_X(), y=y, rate=0.2).values()
        assert seen["y"] is y and mask.sum() == 2

    def test_weights_skew_toward_high_p(self):
        p = np.zeros((10, 1))
        p[:2, 0] = 0.5  # mass only on first two rows
        selector = ExactCountSelector("a", weights=p,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None, rate=0.2).values()
        assert mask.sum() == 2 and set(np.where(mask)[0]) <= {0, 1}

    def test_weights_2d_shape_mismatch_raises_on_call(self):
        selector = ExactCountSelector(["a", "b"],
                                      weights=np.ones((10, 1)),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None, rate=0.3)

    def test_weights_negative_raises_on_call(self):
        selector = ExactCountSelector("a", weights=np.full((10, 1), -0.5),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None, rate=0.3)

    def test_rate_one_selects_all_rows(self):
        selector = ExactCountSelector("a", rng=np.random.default_rng(0))
        # rate 1.0 -> count == n_rows: valid. rate > 1 rejected at call.
        (mask,) = selector(_X(), y=None, rate=1.0).values()
        assert mask.all()

    def test_missing_rng_raises_on_call(self):
        with pytest.raises(ValueError, match="random generator"):
            ExactCountSelector("a")(_X(), y=None, rate=0.3)

    def test_rate_validation_on_call(self):
        selector = ExactCountSelector("a", rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            selector(_X(), y=None, rate=-0.1)
        with pytest.raises(ValueError, match="rate"):
            selector(_X(), y=None, rate=1.5)
        with pytest.raises(TypeError):
            selector(_X(), y=None)

    def test_variables_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            ExactCountSelector([])

    def test_reproducible_for_same_seed(self):
        s1 = ExactCountSelector("a", rng=np.random.default_rng(5))
        s2 = ExactCountSelector("a", rng=np.random.default_rng(5))
        m1 = s1(_X(), y=None, rate=0.3)
        m2 = s2(_X(), y=None, rate=0.3)
        assert np.array_equal(m1["a"], m2["a"])

    def test_repr_is_informative(self):
        text = repr(ExactCountSelector("temperature"))
        assert "ExactCountSelector" in text and "temperature" in text


class TestTopQuantileSelector:
    """Fires rows above the ``quantile`` of a (driver) column.

    Legacy parity: ``threshold = quantile(col, 1 - rate)`` and
    ``fires = value >= threshold`` (the knob path's ``observed``/``self``
    mechanisms) -- hence the knob path passes ``quantile = 1 - rate``.
    """

    MAPPING = {"a": 0, "b": 1, "c": 2}

    def test_satisfies_protocol(self):
        selector = TopQuantileSelector("a", self.MAPPING)
        assert isinstance(selector, Selector)

    def test_fires_rows_above_explicit_quantile(self):
        # quantile=0.75 -> keep everything above Q3
        selector = TopQuantileSelector("a", self.MAPPING)
        X = _X(10, 3)  # col 0 == 0..9
        (mask,) = selector(X, y=None, quantile=0.75).values()
        thr = np.quantile(X[:, 0], 0.75)
        np.testing.assert_array_equal(mask, X[:, 0] >= thr)
        assert mask.sum() >= 1  # at least the maximum fires

    def test_returns_dict_keyed_by_variable(self):
        selector = TopQuantileSelector(["a", "b"], self.MAPPING)
        masks = selector(_X(10, 3), y=None, quantile=0.8)
        assert set(masks) == {"a", "b"}
        for mask in masks.values():
            assert mask.dtype == bool and mask.shape == (10,)

    def test_self_mode_masks_differ_per_variable(self):
        # make column b the REVERSE of column a: the same rule then fires
        # DIFFERENT rows for the two variables
        selector = TopQuantileSelector(["a", "b"], self.MAPPING)
        X = _X(10, 3)
        X[:, 1] = X[::-1, 0]
        masks = selector(X, y=None, quantile=0.7)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_driver_mode_single_str_broadcast(self):
        # one driver gates ALL variables; the mask compares the DRIVER
        # column's values against its own threshold.
        selector = TopQuantileSelector(
            ["b", "c"], self.MAPPING, drivers="a"
        )
        X = _X(10, 3)
        masks = selector(X, y=None, quantile=0.8)
        thr = np.quantile(X[:, 0], 0.8)
        expected = X[:, 0] >= thr
        np.testing.assert_array_equal(masks["b"], expected)
        np.testing.assert_array_equal(masks["c"], expected)

    def test_driver_mode_per_variable_list(self):
        # driver a gates variable b; driver b gates variable c
        selector = TopQuantileSelector(
            ["b", "c"], self.MAPPING, drivers=["a", "b"]
        )
        X = _X(10, 3)
        masks = selector(X, y=None, quantile=0.8)
        np.testing.assert_array_equal(masks["b"], X[:, 0] >= np.quantile(X[:, 0], 0.8))
        np.testing.assert_array_equal(masks["c"], X[:, 1] >= np.quantile(X[:, 1], 0.8))

    def test_quantile_is_swept_on_one_object(self):
        # quantile is a CALL argument: the same selector object can be
        # used with different thresholds without re-constructing it
        selector = TopQuantileSelector("a", self.MAPPING)
        X = _X(10, 3)
        m1 = selector(X, y=None, quantile=0.5)["a"]
        m2 = selector(X, y=None, quantile=0.9)["a"]
        assert m1.sum() > m2.sum()

    def test_stats_kwarg_thresholds_on_other_rows(self):
        # Legacy noise-expansion case: thresholds computed on the original
        # (stats) rows, comparisons against the expanded rows.
        selector = TopQuantileSelector("a", self.MAPPING)
        stats = _X(10, 3)
        rows = np.repeat(stats, 3, axis=0)  # 30 expanded rows
        masks = selector(rows, y=None, quantile=0.8, stats=stats)
        thr = np.quantile(stats[:, 0], 0.8)
        np.testing.assert_array_equal(masks["a"], rows[:, 0] >= thr)

    def test_deterministic_no_rng(self):
        # Quantile rules involve no randomness: two calls agree, and the
        # constructor needs no rng.
        selector = TopQuantileSelector("a", self.MAPPING)
        X = _X(10, 3)
        m1 = selector(X, y=None, quantile=0.8)["a"]
        m2 = selector(X, y=None, quantile=0.8)["a"]
        assert np.array_equal(m1, m2)

    def test_quantile_one_fires_only_max_rows(self):
        # threshold = column max; rows equal to the max still fire (>=)
        selector = TopQuantileSelector("a", self.MAPPING)
        X = _X(10, 3)
        (mask,) = selector(X, y=None, quantile=1.0).values()
        assert mask.sum() == int((X[:, 0] == X[:, 0].max()).sum())

    def test_quantile_zero_fires_all_rows(self):
        selector = TopQuantileSelector("a", self.MAPPING)
        (mask,) = selector(_X(10, 3), y=None, quantile=0.0).values()
        assert mask.all()

    def test_unknown_variable_raises_at_construction(self):
        with pytest.raises(ValueError, match="not in column_mapping"):
            TopQuantileSelector("zzz", self.MAPPING)

    def test_unknown_driver_raises_at_construction(self):
        with pytest.raises(ValueError, match="not in column_mapping"):
            TopQuantileSelector("a", self.MAPPING, drivers="zzz")

    def test_drivers_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="drivers"):
            TopQuantileSelector(["a", "b"], self.MAPPING,
                                drivers=["c"])

    def test_quantile_validation_on_call(self):
        selector = TopQuantileSelector("a", self.MAPPING)
        with pytest.raises(ValueError, match="quantile"):
            selector(_X(), y=None, quantile=-0.1)
        with pytest.raises(ValueError, match="quantile"):
            selector(_X(), y=None, quantile=1.5)
        with pytest.raises(TypeError):
            selector(_X(), y=None)

    def test_variables_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            TopQuantileSelector([], self.MAPPING)

    def test_repr_is_informative(self):
        text = repr(TopQuantileSelector("a", self.MAPPING, drivers="c"))
        assert "TopQuantileSelector" in text
        assert "a" in text and "c" in text
