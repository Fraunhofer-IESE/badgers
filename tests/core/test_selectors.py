"""Tests for the Selector protocol and the selector presets."""

import numpy as np
import pytest

from badgers.core.selectors import (
    Selector,
    AlwaysSelector,
    BernoulliSelector,
    ExactCountSelector,
    TopQuantileSelector,
    ClassSelector,
    ProportionalSelector,
)


def _X(n=10, d=2):
    return np.arange(n * d, dtype=float).reshape(n, d)


class TestSelectorProtocol:
    def test_bernoulli_satisfies_protocol(self):
        selector = BernoulliSelector(
            "temperature", rate=0.3, rng=np.random.default_rng(0)
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
            "temperature", rate=0.3, rng=np.random.default_rng(0)
        )
        X = _X()
        masks = selector(X, y=None)
        assert set(masks) == {"temperature"}
        (mask,) = masks.values()
        assert mask.shape == (X.shape[0],)
        assert mask.dtype == bool

    def test_accepts_list_of_variables(self):
        selector = BernoulliSelector(
            ["pressure", "temperature"], rate=0.5,
            rng=np.random.default_rng(0)
        )
        masks = selector(_X(50), y=None)
        assert set(masks) == {"pressure", "temperature"}
        for mask in masks.values():
            assert mask.shape == (50,)
            assert mask.dtype == bool

    def test_variables_are_drawn_independently(self):
        selector = BernoulliSelector(
            ["a", "b"], rate=0.5, rng=np.random.default_rng(0)
        )
        masks = selector(_X(200), y=None)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_single_string_is_normalized_to_tuple(self):
        selector = BernoulliSelector(
            "temperature", rate=0.3, rng=np.random.default_rng(0)
        )
        assert selector.variables == ("temperature",)

    def test_duplicate_variables_raise(self):
        with pytest.raises(ValueError, match="unique"):
            BernoulliSelector(["a", "a"], rate=0.3,
                              rng=np.random.default_rng(0))

    def test_empty_variables_raise(self):
        with pytest.raises(ValueError, match="empty"):
            BernoulliSelector([], rate=0.3, rng=np.random.default_rng(0))

    def test_non_str_variable_raises(self):
        with pytest.raises(ValueError, match="str"):
            BernoulliSelector(["a", 0], rate=0.3,
                              rng=np.random.default_rng(0))

    def test_rate_zero_all_false(self):
        selector = BernoulliSelector(
            "a", rate=0.0, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X(), y=None).values()
        assert not mask.any()

    def test_rate_one_all_true(self):
        selector = BernoulliSelector(
            "a", rate=1.0, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X(), y=None).values()
        assert mask.all()

    def test_rate_fraction(self):
        selector = BernoulliSelector(
            "a", rate=0.5, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X(2000), y=None).values()
        assert 0.45 < mask.mean() < 0.55

    def test_rate_stored_on_object(self):
        # rate is a constructor argument: intensity is part of the selector
        selector = BernoulliSelector("a", rate=0.25)
        assert selector.rate == 0.25

    def test_y_is_ignored(self):
        # Two identically-seeded selectors: passing y must not change the draw
        m1 = BernoulliSelector("a", rate=0.3, rng=np.random.default_rng(7))(
            _X(), y=None
        )
        m2 = BernoulliSelector("a", rate=0.3, rng=np.random.default_rng(7))(
            _X(), y=np.arange(10)
        )
        assert np.array_equal(m1["a"], m2["a"])

    def test_missing_rng_raises_on_call(self):
        selector = BernoulliSelector("a", rate=0.3)
        with pytest.raises(ValueError, match="random generator"):
            selector(_X(), y=None)

    def test_rate_validation_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector("a", rate=-0.1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector("a", rate=1.5, rng=np.random.default_rng(0))

    def test_reproducible_for_same_seed(self):
        s1 = BernoulliSelector("a", rate=0.3, rng=np.random.default_rng(5))
        s2 = BernoulliSelector("a", rate=0.3, rng=np.random.default_rng(5))
        m1 = s1(_X(), y=None)
        m2 = s2(_X(), y=None)
        assert np.array_equal(m1["a"], m2["a"])

    def test_repr_is_informative(self):
        selector = BernoulliSelector("temperature", rate=0.3)
        text = repr(selector)
        assert "BernoulliSelector" in text
        assert "temperature" in text


class TestExactCountSelector:
    def _selector(self, **kw):
        kw.setdefault("rate", 0.3)
        kw.setdefault("rng", np.random.default_rng(0))
        return ExactCountSelector("temperature", **kw)

    def test_satisfies_protocol_and_exact_count(self):
        selector = self._selector()
        assert isinstance(selector, Selector)
        X = _X(50)
        (mask,) = selector(X, y=None).values()
        assert mask.shape == (50,) and mask.sum() == int(0.3 * 50)

    def test_accepts_list_of_variables_independent_masks(self):
        selector = ExactCountSelector(["a", "b"], rate=0.3,
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

        selector = ExactCountSelector("a", rate=0.2, weights=weights,
                                      rng=np.random.default_rng(0))
        y = np.arange(10)
        (mask,) = selector(_X(), y=y).values()
        assert seen["y"] is y and mask.sum() == 2

    def test_weights_skew_toward_high_p(self):
        p = np.zeros((10, 1))
        p[:2, 0] = 0.5  # mass only on first two rows
        selector = ExactCountSelector("a", rate=0.2, weights=p,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None).values()
        assert mask.sum() == 2 and set(np.where(mask)[0]) <= {0, 1}

    def test_weights_2d_shape_mismatch_raises_on_call(self):
        selector = ExactCountSelector(["a", "b"], rate=0.3,
                                      weights=np.ones((10, 1)),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None)

    def test_weights_negative_raises_on_call(self):
        selector = ExactCountSelector("a", rate=0.3,
                                      weights=np.full((10, 1), -0.5),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X(), y=None)

    def test_rate_one_selects_all_rows(self):
        selector = ExactCountSelector("a", rate=1.0,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X(), y=None).values()
        assert mask.all()

    def test_missing_rng_raises_on_call(self):
        with pytest.raises(ValueError, match="random generator"):
            ExactCountSelector("a", rate=0.3)(_X(), y=None)

    def test_rate_validation_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector("a", rate=-0.1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector("a", rate=1.5, rng=np.random.default_rng(0))

    def test_variables_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            ExactCountSelector([], rate=0.3)

    def test_reproducible_for_same_seed(self):
        s1 = ExactCountSelector("a", rate=0.3, rng=np.random.default_rng(5))
        s2 = ExactCountSelector("a", rate=0.3, rng=np.random.default_rng(5))
        m1 = s1(_X(), y=None)
        m2 = s2(_X(), y=None)
        assert np.array_equal(m1["a"], m2["a"])

    def test_repr_is_informative(self):
        text = repr(ExactCountSelector("temperature", rate=0.3))
        assert "ExactCountSelector" in text and "temperature" in text


class TestTopQuantileSelector:
    """Fires rows above the ``quantile`` of a (driver) column.

    Legacy parity: ``threshold = quantile(col, 1 - rate)`` and
    ``fires = value >= threshold`` (the knob path's ``observed``/``self``
    mechanisms) -- hence the knob path passed ``quantile = 1 - rate``.
    """

    MAPPING = {"a": 0, "b": 1, "c": 2}

    def test_satisfies_protocol(self):
        selector = TopQuantileSelector("a", self.MAPPING, quantile=0.75)
        assert isinstance(selector, Selector)

    def test_fires_rows_above_explicit_quantile(self):
        # quantile=0.75 -> keep everything above Q3
        selector = TopQuantileSelector("a", self.MAPPING, quantile=0.75)
        X = _X(10, 3)  # col 0 == 0..9
        (mask,) = selector(X, y=None).values()
        thr = np.quantile(X[:, 0], 0.75)
        np.testing.assert_array_equal(mask, X[:, 0] >= thr)
        assert mask.sum() >= 1  # at least the maximum fires

    def test_returns_dict_keyed_by_variable(self):
        selector = TopQuantileSelector(["a", "b"], self.MAPPING, quantile=0.8)
        masks = selector(_X(10, 3), y=None)
        assert set(masks) == {"a", "b"}
        for mask in masks.values():
            assert mask.dtype == bool and mask.shape == (10,)

    def test_self_mode_masks_differ_per_variable(self):
        # make column b the REVERSE of column a: the same rule then fires
        # DIFFERENT rows for the two variables
        selector = TopQuantileSelector(["a", "b"], self.MAPPING, quantile=0.7)
        X = _X(10, 3)
        X[:, 1] = X[::-1, 0]
        masks = selector(X, y=None)
        assert not np.array_equal(masks["a"], masks["b"])

    def test_driver_mode_single_str_broadcast(self):
        # one driver gates ALL variables; the mask compares the DRIVER
        # column's values against its own threshold.
        selector = TopQuantileSelector(
            ["b", "c"], self.MAPPING, quantile=0.8, drivers="a"
        )
        X = _X(10, 3)
        masks = selector(X, y=None)
        thr = np.quantile(X[:, 0], 0.8)
        expected = X[:, 0] >= thr
        np.testing.assert_array_equal(masks["b"], expected)
        np.testing.assert_array_equal(masks["c"], expected)

    def test_driver_mode_per_variable_list(self):
        # driver a gates variable b; driver b gates variable c
        selector = TopQuantileSelector(
            ["b", "c"], self.MAPPING, quantile=0.8, drivers=["a", "b"]
        )
        X = _X(10, 3)
        masks = selector(X, y=None)
        np.testing.assert_array_equal(masks["b"], X[:, 0] >= np.quantile(X[:, 0], 0.8))
        np.testing.assert_array_equal(masks["c"], X[:, 1] >= np.quantile(X[:, 1], 0.8))

    def test_stats_kwarg_thresholds_on_other_rows(self):
        # Legacy noise-expansion case: thresholds computed on the original
        # (stats) rows, comparisons against the expanded rows.
        selector = TopQuantileSelector("a", self.MAPPING, quantile=0.8)
        stats = _X(10, 3)
        rows = np.repeat(stats, 3, axis=0)  # 30 expanded rows
        masks = selector(rows, y=None, stats=stats)
        thr = np.quantile(stats[:, 0], 0.8)
        np.testing.assert_array_equal(masks["a"], rows[:, 0] >= thr)

    def test_deterministic_no_rng(self):
        # Quantile rules involve no randomness: two calls agree, and the
        # constructor needs no rng.
        selector = TopQuantileSelector("a", self.MAPPING, quantile=0.8)
        X = _X(10, 3)
        m1 = selector(X, y=None)["a"]
        m2 = selector(X, y=None)["a"]
        assert np.array_equal(m1, m2)

    def test_quantile_one_fires_only_max_rows(self):
        # threshold = column max; rows equal to the max still fire (>=)
        selector = TopQuantileSelector("a", self.MAPPING, quantile=1.0)
        X = _X(10, 3)
        (mask,) = selector(X, y=None).values()
        assert mask.sum() == int((X[:, 0] == X[:, 0].max()).sum())

    def test_quantile_zero_fires_all_rows(self):
        selector = TopQuantileSelector("a", self.MAPPING, quantile=0.0)
        (mask,) = selector(_X(10, 3), y=None).values()
        assert mask.all()

    def test_unknown_variable_raises_at_construction(self):
        with pytest.raises(ValueError, match="not in column_mapping"):
            TopQuantileSelector("zzz", self.MAPPING, quantile=0.8)

    def test_unknown_driver_raises_at_construction(self):
        with pytest.raises(ValueError, match="not in column_mapping"):
            TopQuantileSelector("a", self.MAPPING, quantile=0.8, drivers="zzz")

    def test_drivers_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="drivers"):
            TopQuantileSelector(["a", "b"], self.MAPPING, quantile=0.8,
                                drivers=["c"])

    def test_quantile_validation_at_construction(self):
        with pytest.raises(ValueError, match="quantile"):
            TopQuantileSelector("a", self.MAPPING, quantile=-0.1)
        with pytest.raises(ValueError, match="quantile"):
            TopQuantileSelector("a", self.MAPPING, quantile=1.5)

    def test_variables_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            TopQuantileSelector([], self.MAPPING, quantile=0.8)

    def test_repr_is_informative(self):
        text = repr(TopQuantileSelector("a", self.MAPPING, quantile=0.8, drivers="c"))
        assert "TopQuantileSelector" in text
        assert "a" in text and "c" in text


class TestAlwaysSelector:
    def test_satisfies_protocol(self):
        assert isinstance(AlwaysSelector("a"), Selector)

    def test_all_true_masks(self):
        selector = AlwaysSelector(["a", "b"])
        masks = selector(_X(7), y=None)
        assert set(masks) == {"a", "b"}
        for mask in masks.values():
            assert mask.dtype == bool and mask.shape == (7,) and mask.all()

    def test_no_intensity_kwarg(self):
        # the trivial rule has nothing to sweep
        selector = AlwaysSelector("a")
        masks = selector(_X(5), y=None)
        assert masks["a"].all()

    def test_empty_variables_raise(self):
        with pytest.raises(ValueError, match="empty"):
            AlwaysSelector([])

    def test_repr_is_informative(self):
        text = repr(AlwaysSelector("temperature"))
        assert "AlwaysSelector" in text and "temperature" in text


class TestClassSelector:
    """ClassSelector: per-class fixed-count sampling on a discrete column."""

    def _X_classes(self, n_per_class=10):
        X = _X(2 * n_per_class)
        X[:, 1] = np.repeat([0, 1], n_per_class)
        return X

    def test_satisfies_protocol(self):
        selector = ClassSelector(
            "cls", {"cls": 1}, {"cls": {0: 0.5}}, rng=np.random.default_rng(0)
        )
        assert isinstance(selector, Selector)

    def test_exact_counts_per_class(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            "cls", {"cls": 1}, {"cls": {0: 0.5, 1: 0.25}},
            rng=np.random.default_rng(0),
        )
        mask = selector(X)["cls"]
        # proportions are shares of the TOTAL row count (20 rows)
        assert mask.sum() == 10 + 5
        assert mask[X[:, 1] == 0].sum() == 10
        assert mask[X[:, 1] == 1].sum() == 5

    def test_classes_absent_from_mapping_are_kept(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            "cls", {"cls": 1}, {"cls": {1: 0.25}},
            rng=np.random.default_rng(0),
        )
        mask = selector(X)["cls"]
        assert mask[X[:, 1] == 0].sum() == 0
        assert mask[X[:, 1] == 1].sum() == min(int(0.25 * 20), 10)

    def test_unknown_class_value_raises(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            "cls", {"cls": 1}, {"cls": {7: 0.5}},
            rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match="not found in column"):
            selector(X)

    def test_proportions_out_of_range_raise(self):
        with pytest.raises(ValueError, match="proportions"):
            ClassSelector("cls", {"cls": 1}, {"cls": {0: 1.5}},
                          rng=np.random.default_rng(0))

    def test_reserved_y_name_reads_target(self):
        X = _X(10)
        X[:, 1] = np.repeat([0, 1], 5)
        selector = ClassSelector("y", {"y": 1}, {"y": {0: 0.4}}, rng=np.random.default_rng(0))
        mask = selector(X)["y"]
        assert mask.sum() == min(int(0.4 * 10), 5) == 4
        assert mask[X[:, 1] == 0].sum() == 4
        assert mask[X[:, 1] == 1].sum() == 0

    def test_y_is_unused_for_selection(self):
        # y is deprecated as a data carrier: selection must not depend on it
        X = self._X_classes(10)
        selector = ClassSelector("cls", {"cls": 1}, {"cls": {0: 0.5}},
                                 rng=np.random.default_rng(0))
        mask_a = selector(X, y=np.arange(20))["cls"]
        mask_b = selector(X, y=None)["cls"]
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_calling_without_rng_raises(self):
        selector = ClassSelector("cls", {"cls": 1}, {"cls": {0: 0.5}})
        with pytest.raises(ValueError, match="random generator"):
            selector(self._X_classes(10))

    def test_missing_proportions_entry_raises(self):
        with pytest.raises(ValueError, match="no entry in proportions"):
            ClassSelector("cls", {"cls": 1}, {"other": {0: 0.5}})

    def test_variable_not_in_column_mapping_raises(self):
        with pytest.raises(ValueError, match="column_mapping"):
            ClassSelector("nope", {"cls": 1}, {"nope": {0: 0.5}})

    def test_repr_is_informative(self):
        selector = ClassSelector("cls", {"cls": 1}, {"cls": {0: 0.5}})
        text = repr(selector)
        assert "ClassSelector" in text and "cls" in text


class TestProportionalSelector:
    def _X(self):
        return np.array([1.0, 2.0, 3.0, 6.0]).reshape(-1, 1)

    def test_scores_are_normalized_column_values(self):
        selector = ProportionalSelector("f0", {"f0": 0})
        scores = selector(self._X())["f0"]
        # min-max to [0, 1], then normalized to sum 1
        np.testing.assert_allclose(scores, [0.0, 0.2 / 1.6, 0.4 / 1.6, 1.0 / 1.6])
        assert scores.sum() == pytest.approx(1.0)

    def test_fulfills_protocol(self):
        assert isinstance(ProportionalSelector("f0", {"f0": 0}), Selector)

    def test_multiple_variables_independent(self):
        X = np.arange(8.0).reshape(4, 2)
        selector = ProportionalSelector(["a", "b"], {"a": 0, "b": 1})
        out = selector(X)
        assert set(out) == {"a", "b"}
        np.testing.assert_allclose(out["a"], minmax_normalize(X[:, 0]))
        np.testing.assert_allclose(out["b"], minmax_normalize(X[:, 1]))

    def test_variable_not_in_column_mapping_raises(self):
        with pytest.raises(ValueError, match="column_mapping"):
            ProportionalSelector("nope", {"f0": 0})

    def test_repr_is_informative(self):
        selector = ProportionalSelector("f0", {"f0": 0})
        assert "ProportionalSelector" in repr(selector) and "f0" in repr(selector)


def minmax_normalize(values):
    span = values.max() - values.min()
    unit = (values - values.min()) / span if span > 0 else np.ones_like(values)
    return unit / unit.sum()
