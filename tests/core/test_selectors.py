"""Tests for the Selector protocol and the selector presets.

The selector API is INDEX-ONLY: columns, drivers and class columns are
int positions into the array passed at call time; the presets return
``dict[int, spec]`` keyed by those positions. Variable-name and
``column_mapping`` based construction does not exist at the selector
layer.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from badgers.core.selectors import (
    Selector,
    AlwaysSelector,
    BernoulliSelector,
    ExactCountSelector,
    TopQuantileSelector,
    ClassSelector,
    ProportionalSelector,
)
from badgers.core.selectors.base import (
    _check_spec,
    _resolve_targets,
    _validate_columns,
    fired_mask,
    realize_firing,
)


def _X(n=10, d=2):
    return np.arange(n * d, dtype=float).reshape(n, d)


class TestSelectorProtocol:
    def test_bernoulli_satisfies_protocol(self):
        selector = BernoulliSelector(
            0, rate=0.3, rng=np.random.default_rng(0)
        )
        assert isinstance(selector, Selector)

    def test_plain_function_satisfies_protocol(self):
        def selector(X):
            return {0: np.ones(len(X), dtype=bool)}

        assert isinstance(selector, Selector)

    def test_non_callable_does_not_satisfy_protocol(self):
        assert not isinstance(object(), Selector)


# The remaining classes test the concrete presets on the index-only
# API: construction binds int column positions (the selector's scope
# and, for value-driven rules, the column(s) it reads) and nothing
# is swept on the call; ``targets`` at call time must stay inside the
# scope. Value-free presets are exercised in TestValueFree and
# value-driven presets in TestValueDriven; the classes below pin the
# full per-preset behavior on the same API.


class TestBernoulliSelector:
    def test_returns_dict_keyed_by_column_position(self):
        selector = BernoulliSelector(1, rate=0.3, rng=np.random.default_rng(0))
        X = _X()
        masks = selector(X)
        assert set(masks) == {1}
        (mask,) = masks.values()
        assert mask.shape == (X.shape[0],)
        assert mask.dtype == bool

    def test_accepts_sequence_of_positions(self):
        selector = BernoulliSelector(
            [0, 1], rate=0.5, rng=np.random.default_rng(0)
        )
        masks = selector(_X(50))
        assert set(masks) == {0, 1}
        for mask in masks.values():
            assert mask.shape == (50,)
            assert mask.dtype == bool

    def test_columns_are_drawn_independently(self):
        selector = BernoulliSelector(
            [0, 1], rate=0.5, rng=np.random.default_rng(0)
        )
        masks = selector(_X(200))
        assert not np.array_equal(masks[0], masks[1])

    def test_single_int_is_normalized_to_tuple(self):
        selector = BernoulliSelector(0, rate=0.3, rng=np.random.default_rng(0))
        assert selector.columns == (0,)

    def test_duplicate_columns_raise(self):
        with pytest.raises(ValueError, match="unique"):
            BernoulliSelector([0, 0], rate=0.3, rng=np.random.default_rng(0))

    def test_empty_columns_raise(self):
        with pytest.raises(ValueError, match="empty"):
            BernoulliSelector([], rate=0.3, rng=np.random.default_rng(0))

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            BernoulliSelector([-1], rate=0.3, rng=np.random.default_rng(0))

    def test_non_int_position_raises(self):
        with pytest.raises(ValueError, match="must be ints"):
            BernoulliSelector(["a"], rate=0.3, rng=np.random.default_rng(0))

    def test_rate_zero_all_false(self):
        selector = BernoulliSelector(
            0, rate=0.0, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X()).values()
        assert not mask.any()

    def test_rate_one_all_true(self):
        selector = BernoulliSelector(
            0, rate=1.0, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X()).values()
        assert mask.all()

    def test_rate_fraction(self):
        selector = BernoulliSelector(
            0, rate=0.5, rng=np.random.default_rng(0)
        )
        (mask,) = selector(_X(2000)).values()
        assert 0.45 < mask.mean() < 0.55

    def test_rate_stored_on_object(self):
        # rate is a constructor argument: intensity is part of the selector
        selector = BernoulliSelector(0, rate=0.25)
        assert selector.rate == 0.25

    def test_missing_rng_raises_on_call(self):
        selector = BernoulliSelector(0, rate=0.3)
        with pytest.raises(ValueError, match="random generator"):
            selector(_X())

    def test_rate_validation_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector(0, rate=-0.1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            BernoulliSelector(0, rate=1.5, rng=np.random.default_rng(0))

    def test_out_of_range_targets_raise(self):
        selector = BernoulliSelector(0, rate=0.3, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="out of range"):
            selector(_X(4, 2), targets=[2])
        with pytest.raises(ValueError, match="outside the selector"):
            selector(_X(4, 2), targets=[1])

    def test_reproducible_for_same_seed(self):
        s1 = BernoulliSelector(0, rate=0.3, rng=np.random.default_rng(5))
        s2 = BernoulliSelector(0, rate=0.3, rng=np.random.default_rng(5))
        m1 = s1(_X())
        m2 = s2(_X())
        assert np.array_equal(m1[0], m2[0])

    def test_repr_is_informative(self):
        selector = BernoulliSelector(3, rate=0.3)
        text = repr(selector)
        assert "BernoulliSelector" in text
        assert "(3,)" in text


class TestExactCountSelector:
    def _selector(self, **kw):
        kw.setdefault("rate", 0.3)
        kw.setdefault("rng", np.random.default_rng(0))
        return ExactCountSelector(0, **kw)

    def test_satisfies_protocol_and_exact_count(self):
        selector = self._selector()
        assert isinstance(selector, Selector)
        X = _X(50)
        (mask,) = selector(X).values()
        assert mask.shape == (50,)
        assert mask.dtype == bool
        assert mask.sum() == int(0.3 * 50)

    def test_accepts_list_of_positions_independent_masks(self):
        selector = ExactCountSelector([0, 1], rate=0.3,
                                      rng=np.random.default_rng(0))
        masks = selector(_X(30))
        assert set(masks) == {0, 1}
        # counts match but selections are independent
        assert masks[0].sum() == masks[1].sum() == int(0.3 * 30)
        assert not np.array_equal(masks[0], masks[1])

    def test_weights_callable_receives_X(self):
        seen = {}

        def weights(X):
            seen["X"] = X
            return np.full((X.shape[0], 1), 1.0 / X.shape[0])

        selector = ExactCountSelector(0, rate=0.2, weights=weights,
                                      rng=np.random.default_rng(0))
        X = _X()
        (mask,) = selector(X).values()
        assert seen["X"] is X and mask.sum() == 2

    def test_weights_skew_toward_high_p(self):
        p = np.zeros((10, 1))
        p[:2, 0] = 0.5  # mass only on first two rows
        selector = ExactCountSelector(0, rate=0.2, weights=p,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X()).values()
        assert mask.sum() == 2 and set(np.where(mask)[0]) <= {0, 1}

    def test_weights_2d_shape_mismatch_raises_on_call(self):
        selector = ExactCountSelector([0, 1], rate=0.3,
                                      weights=np.ones((10, 1)),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X())

    def test_weights_negative_raises_on_call(self):
        selector = ExactCountSelector(0, rate=0.3,
                                      weights=np.full((10, 1), -0.5),
                                      rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="weights"):
            selector(_X())

    def test_rate_one_selects_all_rows(self):
        selector = ExactCountSelector(0, rate=1.0,
                                      rng=np.random.default_rng(0))
        (mask,) = selector(_X()).values()
        assert mask.all()

    def test_missing_rng_raises_on_call(self):
        with pytest.raises(ValueError, match="random generator"):
            ExactCountSelector(0, rate=0.3)(_X())

    def test_rate_validation_at_construction(self):
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector(0, rate=-0.1, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="rate"):
            ExactCountSelector(0, rate=1.5, rng=np.random.default_rng(0))

    def test_columns_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            ExactCountSelector([], rate=0.3)

    def test_reproducible_for_same_seed(self):
        s1 = ExactCountSelector(0, rate=0.3, rng=np.random.default_rng(5))
        s2 = ExactCountSelector(0, rate=0.3, rng=np.random.default_rng(5))
        m1 = s1(_X())
        m2 = s2(_X())
        assert np.array_equal(m1[0], m2[0])

    def test_repr_is_informative(self):
        text = repr(ExactCountSelector(3, rate=0.3))
        assert "ExactCountSelector" in text and "(3,)" in text


class TestTopQuantileSelector:
    """Fires rows where the driver value is >= its quantile.

    ``threshold = np.quantile(X[:, driver], quantile)``; a row fires when
    the driver value is ``>= threshold`` and the same mask is returned
    under every target column. ``driver=None`` = self mode (the column's
    own threshold).
    """

    def test_satisfies_protocol(self):
        selector = TopQuantileSelector(0, column=0, quantile=0.75)
        assert isinstance(selector, Selector)

    def test_fires_rows_above_explicit_quantile(self):
        # quantile=0.75 -> keep everything above Q3
        selector = TopQuantileSelector(0, column=0, quantile=0.75)
        X = _X(10, 3)  # col 0 == 0, 3, ..., 27
        (mask,) = selector(X).values()
        thr = np.quantile(X[:, 0], 0.75)
        np.testing.assert_array_equal(mask, X[:, 0] >= thr)
        assert mask.sum() >= 1  # at least the maximum fires

    def test_returns_dict_keyed_by_position(self):
        selector = TopQuantileSelector([0, 1], column=0, quantile=0.8)
        masks = selector(_X(10, 3))
        assert set(masks) == {0, 1}
        for mask in masks.values():
            assert mask.dtype == bool and mask.shape == (10,)

    def test_self_mode_masks_differ_per_column_only_when_data_differ(self):
        # Self mode compares each target column against ITS OWN threshold:
        # reversed column data then fire different rows. In self mode the
        # mask itself is always built from column 0 (the read column), so
        # this only holds when the READ column differs between calls --
        # expressed here by two selectors bound to different read columns.
        X = _X(10, 3)
        X[:, 1] = X[::-1, 0]
        sel_a = TopQuantileSelector([0], column=0, quantile=0.7)
        sel_b = TopQuantileSelector([1], column=1, quantile=0.7)
        assert not np.array_equal(sel_a(X)[0], sel_b(X)[1])

    def test_one_read_column_broadcasts_same_mask_to_every_target(self):
        # One bound read column: the mask compares THAT column's values
        # against its own threshold and is shared by all targets.
        selector = TopQuantileSelector(
            [1, 2], column=0, quantile=0.8
        )
        X = _X(10, 3)
        masks = selector(X)
        thr = np.quantile(X[:, 0], 0.8)
        expected = X[:, 0] >= thr
        np.testing.assert_array_equal(masks[1], expected)
        np.testing.assert_array_equal(masks[2], expected)
        assert masks[1] is masks[2]

    def test_driver_mode_read_column_broadcast(self):
        # driver=1: the mask compares the DRIVER column's values against
        # its own threshold (read column 0 is then only a slot holder).
        selector = TopQuantileSelector(
            [1, 2], column=0, quantile=0.8, driver=1
        )
        X = _X(10, 3)
        masks = selector(X)
        thr = np.quantile(X[:, 1], 0.8)
        expected = X[:, 1] >= thr
        np.testing.assert_array_equal(masks[1], expected)
        np.testing.assert_array_equal(masks[2], expected)
        assert masks[1] is masks[2]

    def test_deterministic_no_rng(self):
        # Quantile rules involve no randomness: two calls agree, and the
        # constructor needs no rng.
        selector = TopQuantileSelector(0, column=0, quantile=0.8)
        X = _X(10, 3)
        m1 = selector(X)[0]
        m2 = selector(X)[0]
        assert np.array_equal(m1, m2)

    def test_quantile_one_fires_only_max_rows(self):
        # threshold = column max; rows equal to the max still fire (>=)
        selector = TopQuantileSelector(0, column=0, quantile=1.0)
        X = _X(10, 3)
        (mask,) = selector(X).values()
        assert mask.sum() == int((X[:, 0] == X[:, 0].max()).sum())

    def test_quantile_zero_fires_all_rows(self):
        selector = TopQuantileSelector(0, column=0, quantile=0.0)
        (mask,) = selector(_X(10, 3)).values()
        assert mask.all()

    def test_non_int_read_column_raises_at_construction(self):
        with pytest.raises(ValueError, match="column must be an int"):
            TopQuantileSelector(0, column="zzz", quantile=0.8)

    def test_quantile_validation_at_construction(self):
        with pytest.raises(ValueError, match="quantile"):
            TopQuantileSelector(0, column=0, quantile=-0.1)
        with pytest.raises(ValueError, match="quantile"):
            TopQuantileSelector(0, column=0, quantile=1.5)

    def test_columns_validation_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            TopQuantileSelector([], column=0, quantile=0.8)

    def test_repr_is_informative(self):
        text = repr(TopQuantileSelector(0, column=0, quantile=0.8, driver=2))
        assert "TopQuantileSelector" in text
        assert "(0,)" in text and "driver=2" in text


class TestAlwaysSelector:
    def test_satisfies_protocol(self):
        assert isinstance(AlwaysSelector(0), Selector)

    def test_uniform_spec_one_for_all_targets(self):
        selector = AlwaysSelector([0, 1])
        out = selector(_X(7))
        assert set(out) == {0, 1}
        for spec in out.values():
            assert spec == 1.0

    def test_no_intensity_kwarg(self):
        # the trivial rule has nothing to sweep
        selector = AlwaysSelector(0)
        assert selector(_X(5))[0] == 1.0

    def test_empty_columns_raise(self):
        with pytest.raises(ValueError, match="empty"):
            AlwaysSelector([])

    def test_out_of_scope_targets_raise(self):
        with pytest.raises(ValueError, match="outside the selector"):
            AlwaysSelector(0)(_X(4, 2), targets=[1])

    def test_repr_is_informative(self):
        text = repr(AlwaysSelector(2))
        assert "AlwaysSelector" in text and "(2,)" in text


class TestClassSelector:
    """ClassSelector: per-class fixed-count sampling on a discrete column."""

    def _X_classes(self, n_per_class=10):
        X = _X(2 * n_per_class)
        X[:, 1] = np.repeat([0, 1], n_per_class)
        return X

    def test_satisfies_protocol(self):
        selector = ClassSelector(
            0, class_column=1, proportions={0: 0.5},
            rng=np.random.default_rng(0),
        )
        assert isinstance(selector, Selector)

    def test_exact_counts_per_class(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            0, class_column=1, proportions={0: 0.5, 1: 0.25},
            rng=np.random.default_rng(0),
        )
        mask = selector(X)[0]
        assert mask.shape == (20,)
        assert mask.dtype == bool
        # proportions are shares of the TOTAL row count (20 rows)
        assert mask.sum() == 10 + 5
        assert mask[X[:, 1] == 0].sum() == 10
        assert mask[X[:, 1] == 1].sum() == 5

    def test_classes_absent_from_proportions_are_kept(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            0, class_column=1, proportions={1: 0.25},
            rng=np.random.default_rng(0),
        )
        mask = selector(X)[0]
        assert mask[X[:, 1] == 0].sum() == 0
        assert mask[X[:, 1] == 1].sum() == min(int(0.25 * 20), 10)

    def test_share_clipped_to_class_size(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            0, class_column=1, proportions={0: 1.0},
            rng=np.random.default_rng(0),
        )
        mask = selector(X)[0]
        # int(1.0 * 20) = 20 wanted, but class 0 only has 10 rows
        assert mask.sum() == 10
        assert mask[X[:, 1] == 0].all()

    def test_unknown_class_value_raises(self):
        X = self._X_classes(10)
        selector = ClassSelector(
            0, class_column=1, proportions={7: 0.5},
            rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match="not found in column"):
            selector(X)

    def test_proportions_out_of_range_raise(self):
        with pytest.raises(ValueError, match="proportions"):
            ClassSelector(0, class_column=1, proportions={0: 1.5},
                          rng=np.random.default_rng(0))

    def test_class_column_is_read_from_X(self):
        X = _X(10)
        X[:, 1] = np.repeat([0, 1], 5)
        selector = ClassSelector(0, class_column=1, proportions={0: 0.4},
                                 rng=np.random.default_rng(0))
        mask = selector(X)[0]
        assert mask.sum() == min(int(0.4 * 10), 5) == 4
        assert mask[X[:, 1] == 0].sum() == 4
        assert mask[X[:, 1] == 1].sum() == 0

    def test_calling_without_rng_raises(self):
        selector = ClassSelector(0, class_column=1, proportions={0: 0.5})
        with pytest.raises(ValueError, match="random generator"):
            selector(self._X_classes(10))

    def test_non_int_class_column_raises(self):
        with pytest.raises(ValueError, match="class_column must be an int"):
            ClassSelector(0, class_column="y", proportions={0: 0.5})

    def test_repr_is_informative(self):
        selector = ClassSelector(0, class_column=1, proportions={0: 0.5})
        text = repr(selector)
        assert "ClassSelector" in text and "class_column=1" in text


class TestProportionalSelector:
    def _X(self):
        return np.array([1.0, 2.0, 3.0, 6.0]).reshape(-1, 1)

    def test_scores_are_normalized_column_values(self):
        selector = ProportionalSelector(0, column=0)
        scores = selector(self._X())[0]
        # min-max to [0, 1], then normalized to sum 1
        np.testing.assert_allclose(scores, [0.0, 0.2 / 1.6, 0.4 / 1.6, 1.0 / 1.6])
        assert scores.shape == (4,)
        assert np.issubdtype(scores.dtype, np.floating)
        assert scores.sum() == pytest.approx(1.0)

    def test_fulfills_protocol(self):
        assert isinstance(ProportionalSelector(0, column=0), Selector)

    def test_constant_column_gives_uniform_scores(self):
        X = np.ones((5, 2))
        selector = ProportionalSelector(0, column=0)
        scores = selector(X)[0]
        assert scores.shape == (5,)
        assert scores.sum() == pytest.approx(1.0)
        assert not (scores < 0).any() and not (scores > 1).any()

    def test_bound_column_drives_all_targets(self):
        X = np.arange(8.0).reshape(4, 2)
        selector = ProportionalSelector([0, 1], column=0)
        out = selector(X)
        assert set(out) == {0, 1}
        # one read column: identical scores object under every target
        assert out[0] is out[1]
        np.testing.assert_allclose(out[0], minmax_normalize(X[:, 0]))

    def test_per_target_read_columns(self):
        X = np.arange(8.0).reshape(4, 2)
        for col in (0, 1):
            selector = ProportionalSelector([0, 1], column=col)
            out = selector(X)
            np.testing.assert_allclose(out[0], minmax_normalize(X[:, col]))
            np.testing.assert_allclose(out[1], minmax_normalize(X[:, col]))

    def test_non_int_column_raises(self):
        with pytest.raises(ValueError, match="column must be an int"):
            ProportionalSelector(0, column="nope")

    def test_repr_is_informative(self):
        selector = ProportionalSelector(0, column=0)
        assert "ProportionalSelector" in repr(selector) and "column=0" in repr(selector)


def minmax_normalize(values):
    span = values.max() - values.min()
    unit = (values - values.min()) / span if span > 0 else np.ones_like(values)
    return unit / unit.sum()


class _FakeSel:
    columns = (0, 1)

    def __call__(self, X, targets=None):
        targets = _resolve_targets(self, X, targets)
        return {c: 1.0 for c in targets}


class TestBaseContract:
    def test_validate_columns_ints(self):
        assert _validate_columns([2, 0]) == (2, 0)
        assert _validate_columns(3) == (3,)

    @pytest.mark.parametrize("bad", [[-1], [0, 0], [], ["a"]])
    def test_validate_columns_rejects(self, bad):
        with pytest.raises(ValueError):
            _validate_columns(bad)

    def test_resolve_targets_defaults_to_own_columns(self):
        X = np.zeros((4, 3))
        assert _resolve_targets(_FakeSel(), X, None) == (0, 1)

    def test_resolve_targets_subset_ok(self):
        X = np.zeros((4, 3))
        assert _resolve_targets(_FakeSel(), X, [1]) == (1,)

    def test_resolve_targets_out_of_own_scope_raises(self):
        X = np.zeros((4, 3))
        with pytest.raises(ValueError, match="outside the selector"):
            _resolve_targets(_FakeSel(), X, [2])

    def test_resolve_targets_out_of_range_raises(self):
        X = np.zeros((4, 3))
        with pytest.raises(ValueError, match="out of range"):
            _resolve_targets(_FakeSel(), X, [7])

    def test_check_spec_accepts_scalar_and_array(self):
        _check_spec(0.3, 5)
        _check_spec(np.zeros(5, dtype=bool), 5)
        _check_spec(np.array([0.0, 1.0, 0.5, 1.0, 0.0]), 5)

    @pytest.mark.parametrize("bad", [2.0, -0.1, np.zeros(3), np.zeros((5, 1))])
    def test_check_spec_rejects(self, bad):
        with pytest.raises(ValueError):
            _check_spec(bad, 5)

    def test_realize_firing_deterministic(self):
        X = np.zeros((6, 2))
        fired = realize_firing(X, _FakeSel(), default_rng(0))
        assert fired.shape == (6, 2)
        assert fired.all()  # spec 1.0 -> rng < 1 always

    def test_realize_firing_probabilities(self):
        sel = lambda X, targets=None: {0: 0.0, 1: 0.25}  # noqa: E731
        fired = realize_firing(np.zeros((1000, 2)), sel, default_rng(0))
        assert not fired[:, 0].any()
        assert 0.2 < fired[:, 1].mean() < 0.3

    def test_fired_mask_is_row_or(self):
        sel = lambda X, targets=None: {0: 0.0, 1: 1.0}  # noqa: E731
        fired = fired_mask(np.zeros((4, 2)), sel, default_rng(0))
        assert fired.tolist() == [True, True, True, True]

    def test_realize_firing_selector_none(self):
        fired = realize_firing(np.zeros((3, 2)), None, default_rng(0))
        assert fired.shape == (3, 2) and fired.all()


class TestValueFree:
    def test_always_uniform_spec_per_target(self):
        sel = AlwaysSelector([0, 2])
        out = sel(np.zeros((4, 3)))
        assert set(out) == {0, 2}
        assert all(v == 1.0 for v in out.values())

    def test_always_targets_subset(self):
        sel = AlwaysSelector([0, 1, 2])
        assert set(sel(np.zeros((2, 3)), targets=[1])) == {1}

    def test_always_out_of_scope_raises(self):
        with pytest.raises(ValueError, match="outside the selector"):
            AlwaysSelector([0])(np.zeros((2, 3)), targets=[2])

    def test_bernoulli_shape_and_rate(self):
        sel = BernoulliSelector([0, 1], rate=0.5, rng=default_rng(0))
        out = sel(np.zeros((1000, 2)))
        for mask in out.values():
            assert mask.shape == (1000,) and mask.dtype == bool
        assert 0.4 < out[0].mean() < 0.6

    def test_bernoulli_independent_per_column(self):
        sel = BernoulliSelector([0, 1], rate=0.5, rng=default_rng(1))
        out = sel(np.zeros((2000, 2)))
        assert out[0] is not out[1]
        assert (out[0] != out[1]).mean() > 0.2  # independent draws disagree

    def test_bernoulli_no_rng_raises(self):
        with pytest.raises(ValueError, match="no random generator"):
            BernoulliSelector([0], rate=0.5)(np.zeros((3, 1)))

    def test_exact_count_exact_per_column(self):
        sel = ExactCountSelector([0, 1, 2], rate=0.25, rng=default_rng(0))
        out = sel(np.zeros((20, 3)))
        assert [m.sum() for m in out.values()] == [5, 5, 5]

    def test_exact_count_weights(self):
        n, k = 10, 2
        w = np.zeros((n, k)); w[0] = [1.0, 0.0]; w[1] = [0.0, 1.0]
        sel = ExactCountSelector([0, 1], rate=0.1, weights=w, rng=default_rng(0))
        out = sel(np.zeros((n, k)))
        assert out[0][0] and out[1][1]

    def test_exact_count_bad_weights_shape(self):
        sel = ExactCountSelector([0, 1], rate=0.1, weights=np.zeros(10), rng=default_rng(0))
        with pytest.raises(ValueError, match="weights must have shape"):
            sel(np.zeros((10, 2)))


class TestValueDriven:
    X4 = np.array([[0.0, 1, 0], [1.0, 1, 0], [2.0, 1, 1], [9.0, 2, 1]])

    def test_class_selector_fires_share_of_class(self):
        # class column = 1, values {0, 1, 2}; fire half of the total from class 1
        sel = ClassSelector([0], class_column=1,
                            proportions={1: 0.5}, rng=default_rng(0))
        out = sel(self.X4)
        assert set(out) == {0}
        mask = out[0]
        assert mask.sum() == 2  # int(0.5 * 4)
        # fired rows must belong to class 1 (rows 0,1)
        assert mask[2:].all() == False and self.X4[mask, 1].tolist() == [1, 1]

    def test_class_selector_unknown_class_raises(self):
        sel = ClassSelector([0], class_column=1, proportions={7: 0.1},
                            rng=default_rng(0))
        with pytest.raises(ValueError, match="class value"):
            sel(self.X4)

    def test_class_selector_broadcasts_to_all_targets(self):
        sel = ClassSelector([0, 2], class_column=1,
                            proportions={1: 0.5}, rng=default_rng(0))
        out = sel(self.X4)
        assert out[0] is out[2]  # same mask object under every target

    def test_top_quantile_self(self):
        sel = TopQuantileSelector([0], column=0, quantile=0.5)
        mask = sel(self.X4)[0]
        # values 0,1,2,9 -> Q2 = 1.5 -> rows 2,3 fire
        assert mask.tolist() == [False, False, True, True]

    def test_top_quantile_driver(self):
        sel = TopQuantileSelector([0], column=0, quantile=1.0, driver=1)
        mask = sel(self.X4)[0]
        # driver col 1 max = 2 -> row 3 only
        assert mask.tolist() == [False, False, False, True]

    def test_proportional_sums_to_one(self):
        sel = ProportionalSelector([0, 1], column=0)
        out = sel(self.X4)
        scores = out[0]
        assert abs(scores.sum() - 1.0) < 1e-9
        assert out[0] is out[1]
        assert scores.tolist()[0] == 0.0  # min value -> 0 prob
