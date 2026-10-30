import numpy as np
import pytest

from badgers.generators.tabular_data.missingness.missingness import (
    DummyMissingAtRandom,
    DummyMissingNotAtRandom,
    MissingCompletelyAtRandom,
    MissingValueGenerator,
)

ALL_GENERATORS = [
    MissingCompletelyAtRandom,
    DummyMissingAtRandom,
    DummyMissingNotAtRandom,
]

# fixed scenario for the mechanism-oracle tests: strong signal, fixed seed
N_ROWS = 300
PERCENTAGE = 0.3
MAR_GAP_MARGIN = 0.12  # clean-source driver coupling is 0.20-0.29; uniform noise is < 0.1
# end-to-end draw check: top vs bottom-half selection-rate gap
# clean-source behavior at seed 0 is ~ +0.07 .. +0.13; noise-only draws
# (mechanism weights ignored) give ~ -0.03 .. +0.04 per column
MNAR_GAP_MARGIN = 0.05


def _mechanism_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_ROWS, 3))


def _tercile_nan_rate_gap(Xt, X, target_col, driver_col):
    """NaN rate of the top driver tercile minus the bottom tercile."""
    nan = np.isnan(Xt[:, target_col])
    vals = X[:, driver_col]
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    return nan[vals >= q2].mean() - nan[vals <= q1].mean()


def test_missingness__preserves_shape_and_inserts_nans(tabular_data):
    """All MissingValueGenerator subclasses preserve shape and insert correct NaN count."""
    X = np.asarray(tabular_data)
    n_features = X.shape[1] if X.ndim > 1 else 1
    total_cells = X.shape[0] * n_features
    percentage_missing = 0.1

    for cls in MissingValueGenerator.__subclasses__():
        transformer = cls(random_generator=np.random.default_rng(0))
        Xt = transformer.generate(X.copy(), percentage_missing=percentage_missing)
        assert Xt.shape[1] == n_features
        assert np.sum(np.isnan(Xt).sum()) == int(percentage_missing * total_cells)


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_missingness__same_seed_is_bit_identical(cls):
    """Same seed -> bit-identical NaN mask (regression pin without a golden file)."""
    X = _mechanism_data()
    Xt1 = cls(random_generator=np.random.default_rng(0)).generate(
        X.copy(), percentage_missing=PERCENTAGE)
    Xt2 = cls(random_generator=np.random.default_rng(0)).generate(
        X.copy(), percentage_missing=PERCENTAGE)
    np.testing.assert_array_equal(np.isnan(Xt1), np.isnan(Xt2))
    np.testing.assert_array_equal(Xt1, Xt2)


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_missing_values_indices__exactly_match_nan_positions(cls):
    """``missing_values_indices_`` lists every NaN cell and nothing else,
    ordered column-major."""
    X = _mechanism_data()
    gen = cls(random_generator=np.random.default_rng(0))
    Xt = gen.generate(X.copy(), percentage_missing=PERCENTAGE)

    indices = gen.missing_values_indices_
    assert indices is not None
    assert len(indices) == int(np.isnan(Xt).sum())

    indexed_cells = set(indices)
    nan_cells = {tuple(rc) for rc in np.argwhere(np.isnan(Xt))}
    assert indexed_cells == nan_cells

    cols = [col for _, col in indices]
    assert cols == sorted(cols), "indices must be in column-major order"
    for col in set(cols):
        rows = [row for row, c in indices if c == col]
        assert rows == sorted(rows)


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_missing_values_indices__tracks_most_recent_call(cls):
    """A second generate() overwrites the indices (no accumulation)."""
    X = _mechanism_data()
    gen = cls(random_generator=np.random.default_rng(0))
    gen.generate(X.copy(), percentage_missing=0.2)
    n_first = len(gen.missing_values_indices_)
    gen.generate(X.copy(), percentage_missing=0.2)
    assert len(gen.missing_values_indices_) == n_first


@pytest.mark.parametrize("percentage", [-0.1, 1.1, 2.0])
@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_percentage_missing__out_of_range_raises(cls, percentage):
    X = _mechanism_data()
    gen = cls(random_generator=np.random.default_rng(0))
    with pytest.raises(ValueError, match="percentage_missing"):
        gen.generate(X.copy(), percentage_missing=percentage)


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_percentage_missing__boundaries(cls):
    """percentage_missing=0.0 keeps X unchanged; 1.0 hides every cell."""
    X = _mechanism_data()
    gen = cls(random_generator=np.random.default_rng(0))
    Xt0 = gen.generate(X.copy(), percentage_missing=0.0)
    np.testing.assert_array_equal(Xt0, X)
    Xt1 = gen.generate(X.copy(), percentage_missing=1.0)
    assert np.isnan(Xt1).all()


def test_mar__nan_positions_coupled_to_a_driver_column():
    """MAR: whether a cell goes missing must depend on ANOTHER column's value.

    For every target column, the NaN-rate gap between the extreme terciles of
    its strongest driver must be large (uniform selection gives < 0.1 under
    this seed)."""
    X = _mechanism_data()
    gen = DummyMissingAtRandom(random_generator=np.random.default_rng(0))
    Xt = gen.generate(X.copy(), percentage_missing=PERCENTAGE)

    for target in range(X.shape[1]):
        strongest = max(
            abs(_tercile_nan_rate_gap(Xt, X, target, driver))
            for driver in range(X.shape[1]) if driver != target
        )
        assert strongest > MAR_GAP_MARGIN, (
            f"column {target}: NaN positions look independent of every other "
            f"column (strongest tercile gap {strongest:.3f}) -- not MAR"
        )


def test_mnar__selection_probability_grows_with_own_value():
    """MNAR (docstring): P(missing) grows with the cell's own value -- a cell
    at the column maximum has the highest missing probability, one at the
    minimum the lowest.

    Distribution-free pin on the mechanism weights: the selection
    probability vector must be strictly monotone INCREASING in the cell's
    own value (the docstring fixes the direction explicitly and states a
    linear dependence). Kills the reversed (descending) implementation and
    uniform/absent weights; accepts any increasing linear rule regardless
    of its scale.
    """
    from badgers.generators.tabular_data.missingness.missingness import (
        _mnar_weights,
    )

    X = _mechanism_data()
    p = _mnar_weights(X)
    ranges = X.max(axis=0) - X.min(axis=0)

    # increasing pairs of rows per column: value_b > value_a must imply
    # prob_b > prob_a
    for col in np.flatnonzero(ranges > 0):
        vals = X[:, col]
        order = np.argsort(vals, kind="stable")
        assert np.all(np.diff(p[order, col]) > 0), (
            f"column {col}: selection probability is not strictly increasing "
            f"in the own value -- contradicts the documented MNAR direction"
        )


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c.__name__)
def test_constant_column__selection_stays_uniform_and_silent(cls):
    """A constant (zero-variance) column must not poison the mechanism:
    no divide warning, correct NaN count, NaNs land in every column."""
    X = _mechanism_data()
    X[:, 1] = 5.0
    gen = cls(random_generator=np.random.default_rng(0))
    with np.errstate(invalid="raise"):
        Xt = gen.generate(X.copy(), percentage_missing=PERCENTAGE)
    assert int(np.isnan(Xt).sum()) == int(PERCENTAGE * X.size)
    for col in range(X.shape[1]):
        assert np.isnan(Xt[:, col]).any(), f"column {col} got no NaNs"


def test_mnar__end_to_end_draw_skews_toward_high_values():
    """End-to-end wiring check: with the documented mechanism the actual
    mask must put more NaNs in the top half than in the bottom half, by a
    margin a mechanism-free draw cannot reach (seed-0 behavior measured on
    the clean source)."""
    X = _mechanism_data()
    gen = DummyMissingNotAtRandom(random_generator=np.random.default_rng(0))
    Xt = gen.generate(X.copy(), percentage_missing=PERCENTAGE)

    gaps = []
    for col in range(X.shape[1]):
        nan = np.isnan(Xt[:, col])
        median = np.median(X[:, col])
        gaps.append(nan[X[:, col] >= median].mean()
                    - nan[X[:, col] < median].mean())
    mean_gap = float(np.mean(gaps))
    assert mean_gap > MNAR_GAP_MARGIN, (
        f"NaN placement looks independent of the values (mean top-half minus "
        f"bottom-half gap {mean_gap:.3f}) -- the MNAR mechanism is not "
        f"reaching the draw"
    )
