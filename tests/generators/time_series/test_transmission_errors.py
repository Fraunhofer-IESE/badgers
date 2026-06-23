import pandas as pd
import pytest
from numpy.random import default_rng

from badgers.generators.time_series.transmission_errors import RandomTimeSwitchGenerator, RandomRepeatGenerator, \
    RandomDropGenerator, LocalRegionsRandomDropGenerator, LocalRegionsRandomRepeatGenerator


def test_random_time_switch__no_switch_raises():
    """RandomTimeSwitchGenerator raises AssertionError for n_switches=0."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    with pytest.raises(AssertionError):
        generator.generate(X.copy(), y=None, n_switches=0)


def test_random_time_switch__single_switch():
    """RandomTimeSwitchGenerator with 1 switch: same values, 2 positions differ."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)
    assert set(X[0]) == set(Xt[0])
    assert (X != Xt).sum().values[0] == 2


def test_random_time_switch__single_switch_frame():
    """RandomTimeSwitchGenerator with multi-column frame: same values, 2 positions differ."""
    X = pd.DataFrame(data=[
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
        [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9],
    ])
    generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)
    assert set(X[0]) == set(Xt[0])
    assert (X != Xt).sum().values[0] == 2


def test_random_repeat__no_repeat_raises():
    """RandomRepeatGenerator raises AssertionError for n_repeats=0."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
    with pytest.raises(AssertionError):
        generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)


def test_random_repeat__two_repeats():
    """RandomRepeatGenerator with 2 repeats: length increases by 4, same values."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_repeats=2, min_nb_repeats=2, max_nb_repeats=3)
    assert Xt.shape[0] == X.shape[0] + 2 * 2
    assert set(X[0]) == set(Xt[0])


def test_local_regions_random_repeat__no_repeat_raises():
    """LocalRegionsRandomRepeatGenerator raises AssertionError for n_repeats=0."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    with pytest.raises(AssertionError):
        generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)


def test_local_regions_random_repeat__single_repeat_single_region():
    """LocalRegionsRandomRepeatGenerator: 1 repeat, 1 region, length +2."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_repeats=1, min_nb_repeats=2, max_nb_repeats=3,
                               n_regions=1, min_width_regions=3, max_width_regions=7)
    assert Xt.shape[0] == X.shape[0] + 2
    assert set(X[0]) == set(Xt[0])


def test_local_regions_random_repeat__many_repeats_single_region():
    """LocalRegionsRandomRepeatGenerator: 3 repeats, 1 region, length +6."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_repeats=3, min_nb_repeats=2, max_nb_repeats=3,
                               n_regions=1, min_width_regions=3, max_width_regions=7)
    assert Xt.shape[0] == X.shape[0] + 2 * 3
    assert set(X[0]) == set(Xt[0])


def test_local_regions_random_repeat__many_repeats_many_regions():
    """LocalRegionsRandomRepeatGenerator: 4 repeats, 2 regions, length +8."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
    Xt, _ = generator.generate(X.copy(), y=None, n_repeats=4, min_nb_repeats=2, max_nb_repeats=3,
                               n_regions=2, min_width_regions=3, max_width_regions=5)
    assert Xt.shape[0] == X.shape[0] + 2 * 4
    assert set(X[0]) == set(Xt[0])


def test_random_drop__no_drop_raises():
    """RandomDropGenerator raises AssertionError for n_drops=0."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomDropGenerator(random_generator=default_rng(0))
    with pytest.raises(AssertionError):
        generator.generate(X.copy(), y=None, n_drops=0)


def test_random_drop__single_drop():
    """RandomDropGenerator with 1 drop: length decreases by 1."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = RandomDropGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y=None, n_drops=1)
    assert X.shape[0] == Xt.shape[0] + 1


def test_local_regions_random_drop__no_drop_raises():
    """LocalRegionsRandomDropGenerator raises AssertionError for n_drops=0."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
    with pytest.raises(AssertionError):
        generator.generate(X.copy(), y=None, n_drops=0, n_regions=0)


def test_local_regions_random_drop__single_drop():
    """LocalRegionsRandomDropGenerator: 1 drop, length -1."""
    X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y=None, n_drops=1, n_regions=1, min_width_regions=2, max_width_regions=4)
    assert X.shape[0] == Xt.shape[0] + 1


def test_local_regions_random_drop__many_drops_single_region():
    """LocalRegionsRandomDropGenerator: 5 drops, length -5."""
    X = pd.DataFrame(range(100))
    generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
    Xt, _ = generator.generate(X.copy(), y=None, n_drops=5, n_regions=1, min_width_regions=10, max_width_regions=20)
    assert X.shape[0] == Xt.shape[0] + 5
