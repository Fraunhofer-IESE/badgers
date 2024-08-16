import unittest

import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.transmission_errors import RandomTimeSwitchGenerator, RandomRepeatGenerator, \
    RandomDropGenerator, LocalRegionsRandomDropGenerator, LocalRegionsRandomRepeatGenerator


class TestRandomTimeSwitchGenerator(unittest.TestCase):

    def test_no_switch(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
        with self.assertRaises(AssertionError):
            generator.generate(X.copy(), y=None, n_switches=0)

    def test_single_switch(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)

        # assert same values in both arrays
        self.assertSetEqual(set(X[0]), set(Xt[0]))
        # assert only 2 values differs (n_switches = 1)
        self.assertEqual((X != Xt).sum().values[0], 2)

    def test_single_switch_frame(self):
        X = pd.DataFrame(
            data=[
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9]
            ]
        )
        generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_switches=1)

        # assert same values in both arrays
        self.assertSetEqual(set(X[0]), set(Xt[0]))
        # assert only 2 values differs (n_switches = 1)
        self.assertEqual((X != Xt).sum().values[0], 2)


class TestRandomRepeatGenerator(unittest.TestCase):

    def test_no_repeat(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
        with self.assertRaises(AssertionError):
            generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)

    def test_two_repeats(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=2, min_nb_repeats=2, max_nb_repeats=3)
        self.assertEqual(Xt.shape[0], X.shape[0] + 2 * 2)
        self.assertSetEqual(set(X[0]), set(Xt[0]))


class TestLocalRegionsRandomRepeatGenerator(unittest.TestCase):

    def test_no_repeat(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
        with self.assertRaises(AssertionError):
            generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)

    def test_single_repeat_single_region(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=1, min_nb_repeats=2, max_nb_repeats=3, n_regions=1,
                                   min_width_regions=3, max_width_regions=7)
        self.assertEqual(Xt.shape[0], X.shape[0] + 2)
        self.assertSetEqual(set(X[0]), set(Xt[0]))

    def test_many_repeats_single_region(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=3, min_nb_repeats=2, max_nb_repeats=3, n_regions=1,
                                   min_width_regions=3, max_width_regions=7)
        self.assertEqual(Xt.shape[0], X.shape[0] + 2 * 3)
        self.assertSetEqual(set(X[0]), set(Xt[0]))

    def test_many_repeats_many_regions(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomRepeatGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=4, min_nb_repeats=2, max_nb_repeats=3, n_regions=2,
                                   min_width_regions=3, max_width_regions=5)
        self.assertEqual(Xt.shape[0], X.shape[0] + 2 * 4)
        self.assertSetEqual(set(X[0]), set(Xt[0]))


class TestRandomDrop(unittest.TestCase):

    def test_no_drop(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomDropGenerator(random_generator=default_rng(0))
        with self.assertRaises(AssertionError):
            generator.generate(X.copy(), y=None, n_drops=0)

    def test_single_drop(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomDropGenerator(random_generator=default_rng(0))
        Xt, _ = generator.generate(X.copy(), y=None, n_drops=1)
        self.assertEqual(X.shape[0], Xt.shape[0] + 1)


class TestLocalRegionsRandomDropGenerator(unittest.TestCase):

    def test_no_drop(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
        with self.assertRaises(AssertionError):
            generator.generate(X.copy(), y=None, n_drops=0, n_regions=0)

    def test_single_drop(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
        Xt, _ = generator.generate(X.copy(), y=None, n_drops=1, n_regions=1, min_width_regions=2, max_width_regions=4)
        self.assertEqual(X.shape[0], Xt.shape[0] + 1)

    def test_many_drops_single_region(self):
        X = pd.DataFrame(range(100))
        generator = LocalRegionsRandomDropGenerator(random_generator=default_rng(0))
        Xt, _ = generator.generate(X.copy(), y=None, n_drops=5, n_regions=1, min_width_regions=10, max_width_regions=20)
        self.assertEqual(X.shape[0], Xt.shape[0] + 5)


if __name__ == '__main__':
    unittest.main()
