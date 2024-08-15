import unittest

import pandas as pd
from numpy.random import default_rng

from badgers.generators.time_series.transmission_errors import RandomTimeSwitchGenerator, RandomRepeatGenerator


class TestRandomTimeSwitchGenerator(unittest.TestCase):

    def test_no_switch(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomTimeSwitchGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_switches=0)

        # nothing should have happened (n_switch = 0)
        pd.testing.assert_frame_equal(X, Xt)

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
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=0, min_nb_repeats=2, max_nb_repeats=3)
        pd.testing.assert_frame_equal(X, Xt)

    def test_two_repeats(self):
        X = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        generator = RandomRepeatGenerator(random_generator=default_rng(seed=0))
        Xt, _ = generator.generate(X.copy(), y=None, n_repeats=2, min_nb_repeats=2, max_nb_repeats=3)
        self.assertEqual(Xt.shape[0], X.shape[0] + 2 * 2)
        self.assertSetEqual(set(X[0]), set(Xt[0]))


if __name__ == '__main__':
    unittest.main()
