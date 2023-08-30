from unittest import TestCase

import numpy as np

from badgers.generators.time_series.patterns import RandomConstantPatterns, RandomLinearPatterns


class TestRandomConstantPatterns(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = RandomConstantPatterns(n_patterns=3, patterns_width=5, constant_value=0)
        Xt, _ = generator.generate(self.X, None)
        for start, end in generator.patterns_indices_:
            self.assertListEqual(Xt[start:end, :].tolist(), np.zeros((end - start, self.X.shape[1])).tolist())


class TestRandomLinearPatterns(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = RandomLinearPatterns(n_patterns=3, patterns_width=5)
        Xt, _ = None, None
        for (start, end) in generator.patterns_indices_:
            for col in range(self.X.shape[1]):
                self.assertListEqual(Xt[start:end, col].tolist(),
                                     np.linspace(self.X[start, col], self.X[end, col], end - start).tolist())
