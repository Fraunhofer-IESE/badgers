from unittest import TestCase

import numpy as np

from badgers.generators.time_series.patterns import RandomConstantPatterns, RandomLinearPatterns, Pattern, add_offset, \
    add_linear_trend, scale


class TestPattern(TestCase):

    def test_resample(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        resampled = pattern.resample(10)
        self.assertEqual(len(resampled), 10)
        self.assertAlmostEqual(resampled[0], 1)
        self.assertAlmostEqual(resampled[-1], 5)

    def test_add_offset(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        transformed_pattern = add_offset(pattern.values, 2)
        self.assertListEqual(transformed_pattern.tolist(), [3, 4, 5, 6, 7])

    def test_add_linear_trend(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        transformed_pattern = add_linear_trend(pattern.values, start_value=0, end_value=-1)
        self.assertEqual(len(transformed_pattern), 5)
        self.assertEqual(transformed_pattern[0], 0)
        self.assertEqual(transformed_pattern[-1], -1)

    def test_scale(self):
        pattern = Pattern(np.array([1, 2, 3, 4, 5]))
        transformed_pattern = scale(pattern.values, scaling_factor=2)
        self.assertListEqual(transformed_pattern.tolist(), [2, 4, 6, 8, 10])


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
