from unittest import TestCase

import numpy as np

from badgers.generators.time_series.patterns import Pattern, add_offset, \
    add_linear_trend, scale, RandomlySpacedConstantPatterns, RandomlySpacedLinearPatterns, RandomlySpacedPatterns


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


class TestRandomlySpacedPatterns(TestCase):

    def test_generate_1D(self):
        X = np.zeros(shape=100).reshape(-1,1)
        pattern = Pattern(values=np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]))
        generator = RandomlySpacedPatterns(n_patterns=3, min_width_pattern=5, max_width_patterns=10, pattern=pattern)
        Xt, _ = generator.generate(X, None)
        # assert that the correct number of patterns have been injected
        self.assertEqual(generator.n_patterns, len(generator.patterns_indices_))
        # assert that no pattern overlap
        self.fail('Implement Test!')

    def test_generate_2D(self):
        X = np.zeros(shape=(100, 2))
        pattern = Pattern(
            values=np.array([[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], [0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0]]).T)
        generator = RandomlySpacedPatterns(n_patterns=3, min_width_pattern=5, max_width_patterns=10, pattern=pattern)
        Xt, _ = generator.generate(X, None)
        # assert that the correct number of patterns have been injected
        self.assertEqual(generator.n_patterns, len(generator.patterns_indices_))
        # assert that no pattern overlap
        self.fail('Implement Test!')


class TestRandomlySpacedConstantPatterns(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = RandomlySpacedConstantPatterns(n_patterns=3, min_width_pattern=5, max_width_patterns=10,
                                                   constant_value=0)
        Xt, _ = generator.generate(self.X, None)
        # assert that the correct number of patterns have been injected
        self.assertEqual(generator.n_patterns, len(generator.patterns_indices_))
        # assert that all patterns are constant = 0
        for start, end in generator.patterns_indices_:
            self.assertListEqual(Xt[start:end, :].tolist(), np.zeros((end - start, self.X.shape[1])).tolist())


class TestRandomlySpacedLinearPatterns(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = RandomlySpacedLinearPatterns(n_patterns=3, min_width_pattern=5, max_width_patterns=10)
        Xt, _ = generator.generate(self.X, None)
        for (start, end) in generator.patterns_indices_:
            for col in range(self.X.shape[1]):
                self.assertListEqual(Xt[start:end, col].tolist(),
                                     np.linspace(self.X[start, col], self.X[end, col], end - start).tolist())
