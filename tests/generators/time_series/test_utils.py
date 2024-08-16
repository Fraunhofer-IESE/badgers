from unittest import TestCase

import numpy as np

from badgers.generators.time_series.utils import generate_random_patterns_indices


class TestGenerateRandomPatternsIndices(TestCase):
    def test_no_patterns(self):
        random_generator = np.random.default_rng(0)
        with self.assertRaises(AssertionError):
            generate_random_patterns_indices(
                random_generator=random_generator,
                signal_size=100,
                n_patterns=0,
                min_width_pattern=5,
                max_width_patterns=10
            )

    def test_single_pattern(self):
        random_generator = np.random.default_rng(0)
        signal_size = 100
        n_patterns = 1
        min_width_pattern = 5
        max_width_patterns = 10
        patterns_indices = generate_random_patterns_indices(
            random_generator=random_generator,
            signal_size=signal_size,
            n_patterns=n_patterns,
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns
        )

        self.assertEqual(len(patterns_indices), n_patterns)
        for start, end in patterns_indices:
            self.assertTrue(start < end)
            self.assertTrue(end - start >= min_width_pattern)
            self.assertTrue(end - start < max_width_patterns)
            self.assertTrue(0 <= start < signal_size)
            self.assertTrue(0 <= end < signal_size)

    def test_generate_random_patterns_indices(self):
        random_generator = np.random.default_rng(0)
        signal_size = 100
        n_patterns = 5
        min_width_pattern = 5
        max_width_patterns = 10
        patterns_indices = generate_random_patterns_indices(
            random_generator=random_generator,
            signal_size=signal_size,
            n_patterns=n_patterns,
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns
        )

        self.assertEqual(len(patterns_indices), n_patterns)
        for start, end in patterns_indices:
            self.assertTrue(start < end)
            self.assertTrue(end - start >= min_width_pattern)
            self.assertTrue(end - start < max_width_patterns)
            self.assertTrue(0 <= start < signal_size)
            self.assertTrue(0 <= end < signal_size)
