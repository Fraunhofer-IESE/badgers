import unittest
import numpy as np
import pandas as pd
import networkx as nx
from benchmarks.scenarios.tabular import (
    SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS, SCENARIO_LARGE_BLOBS,
)
from benchmarks.scenarios.time_series import SCENARIO_SINE_WAVE, SCENARIO_RANDOM_WALK
from benchmarks.scenarios.graph import SCENARIO_ERDOS_RENYI
from benchmarks.scenarios.text import SCENARIO_WORD_LIST


class TestTabularScenarios(unittest.TestCase):
    def test_small_blobs(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_SMALL_BLOBS.factory(rng)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100,))

    def test_medium_blobs(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_MEDIUM_BLOBS.factory(rng)
        self.assertEqual(X.shape, (1000, 5))

    def test_large_blobs(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_LARGE_BLOBS.factory(rng)
        self.assertEqual(X.shape, (5000, 10))


class TestTimeSeriesScenarios(unittest.TestCase):
    def test_sine_wave(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_SINE_WAVE.factory(rng)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (200, 1))

    def test_random_walk(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_RANDOM_WALK.factory(rng)
        self.assertEqual(X.shape, (200, 1))


class TestGraphScenarios(unittest.TestCase):
    def test_erdos_renyi(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_ERDOS_RENYI.factory(rng)
        self.assertIsInstance(X, nx.Graph)
        self.assertGreater(X.number_of_nodes(), 0)


class TestTextScenarios(unittest.TestCase):
    def test_word_list(self):
        rng = np.random.default_rng(0)
        X, y = SCENARIO_WORD_LIST.factory(rng)
        self.assertIsInstance(X, list)
        self.assertGreater(len(X), 0)
        self.assertTrue(all(isinstance(w, str) for w in X))