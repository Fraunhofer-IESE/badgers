from unittest import TestCase

import numpy as np

from badgers.generators.time_series.noise import GlobalGaussianNoiseGenerator, LocalGaussianNoiseGenerator, NoiseGenerator





class TestLocalGaussianNoiseGenerator(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = LocalGaussianNoiseGenerator()
        Xt, _ = generator.generate(self.X.reshape(-1, 1), None)


class TestGlobalGaussianNoiseGenerator(TestCase):

    def setUp(self) -> None:
        t = np.linspace(1, 10, 101)
        self.X = (np.sin(t * 2 * np.pi) + 0.5).reshape(-1, 1)

    def test_generate(self):
        generator = GlobalGaussianNoiseGenerator()
        Xt, _ = generator.generate(self.X.reshape(-1, 1), None)
