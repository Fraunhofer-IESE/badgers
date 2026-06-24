import numpy as np

from badgers.generators.time_series.noise import GlobalGaussianNoiseGenerator, LocalGaussianNoiseGenerator


def test_local_gaussian_noise__generates(time_series_sine):
    """LocalGaussianNoiseGenerator runs without error on sine wave."""
    X, _ = time_series_sine
    generator = LocalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None)


def test_global_gaussian_noise__generates(time_series_sine):
    """GlobalGaussianNoiseGenerator runs without error on sine wave."""
    X, _ = time_series_sine
    generator = GlobalGaussianNoiseGenerator()
    Xt, _ = generator.generate(X, None)
