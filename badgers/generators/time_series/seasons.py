import abc

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs

class SeasonsGenerator(GeneratorMixin):
    """
    Base class for transformers that generate seasons in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random number generator instance used for generating random numbers.
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Generates seasonal patterns in the input time-series data.

        :param X: Input features (time-series data).
        :param params: Additional parameters that may be required for generating seasons.
        :return: The modified input features with the seasonal pattern applied.
        """
        pass


class GlobalAdditiveSinusoidalSeasonGenerator(SeasonsGenerator):
    """
    Add a sinusoidal season to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, period: int = 10):
        """
        Adds a global sinusoidal seasonal pattern to the input time-series data.

        :param X: Input features (time-series data). Expected to be a 2D numpy array where each row represents a time step.
        :param period: The period of the sinusoidal season. Determines the length of one complete cycle of the sinusoidal wave.
        :return: The modified input features with the added sinusoidal season.
        """
        t = np.arange(len(X))
        season = np.sin(t[:,np.newaxis]*2*np.pi/period)
        Xt = X + season
        return Xt
