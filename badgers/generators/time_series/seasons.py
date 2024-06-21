import abc
from typing import Tuple

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
        :param random_generator: a random number generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class GlobalAdditiveSinusoidalSeasonGenerator(SeasonsGenerator):
    """
    Add a sinusoidal season to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0), period: int = 10):
        super().__init__(random_generator=random_generator)
        self.period = period

    @preprocess_inputs
    def generate(self, X, y, **params) -> Tuple:
        """

        :param X:
        :param y:
        :param params:
        :return:
        """
        t = np.arange(len(X))
        season = np.sin(t*2*np.pi/self.period)
        Xt = X.add(season, axis=0)
        return Xt, y
