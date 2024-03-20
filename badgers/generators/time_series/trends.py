import abc
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin


class TrendsGenerator(GeneratorMixin):
    """
    Base class for transformers that generate trends in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: a random number generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class GlobalAdditiveLinearTrendGenerator(TrendsGenerator):
    """
    Add a linear trend to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0), slope: float = 0.1):
        super().__init__(random_generator=random_generator)
        self.slope = slope

    def generate(self, X, y, **params) -> Tuple:
        """

        :param X:
        :param y:
        :param params:
        :return:
        """
        trend = self.slope * np.linspace(0, 1, len(X))
        Xt = X + trend
        return Xt, y
