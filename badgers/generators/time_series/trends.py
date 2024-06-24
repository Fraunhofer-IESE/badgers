import abc
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


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

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, slope) -> Tuple:
        """

        :param X:
        :param y:
        :param slope:
        :return:
        """

        t = np.linspace(0, 1, len(X))
        trend = t[:, np.newaxis] * slope
        Xt = X.add(trend, axis=0)
        return Xt, y
