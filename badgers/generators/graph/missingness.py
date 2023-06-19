import abc
from typing import Tuple

import numpy
from numpy.random import default_rng

from core.base import GeneratorMixin


class MissingNodesTransformer(GeneratorMixin):
    """
    Base class for missing nodes transformer
    """

    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param percentage_missing: int, default 10
            The percentage of missing nodes (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        assert 0 <= percentage_missing <= 100
        self.percentage_missing = percentage_missing
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y=None, **params) -> Tuple:
        pass
