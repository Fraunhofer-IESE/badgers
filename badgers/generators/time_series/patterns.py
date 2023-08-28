import abc
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin


class PatternsGenerator(GeneratorMixin):
    """
    Base class for transformers that generate patterns in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0), n_patterns: int = 10):
        """
        :param random_generator: a random number generator
        :param n_patterns: the number of patterns to generate
        """
        self.random_generator = random_generator
        self.n_patterns = n_patterns
        self.patterns_indices_ = []

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class RandomConstantPatterns(PatternsGenerator):
    """
    Randomly generates constant patterns of constant value
    """

    def __init__(self, random_generator=default_rng(seed=0), n_patterns: int = 10, patterns_width: int = 10,
                 constant_value: float = 0):
        super().__init__(random_generator=random_generator, n_patterns=n_patterns)
        self.patterns_width = patterns_width
        self.constant_value = constant_value

    def generate(self, X, y, **params) -> Tuple:
        # TODO input validation!
        if X.ndim < 2:
            raise ValueError(
                "Expected 2D array. "
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )
        # generate extreme values indices and values
        self.patterns_indices_ = [(x, x + self.patterns_width) for x in
                                  self.random_generator.choice(X.shape[0] - self.patterns_width, size=self.n_patterns,
                                                               replace=False, p=None)]

        for (start, end) in self.patterns_indices_:
            X[start:end, :] = self.constant_value

        return X, y


class RandomLinearPatterns(PatternsGenerator):
    """
        Randomly generates patterns of constant slope (linear)
        """

    def __init__(self, random_generator=default_rng(seed=0), n_patterns: int = 10, patterns_width: int = 10):
        super().__init__(random_generator=random_generator, n_patterns=n_patterns)
        self.patterns_width = patterns_width

    def generate(self, X, y, **params) -> Tuple:
        # TODO input validation!
        if X.ndim < 2:
            raise ValueError(
                "Expected 2D array. "
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )
        # generate extreme values indices and values
        self.patterns_indices_ = [(x, x + self.patterns_width) for x in
                                  self.random_generator.choice(X.shape[0] - self.patterns_width, size=self.n_patterns,
                                                               replace=False, p=None)]

        for (start, end) in self.patterns_indices_:
            for col in range(X.shape[1]):
                X[start:end, col] = np.linspace(X[start,col],X[end,col], end - start)

        return X, y
