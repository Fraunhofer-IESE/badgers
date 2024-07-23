import abc
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


class MissingValuesGenerator(GeneratorMixin):
    """
    Base class for transformers that generate point outliers in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: a random number generator
        :param n_outliers: the number of outliers to generate
        """
        self.random_generator = random_generator
        self.missing_indices_ = []

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class MissingAtRandomGenerator(MissingValuesGenerator):
    """
    Randomly set data points to nan (missing)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: a random number generator

        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_missing: int = 10) -> Tuple:
        """
        Randomly set values to np.nan (missing)
        :param X:
        :param y:
        :param n_missing: the number of outliers to generate
        :return:
        """
        # generate missing values indices and values
        rows = self.random_generator.choice(X.shape[0], size=n_missing, replace=False, p=None)
        cols = self.random_generator.integers(low=0, high=X.shape[1], size=n_missing)

        self.missing_indices_ = list(zip(rows, cols))

        for r, c in self.missing_indices_:
            X.iloc[r, c] = np.nan

        return X, y
