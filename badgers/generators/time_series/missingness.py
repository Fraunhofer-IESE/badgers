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
        Initialize the MissingValuesGenerator with a given random number generator.

        :param random_generator: An instance of a random number generator from NumPy,
                                 used to introduce randomness in the generation process.
                                 Defaults to a default_rng seeded with 0.
        """
        self.random_generator = random_generator
        self.missing_indices_ = []


class MissingAtRandomGenerator(MissingValuesGenerator):
    """
    Randomly set data points to nan (missing)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the MissingAtRandomGenerator with a given random number generator.

        :param random_generator: An instance of a random number generator from NumPy,
                                 used to introduce randomness in the generation process.
                                 Defaults to a default_rng seeded with 0.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_missing: int = 10) -> Tuple:
        """
        Randomly sets a specified number of values in the input array X to np.nan, representing missing values.

        :param X: A numpy array of shape (n_samples, n_features) containing the input time-series data.
        :param y: A numpy array of shape (n_samples,) containing the target values. This parameter is not modified by this method.
        :param n_missing: The number of missing values to randomly introduce into the data. Defaults to 10.
        :return: A tuple (X_out, y_out) where X_out is the modified array with missing values and y_out is the original target array.
        """
        # generate missing values indices and values
        rows = self.random_generator.choice(X.shape[0], size=n_missing, replace=False, p=None)
        cols = self.random_generator.integers(low=0, high=X.shape[1], size=n_missing)

        self.missing_indices_ = list(zip(rows, cols))

        for r, c in self.missing_indices_:
            X.iloc[r, c] = np.nan

        return X, y
