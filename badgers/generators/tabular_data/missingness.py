import abc

import numpy as np
import numpy.random
import pandas as pd
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators import numpy_API
from badgers.core.utils import normalize_proba


class MissingValueGenerator(GeneratorMixin):
    """
    Base class for missing values transformer
    """

    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param percentage_missing: The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: A random generator
        """
        assert 0 <= percentage_missing <= 100
        self.percentage_missing = percentage_missing
        self.random_generator = random_generator
        self.missing_values_indices_ = None

    @abc.abstractmethod
    def generate(self, X, y, **params):
        pass


class MissingCompletelyAtRandom(MissingValueGenerator):
    """
    A generator that removes values completely at random (MCAR [1]) (uniform distribution over all data).

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html
    """

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """

        :param percentage_missing: The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    @numpy_API
    def generate(self, X, y, **params):
        """
        Computes indices of missing values using a uniform distribution.

        :param X: the input features
        :param y: the target
        :return: Xt, yt
        """
        # compute number of missing values per column
        nb_missing = int(X.shape[0] * self.percentage_missing / 100)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=None)
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X[rows, col] = np.nan

        return X, y


class DummyMissingAtRandom(MissingValueGenerator):
    """
    A generator that removes values at random (MAR [1]),
    where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
    where j is randomly chosen.

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html
    """

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """

        :param percentage_missing: The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    @numpy_API
    def generate(self, X, y, **params):
        """

        :param X: the input features
        :param y: the target
        :return: Xt, yt
        """
        # initialize probability with zeros
        p = np.zeros_like(X)
        # normalize values between 0 and 1
        X_norm = (X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))
        # make columns i depends on all the other
        if X.shape[1] > 1:
            for i in range(X.shape[1]):
                j = self.random_generator.choice([x for x in range(X.shape[1]) if x != i])
                p[:, i] = X_norm[:, j]
        else:
            p = X_norm
        p = normalize_proba(p)

        # compute number of missing values per column
        nb_missing = int(X.shape[0] * self.percentage_missing / 100)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=p[:, col])
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X[rows, col] = np.nan

        return X, y


class DummyMissingNotAtRandom(MissingValueGenerator):
    """
    A generator that removes values not at random (MNAR [1]),
    where the probability of a data instance X[i,j] missing depends linearly upon its own value.
    A data point X[i,j] = max(X[:,j]) has a missing probability of 1.
    A data point X[i,j] = min(X[:,j]) has a missing probability of 0.
    """

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """

        :param percentage_missing: The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    @numpy_API
    def generate(self, X, y, **params):
        """

        :param X: the input features
        :param y: the target
        :return: Xt, yt
        """

        # normalize values between 0 and 1
        p = (X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))
        # make the sum of each column = 1
        p = normalize_proba(p)

        # compute number of missing values per column
        nb_missing = int(X.shape[0] * self.percentage_missing / 100)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=p[:, col])
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X[rows, col] = np.nan

        return X, y
