import abc

import numpy as np
import numpy.random
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.utils import normalize_proba


class MissingValueGenerator(GeneratorMixin):
    """
    Base class for missing values transformer
    """

    def __init__(self, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
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

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """
        Computes indices of missing values using a uniform distribution.

        :param X: the input features
        :param y: the target
        :param percentage_missing: The percentage of missing values (float value between 0 and 1 included)
        :return: Xt, yt
        """
        assert 0 <= percentage_missing <= 1
        # compute number of missing values per column
        nb_missing = int(X.shape[0] * percentage_missing)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=None)
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X.iloc[rows, col] = np.nan

        return X, y


class DummyMissingAtRandom(MissingValueGenerator):
    """
    A generator that removes values at random (MAR [1]),
    where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
    where j is randomly chosen.

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """

        :param X: the input features
        :param y: the target
        :param percentage_missing: The percentage of missing values (float value between 0 and 1 included)
        :return: Xt, yt
        """
        assert 0 <= percentage_missing <= 1
        # initialize probability with zeros
        p = np.zeros_like(X)
        # normalize values between 0 and 1
        X_norm = ((X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))).values
        # make columns i depends on all the other
        if X.shape[1] > 1:
            for i in range(X.shape[1]):
                j = self.random_generator.choice([x for x in range(X.shape[1]) if x != i])
                p[:, i] = X_norm[:, j]
        else:
            p = X_norm
        p = normalize_proba(p)

        # compute number of missing values per column
        nb_missing = int(X.shape[0] * percentage_missing)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=p[:, col])
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X.iloc[rows, col] = np.nan

        return X, y


class DummyMissingNotAtRandom(MissingValueGenerator):
    """
    A generator that removes values not at random (MNAR [1]),
    where the probability of a data instance X[i,j] missing depends linearly upon its own value.
    A data point X[i,j] = max(X[:,j]) has a missing probability of 1.
    A data point X[i,j] = min(X[:,j]) has a missing probability of 0.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing):
        """

        :param X: the input features
        :param y: the target
        :param percentage_missing: The percentage of missing values (float value between 0 and 1 included)
        :return: Xt, yt
        """
        assert 0 <= percentage_missing <= 1

        # normalize values between 0 and 1
        p = ((X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))).values
        # make the sum of each column = 1
        p = normalize_proba(p)

        # compute number of missing values per column
        nb_missing = int(X.shape[0] * percentage_missing)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=p[:, col])
            self.missing_values_indices_ += [(row, col) for row in rows]
            # generate missing values
            X.iloc[rows, col] = np.nan

        return X, y
