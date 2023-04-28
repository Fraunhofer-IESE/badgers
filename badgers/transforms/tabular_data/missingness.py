import abc

import numpy as np
import numpy.random
import pandas as pd
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from badgers.transforms.tabular_data.utils import normalize_proba


class MissingValueTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for missing values transformer
    """
    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param percentage_missing: int, default 10
            The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        assert 0 <= percentage_missing <= 100
        self.percentage_missing = percentage_missing
        self.random_generator = random_generator

    @abc.abstractmethod
    def fit(self, X, y=None, **fit_params):
        """
        Generates a list of indices for the missing values (`self.missing_values_indices_`).
        The list of indices takes the form of a list of tuples (or list of list)
        each tuple containing the row index and the column index of the missing values.



        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        pass

    def fit_transform(self, X, y=None, **fit_params):
        """

        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        self.fit(X, y=None)
        return self.transform(X)

    def transform(self, X):
        """

        :param X: {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        :return X_transformed: array, shape (n_samples, n_features)
            The array containing missing values.
        """
        check_is_fitted(self, ["missing_values_indices_"])
        X = check_array(X, accept_sparse=False)

        # generate missing values
        for row, col in self.missing_values_indices_:
            X[row, col] = np.nan
        return X


class MissingCompletelyAtRandom(MissingValueTransformer):

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """ A transformer that removes values completely at random (MCAR [1]) (uniform distribution over all data).

        See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html

        :param percentage_missing: int, default 10
            The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    def fit(self, X, y=None, **fit_param):
        """
        Computes indices of missing values using a uniform distribution.

        :param X:
        :param y:
        :param fit_param:
        :return:
        """
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        # compute number of missing values per column
        nb_missing = int(X.shape[0] * self.percentage_missing / 100)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=None)
            self.missing_values_indices_ += [(row, col) for row in rows]

        return self


class DummyMissingAtRandom(MissingValueTransformer):
    """

    """

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """

        :param percentage_missing: int, default 10
            The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, defaut default_rng(seed=0)
            A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    def fit(self, X, y=None, **fit_params):
        """

        :param self:
        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # initialize probability with zeros
        p = np.zeros_like(X)
        # normalize values between 0 and 1
        X_norm = (X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))
        # make columns i depends on all the other
        if X.shape[1] > 1:
            for i in range(X.shape[1]):
                p[:, i] = np.delete(X_norm, i, axis=1).sum(axis=1)
        else:
            p = X_norm
        p = normalize_proba(p)

        # compute number of missing values per column
        nb_missing = int(X.shape[0] * self.percentage_missing / 100)
        # generate missing values indices
        self.missing_values_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_missing, replace=False, p=p[:,col])
            self.missing_values_indices_ += [(row, col) for row in rows]

        return self


class DummyMissingNotAtRandom(MissingValueTransformer):

    def __init__(self, percentage_missing: int = 10, random_generator=default_rng(seed=0)):
        """

        :param percentage_missing: int, default 10
            The percentage of missing values (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        super().__init__(percentage_missing=percentage_missing, random_generator=random_generator)

    def fit(self, X, y=None, **fit_params):
        """

        :param self:
        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
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

        return self
