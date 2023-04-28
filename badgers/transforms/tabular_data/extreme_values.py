import abc

import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class NoiseTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def fit(self, X, y=None, **fit_params):
        """
        Generates a mapping between extreme values indices and the value itself (`extreme_values_mapping_`).
        The mapping must be implemented as a dictionary where:
        - the keys are tuple containing the row index and the column index of the extreme values
        - the values are the extreme values

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
        Replaces values in X with the precomputing extreme values.

        :param X: {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        :return X_transformed: array, shape (n_samples, n_features)
            The array containing missing values.
        """
        check_is_fitted(self, ["extreme_values_mapping_"])
        X = check_array(X, accept_sparse=False)

        # generate missing values
        for (row, col), val in self.extreme_values_mapping_.items():
            X[row, col] = val
        return X


class ZScoreTransformer(NoiseTransformer):
    """
    Randomly generates extreme values
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_extreme_values: int = 10, ):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_extreme_values: int, default 10
            The percentage of extreme values to generate
        """
        super().__init__(random_generator=random_generator)
        assert 0 <= percentage_extreme_values <= 100
        self.percentage_extreme_values = percentage_extreme_values

    def fit(self, X, y=None, **fit_param):
        """
        Computes indices of extreme values using a uniform distribution.
        Computes the absolute values uniformly at random between mean(X) + 3 sigma(X) and mean(X) + 5 sigma(X).
        The sign of the extreme value is the same as the value being replaced.

        :param X:
        :param y:
        :param fit_param:
        :return:
        """
        X = check_array(X, accept_sparse=False)
        means = X.mean(axis=0)
        vars = X.var(axis=0)
        self.n_features_in_ = X.shape[1]
        # compute number of missing values per column
        nb_extreme_values = int(X.shape[0] * self.percentage_extreme_values / 100)
        # generate missing values indices
        self.extreme_values_mapping_ = dict()
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_extreme_values, replace=False, p=None)
            for row in rows:
                abs_value = self.random_generator.uniform(
                    low=means[col] + 3. * vars[col],
                    high=means[col] + 5 * vars[col]
                )
                self.extreme_values_mapping_[(row, col)] = np.sign(X[row, col]) * abs_value

        return self
