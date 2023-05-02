import abc

from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from badgers.transforms.tabular_data.utils import random_sign


class ExtremeValuesTransformer(TransformerMixin, BaseEstimator):
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
    def transform(self, X):
        """
        Replaces values in X with the precomputing extreme values.

        :param X: {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        :return X_transformed: array, shape (n_samples, n_features)
            The array containing missing values.
        """
        pass


class ZScoreTransformer(ExtremeValuesTransformer):
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
        self.extreme_values_mapping_ = None

    def transform(self, X):
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
        # compute number of extreme values per column
        nb_extreme_values = int(X.shape[0] * self.percentage_extreme_values / 100)
        # generate extreme values
        self.extreme_values_mapping_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_extreme_values, replace=False, p=None)
            # keeping track of the extreme values indices
            self.extreme_values_mapping_ += [(row, col) for row in rows]
            # computing extreme values
            for row in rows:
                value = means[col] + random_sign(self.random_generator) * self.random_generator.uniform(
                    low=means[col] + 3. * vars[col],
                    high=means[col] + 5 * vars[col]
                )
                # updating with new extreme value
                X[row, col] = value

        return X
