from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

from badgers.utils.utils import random_sign


class ExtremeValuesTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that add outliers to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator
        self.outliers_indices_ = None


class ZScoreTransformer(ExtremeValuesTransformer):
    """
    Randomly generates outliers based on a z-score
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10, ):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_outliers: int, default 10
            The percentage of outliers to generate
        """
        super().__init__(random_generator=random_generator)
        assert 0 <= percentage_outliers <= 100
        self.percentage_extreme_values = percentage_outliers

    def transform(self, X):
        """
        Computes indices of outliers using a uniform distribution.
        Computes the absolute values uniformly at random between mean(X) + 3 sigma(X) and mean(X) + 5 sigma(X).
        The sign of the outliers is the same as the value being replaced.

        :param X:
        :return:
        """
        X = check_array(X, accept_sparse=False)
        means = X.mean(axis=0)
        vars = X.var(axis=0)
        self.n_features_in_ = X.shape[1]
        # compute number of outliers per column
        nb_extreme_values = int(X.shape[0] * self.percentage_extreme_values / 100)
        # generate outliers
        self.outliers_indices_ = []
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_extreme_values, replace=False, p=None)
            # keeping track of the outliers indices
            self.outliers_indices_ += [(row, col) for row in rows]
            # computing outliers
            for row in rows:
                value = means[col] + random_sign(self.random_generator) * self.random_generator.uniform(
                    low=means[col] + 3. * vars[col],
                    high=means[col] + 5 * vars[col]
                )
                # updating with new outliers
                X[row, col] = value

        return X