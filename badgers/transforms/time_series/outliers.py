from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from badgers.utils.utils import random_sign


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
        self.outliers_indices_ = None


class LocalZScoreTransformer(ExtremeValuesTransformer):
    """
    Randomly generates extreme values
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_extreme_values: int = 10,
                 local_window_size: int = 10):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_extreme_values: int, default 10
            The percentage of extreme values to generate
        :param  local_window_size: int, default 10
            The size (number of data points) of the local window to compute local Z-score
        """
        super().__init__(random_generator=random_generator)
        assert 0 <= percentage_extreme_values <= 100
        self.percentage_extreme_values = percentage_extreme_values
        self.local_window_size = local_window_size

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

        # compute number of extreme values per column
        nb_extreme_values = int(X.shape[0] * self.percentage_extreme_values / 100)
        # generate extreme values indices and values
        self.outliers_indices_ = dict()
        for col in range(X.shape[1]):
            rows = self.random_generator.choice(X.shape[0], size=nb_extreme_values, replace=False, p=None)
            # keeping track of the outliers indices
            self.outliers_indices_ += [(row, col) for row in rows]
            for row in rows:
                local_window = X[row - int(self.local_window_size / 2):row + int(self.local_window_size / 2), col]
                local_mean = local_window.mean()
                local_var = local_window.var()
                value = local_mean + random_sign(self.random_generator) * self.random_generator.uniform(
                    low=3. * local_var[col], high=5 * local_mean[col]
                )
                # updating with new outliers
                X[row, col] = value

        return self