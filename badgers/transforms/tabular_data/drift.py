import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


class DriftTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator


class RandomShift(DriftTransformer):
    """
    Randomly shift (geometrical translation) values of each column independently of one another.
    Data are first standardized (mean = 0, var = 1) and a random number is added to each column.
    The ith columns is simply translated: `$x_i \left arrow x_i + \epsilon_i$`
    """

    def __init__(self, random_generator=default_rng(seed=0), shift_std: float = 0.1):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param shift_std: float, default 0.1
            The standard deviation of the amount of shift applied (shift is chosen from a normal distribution)
        """
        super().__init__(random_generator=random_generator)
        self.shift_std = shift_std

    def transform(self, X):
        X = check_array(X)
        # normalize data
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift
        shift = self.random_generator.normal(loc=0, scale=self.shift_std, size=X.shape[0])
        # add shift
        Xt += shift
        # inverse transform
        return scaler.inverse_transform(Xt)


class RandomShiftClasses(DriftTransformer):
    """
    Randomly shift (geometrical translation) values of each class independently of one another.
    Data are first standardized (mean = 0, var = 1) and
    for each class a random number is added to all instances.
    """

    def __init__(self, random_generator=default_rng(seed=0), shift_std: float = 0.1):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param shift_std: float, default 0.1
            The standard deviation of the amount of shift applied (shift is chosen from a normal distribution)
        """
        super().__init__(random_generator=random_generator)
        self.shift_std = shift_std

    def transform(self, X, y):
        X = check_array(X)
        classes = np.unique(y)
        # normalize data
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift
        shift = self.random_generator.normal(loc=0, scale=self.shift_std, size=X.shape[0])
        # add shift
        for c in classes:
            Xt[y == c] += shift
        # inverse transform
        return scaler.inverse_transform(Xt)