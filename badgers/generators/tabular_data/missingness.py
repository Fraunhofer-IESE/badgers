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
        """
        Abstract method to generate missing values in the input data.
        This should be overridden by subclasses.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, can be a pandas Series or a numpy array.
                  If None, it is assumed that the target is not provided and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param params: Additional keyword arguments that might be required for specific implementations.
        :type params: dict
        """
        pass


class MissingCompletelyAtRandom(MissingValueGenerator):
    """
    A generator that removes values completely at random (MCAR [1]) (uniform distribution over all data).

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the MissingCompletelyAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` completely at random according to a specified percentage.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
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
        Initializes the DummyMissingAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` at random based on another feature,
        where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
        and j is randomly chosen.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
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
        Initializes the DummyMissingNotAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing):
        """
        Introduces missing values into the input features `X` not at random, where the probability of a data instance X[i,j] missing
        depends linearly upon its own value. Specifically, a data point X[i,j] = max(X[:,j]) has a missing probability of 1, and a
        data point X[i,j] = min(X[:,j]) has a missing probability of 0.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
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
