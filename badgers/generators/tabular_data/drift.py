import abc
from typing import Union

import numpy as np
from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler


class DriftGenerator(GeneratorMixin):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the drift generator.
        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params):
        pass


class RandomShiftGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) values of each column independently of one another.
    Data are first standardized (mean = 0, var = 1) and a random number is added to each column.
    The ith columns is simply translated: `$x_i \left arrow x_i + \epsilon_i$`
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomShiftGenerator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, shift_std: Union[float, np.array, list] = 0.1):
        """
        Randomly shift (geometrical translation) values of each column independently of one another.
        Data are first standardized (mean = 0, var = 1), and a random number drawn from a normal distribution
        with mean 0 and standard deviation `shift_std` is added to each column.
        The ith column is simply translated: `$x_i \leftarrow x_i + \epsilon_i$`, where $\epsilon_i \sim \mathcal{N}(0, \text{shift\_std})$.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable, a 1D array-like object (optional). Not used in this implementation.
        :param shift_std: Standard deviation of the normal distribution from which the random shifts are drawn.
                          Can be a single float (applied to all columns) or an array of floats (one per column).
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        # check that if shift_std is an array, it is of length == X.shape[1]
        if hasattr(shift_std, '__len__'):
            assert len(shift_std) == X.shape[
                1], f"shift_std length ({len(shift_std)}) must equal number of features in X ({X.shape[1]})"
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift for each column
        shift = self.random_generator.normal(loc=0, scale=shift_std, size=X.shape[1])
        # add shift
        Xt += shift
        # inverse transform
        return scaler.inverse_transform(Xt), y


class RandomShiftClassesGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) values of each class independently of one another.
    Data are first standardized (mean = 0, var = 1) and
    for each class a random number is added to all instances.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomShiftClassesGenerator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, shift_std: Union[float, np.array] = 0.1):
        """
        Randomly shift (geometrical translation) values of each class independently of one another.
        Data are first standardized (mean = 0, var = 1) and for each class a random number is added to all instances.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable, a 1D array-like object representing the class labels.
        :param shift_std: Standard deviation of the normal distribution from which the random shifts are drawn.
                          Can be a single float (applied to all classes) or an array of floats (one per class).
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        # extract unique labels
        classes = np.unique(y)
        # check that if shift_std is an array, it is either of length == len(classes) or it dimention is (len(classes), X.shape[1])
        if hasattr(shift_std, '__len__'):
            shift_std_len = len(shift_std)
            expected_len = len(classes)
            expected_shape = (len(classes), X.shape[1])

            assert (
                shift_std_len == expected_len or (hasattr(shift_std, 'shape') and shift_std.shape == expected_shape)
            ), f"shift_std length ({shift_std_len}) must equal number of classes ({expected_len}) " \
               f"or shape must be {expected_shape}, got {getattr(shift_std, 'shape', 'no shape attribute')}"
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift
        # generate random values for the shift
        if hasattr(shift_std, 'shape') and shift_std.shape == (len(classes), X.shape[1]):
            # shift_std is a 2D array: generate separate shifts for each class and feature
            shifts = self.random_generator.normal(loc=0, scale=shift_std)
        else:
            # shift_std is 1D or scalar: generate one shift per class, applied to all features
            shifts = self.random_generator.normal(loc=0, scale=shift_std, size=len(classes))
        # add shift
        for c, s in zip(classes, shifts):
            Xt[y == c] += s
        # inverse transform
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y