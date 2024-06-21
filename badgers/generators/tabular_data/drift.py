import abc
from typing import Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs


class DriftGenerator(GeneratorMixin):
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

        :param random_generator: A random generator
        :param shift_std: The standard deviation of the amount of shift applied (shift is chosen from a normal distribution)
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, shift_std: Union[float,np.array] = 0.1):
        """
        Randomly shift (geometrical translation) values of each column independently of one another.
        Data are first standardized (mean = 0, var = 1) and a random number is added to each column.
        The ith columns is simply translated: `$x_i \left arrow x_i + \epsilon_i$`


        :param X:
        :param y:
        :param shift_std:
        :return:
        """
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift for each column
        shift = self.random_generator.normal(loc=0, scale=shift_std, size=X.shape[1])
        # add shift
        Xt += shift
        # inverse transform
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y


class RandomShiftClassesGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) values of each class independently of one another.
    Data are first standardized (mean = 0, var = 1) and
    for each class a random number is added to all instances.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, shift_std: Union[float,np.array] = 0.1):
        """
        Randomly shift (geometrical translation) values of each class independently of one another.
        Data are first standardized (mean = 0, var = 1) and
        for each class a random number is added to all instances.

        :param X:
        :param y:
        :param shift_std: The standard deviation of the amount of shift applied (shift is chosen from a normal distribution)
        """
        # extract unique labels
        classes = np.unique(y)
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift
        shifts = self.random_generator.normal(loc=0, scale=shift_std, size=len(classes))
        # add shift
        for c, s in zip(classes, shifts):
            Xt[y == c] += s
        # inverse transform
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y
