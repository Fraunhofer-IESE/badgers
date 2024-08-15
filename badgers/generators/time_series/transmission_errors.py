import abc
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


class TransmissionErrorGenerator(GeneratorMixin):
    """
    Base class for transformers that generate transmission errors

    Transmission errors are errors that includes values dropped, delay, switching values over time...
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class RandomTimeSwitchGenerator(TransmissionErrorGenerator):
    """
    Switches time randomly (for now uniformly at random).
    A time series x(t) = {x(0), x(1), x(2), ..., x(t), x(t+1), ..., x(n)} is transformed to
                 x'(t) = {x(0), x(1), x(2), ..., x(t+1), x(t), ..., x(n)}
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator:

        """
        super().__init__(random_generator=random_generator)
        self.switch_indices_ = None

    @preprocess_inputs
    def generate(self, X, y, n_switches: int = 10) -> Tuple:
        """
        Switch `n_switches` values between X[i] and X[i+1] where i is chosen uniformly at random in [0,len(X)-1]

        Nothing happens to y

        :param X:
        :param y:
        :param n_switches: number of switches
        :return: Xt, y
        """
        self.switch_indices_ = self.random_generator.choice(len(X) - 1, size=n_switches, replace=False)
        for i in self.switch_indices_:
            tmp = X.iloc[i].copy()
            X.iloc[i] = X.iloc[i + 1].copy()
            X.iloc[i + 1] = tmp
        return X, y


class RandomRepeatGenerator(TransmissionErrorGenerator):
    """
    Repeats randomly values in given regions (patterns)

    For a given region [0, 3[,
    a time series x(t) = {x(0), x(1), x(2), x(3), x(4), x(5)} would be transformed to
                 x'(t) = {x(0), x(0), x(0), x(1), x(1), x(2), x(2), x(2), x(3), x(3), x(5)}

    This simulates a delay in transmission where several values are repeated over time
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator:

        """
        super().__init__(random_generator=random_generator)
        self.repeats_ = None  # to store the indices of the repeats (from the original X) and the length of the repeat

    @preprocess_inputs
    def generate(self, X, y, n_repeats: int = 10, min_nb_repeats: int = 1,
                 max_nb_repeats: int = 10) -> Tuple:
        """

        :param X:
        :param y:
        :param n_repeats:
        :param min_nb_repeats:
        :param max_nb_repeats:
        :return:
        """
        # generate indices for repeats
        self.repeats_ = [(i, l) for i,l in zip(
            sorted(self.random_generator.choice(len(X), size=n_repeats, replace=False, shuffle=False)),
            self.random_generator.integers(low=min_nb_repeats, high=max_nb_repeats, size=n_repeats)
        )]

        Xt = X.values

        offset = 0

        for i, l in self.repeats_:
            repeat = np.repeat(X.iloc[i], l)
            Xt = np.insert(Xt, i + offset, repeat)
            offset += l

        Xt = pd.DataFrame(data=Xt, columns=X.columns)

        return Xt, y
