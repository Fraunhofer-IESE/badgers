import abc
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs
from badgers.generators.time_series.utils import get_patterns_uniform_probability


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
        :return: Xt, y the transformed time series data and y (the same as input)
        """
        assert n_switches > 0, 'n_switches should be strictly greater than 0'

        self.switch_indices_ = self.random_generator.choice(len(X) - 1, size=n_switches, replace=False)
        for i in self.switch_indices_:
            tmp = X.iloc[i].copy()
            X.iloc[i] = X.iloc[i + 1].copy()
            X.iloc[i + 1] = tmp
        return X, y


class RandomRepeatGenerator(TransmissionErrorGenerator):
    """
    Repeats randomly values

    a time series x(t) = {x(0), x(1), x(2), x(3), x(4), x(5)} would be transformed to
                 x'(t) = {x(0), x(1), x(1), x(2), x(3), x(4), x(4), x(4), x(5)}

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
        :param n_repeats: number of values that will be repeated
        :param min_nb_repeats: the minimum number of repeats
        :param max_nb_repeats: the maximum number of repeats
        :return: Xt, y the transformed time series data and y (the same as input)
        """
        assert n_repeats > 0, 'n_repeats should be strictly greater than 0'

        # generate indices for repeats
        self.repeats_ = [(i, l) for i, l in zip(
            sorted(self.random_generator.choice(len(X), size=n_repeats, replace=False, shuffle=False)),
            self.random_generator.integers(low=min_nb_repeats, high=max_nb_repeats, size=n_repeats)
        )]

        Xt = X.values

        # generate the repeats and insert them
        offset = 0

        for i, l in self.repeats_:
            repeat = np.repeat(X.iloc[i], l)
            Xt = np.insert(Xt, i + offset, repeat)
            offset += l

        # create a pandas dataframe with same columns as X
        Xt = pd.DataFrame(data=Xt, columns=X.columns)

        return Xt, y


class RandomDropGenerator(TransmissionErrorGenerator):
    """
    Drops randomly values

    a time series x(t) = {x(0), x(1), x(2), x(3), x(4), x(5)} would be transformed to
                 x'(t) = {x(0), x(1), x(3), x(4)}

    This simulates a problem in transmission where several values are randomly dropped
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator:

        """
        super().__init__(random_generator=random_generator)
        self.drops_indices_ = None  # to store the indices of the drops

    @preprocess_inputs
    def generate(self, X, y, n_drops: int = 10) -> Tuple:
        """

        :param X: time series data
        :param y: not used
        :param n_drops: number of values to drop from the time series
        :return: Xt, y the transformed time series data and y (the same as input)
        """
        assert n_drops > 0, 'n_drops should be strictly greater than 0'

        # generate indices for drops
        self.drops_indices_ = self.random_generator.choice(len(X), size=n_drops, replace=False, shuffle=False)

        Xt = X.drop(X.index[self.drops_indices_], axis=0).reset_index(drop=True)

        return Xt, y


class LocalRegionsRandomDropGenerator(TransmissionErrorGenerator):
    """
    Drops randomly values in specific regions of time

    This simulates a problem in transmission where several values are randomly dropped at specific time intervals (time regions)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator:

        """
        super().__init__(random_generator=random_generator)
        self.drops_indices_ = None  # to store the indices of the drops
        self.drops_probabilities_ = None # to store the probability of dropping values

    @preprocess_inputs
    def generate(self, X, y, n_drops: int = 10, n_regions: int = 5, min_width_regions: int = 5,
                 max_width_regions: int = 10) -> Tuple:
        """

        :param X: time series data
        :param y: not used
        :param n_drops: number of values to drop from the time series
        :param n_regions: number of time regions (or time intervals) where values will be dropped
        :param min_width_regions: minimum width of the time regions (intervals)
        :param max_width_regions: maximum width of the time regions (intervals)
        :return: Xt, y the transformed time series data and y (the same as input)
        """
        assert n_drops > 0, 'n_drops should be strictly greater than 0'

        # generate probability for where to drop points
        self.drops_probabilities_ = get_patterns_uniform_probability(
            signal_size=len(X),
            random_generator=self.random_generator,
            n_patterns=n_regions,
            min_width_pattern=min_width_regions,
            max_width_patterns=max_width_regions
        )

        # generate indices for drops
        self.drops_indices_ = self.random_generator.choice(len(X), size=n_drops, p=self.drops_probabilities_, replace=False, shuffle=False)

        Xt = X.drop(X.index[self.drops_indices_], axis=0).reset_index(drop=True)

        return Xt, y


class LocalRegionsRandomRepeatGenerator(TransmissionErrorGenerator):
    """
    Repeats randomly values x(t) only in certain time regions (time intervals)

    This simulates a delay in transmission where several values are repeated over time (in certain time intervals)
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator:

        """
        super().__init__(random_generator=random_generator)
        self.repeats_ = None  # to store the indices of the repeats (from the original X) and the length of the repeat
        self.repeats_probabilities_ = None  # to store the probability of repeating values

    @preprocess_inputs
    def generate(self, X, y, n_repeats: int = 10, min_nb_repeats: int = 1,
                 max_nb_repeats: int = 10, n_regions: int = 5, min_width_regions: int = 5,
                 max_width_regions: int = 10) -> Tuple:
        """

        :param X:
        :param y:
        :param n_repeats: number of values that will be repeated
        :param min_nb_repeats: the minimum number of repeats
        :param max_nb_repeats: the maximum number of repeats
        :param n_regions: number of time regions (or time intervals) where values will be dropped
        :param min_width_regions: minimum width of the time regions (intervals)
        :param max_width_regions: maximum width of the time regions (intervals)
        :return: Xt, y the transformed time series data and y (the same as input)
        """
        assert n_repeats > 0, 'n_repeats should be strictly greater than 0'

        # generate probability for where to drop points
        self.repeats_probabilities_ = get_patterns_uniform_probability(
            signal_size=len(X),
            random_generator=self.random_generator,
            n_patterns=n_regions,
            min_width_pattern=min_width_regions,
            max_width_patterns=max_width_regions
        )

        # generate indices for repeats
        self.repeats_ = [(i, l) for i, l in zip(
            sorted(self.random_generator.choice(len(X), size=n_repeats, p=self.repeats_probabilities_, replace=False, shuffle=False)),
            self.random_generator.integers(low=min_nb_repeats, high=max_nb_repeats, size=n_repeats)
        )]

        Xt = X.values

        # generate the repeats and insert them
        offset = 0

        for i, l in self.repeats_:
            repeat = np.repeat(X.iloc[i], l)
            Xt = np.insert(Xt, i + offset, repeat)
            offset += l

        # create a pandas dataframe with same columns as X
        Xt = pd.DataFrame(data=Xt, columns=X.columns)

        return Xt, y