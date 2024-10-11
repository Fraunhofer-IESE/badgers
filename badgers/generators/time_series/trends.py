import abc
from typing import Tuple, Union

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs
from badgers.generators.time_series.utils import generate_random_patterns_indices


class TrendsGenerator(GeneratorMixin):
    """
    Base class for transformers that generate trends in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: a random number generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class GlobalAdditiveLinearTrendGenerator(TrendsGenerator):
    """
    Add a linear trend to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, slope) -> Tuple:
        """

        :param X: the input signal to be transformed
        :param y: not changed (here for API compatibility)
        :param slope: the slope of the trend (increase per time unit)
        :type slope: Union[float | list]
        :return: the transformed signal Xt (X + linear trend), and y (not changed)
        """

        offset = np.linspace(0, slope * len(X), len(X))
        Xt = X + offset
        return Xt, y


class AdditiveLinearTrendGenerator(TrendsGenerator):
    """
    Add a linear trend to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, slope, start: int, end: int) -> Tuple:
        """


        :param X: the input signal to be transformed
        :param y: not changed (here for API compatibility)
        :param slope: (increase per time unit)
        :type slope: Union[float | list]
        :param end:
        :param start:
        :return: the transformed signal Xt (X + linear trend), and y (not changed)
        """
        if start is None:
            # when start is not given, it is chosen randomly in the first half of the signal
            start = self.random_generator.uniform(0, int(0.5 * len(X)))

        if end is None:
            # when end is not given, the trend will last until the end of the signal
            end = len(X)

        # computing offset:
        # - 0s until "start"
        # - from "start" to "end": linear trend with slope "slope",
        # - from "end" on: the last value
        offset = np.zeros(shape=X.shape)
        offset[start:end, :] = np.linspace(0, slope * (end - start), end - start)
        offset[end:, :] = offset[end - 1, :]

        Xt = X + offset
        return Xt, y


class RandomlySpacedLinearTrends(TrendsGenerator):
    """
    Generates randomly time intervals where a linear trend is added to the signal
    Slopes, Tme intervals locations and widths are chosen randomly
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10, slope_min: float = -0.05, slope_max: float = 0.05) -> Tuple:
        """
        Generates randomly time intervals where a linear trend is added to the signal
        Slopes, Tme intervals locations and widths are chosen randomly.

        :param X:
        :param y:
        :param n_patterns: the total number of time intervals where a linear trend is add
        :param min_width_pattern: the minimum with of the time intervals
        :param max_width_patterns: the maximum with of the time intervals
        :param slope_min: the minimum value of the slope (slope is chosen uniformly at random between min_slope and max_slope for each time interval and each column of X)
        :param slope_max: the maximum value of the slope (slope is chosen uniformly at random between min_slope and max_slope for each time interval and each column of X)

        :return:
        """

        # generate patterns indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        # generate random slopes
        self.slopes_ = self.random_generator.uniform(low=slope_min, high=slope_max, size=(n_patterns, X.shape[1]))

        offset = np.zeros(shape=X.shape)

        for (start, end), slope in zip(self.patterns_indices_, self.slopes_):
            # computing offset:
            # - don't change until "start"
            # - from "start" to "end": add linear trend with slope "slope",
            # - from "end" on: add the last value
            offset[start:end, :] = np.linspace(offset[start, :], offset[start, :] + slope * (end - start), end - start)
            offset[end:, :] = offset[end - 1, :]

        Xt = X + offset
        return Xt, y
