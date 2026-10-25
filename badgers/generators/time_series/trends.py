import abc

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
        Initialize the TrendsGenerator with a random number generator.

        :param random_generator: An instance of a random number generator,
                                 default is `numpy.random.default_rng(seed=0)`.
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Abstract method to generate trends in time-series data.

        :param X: The input features, typically a 2D array where each row represents a time step.
        :param params: Additional parameters that can be used by the generating method.
        :return: The modified features with generated trends.
        """
        pass


class GlobalAdditiveLinearTrendGenerator(TrendsGenerator):
    """
    Add a linear trend to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the GlobalAdditiveLinearTrendGenerator with a random number generator.

        :param random_generator: An instance of a random number generator,
                                 default is `numpy.random.default_rng(seed=0)`.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, slope):
        """
        Add a global linear trend to the input time-series data.

        :param X: The input signal to be transformed, expected to be a 2D array where each row represents a time step.
        :param slope: The slope of the trend (increase per time unit). Can be a single float value or a list of slopes
                      for each feature in X.
        :type slope: float or list
        :return: The transformed signal Xt (X + linear trend).
        """

        offset = np.linspace(0, slope * len(X), len(X))
        Xt = X + offset
        return Xt


class AdditiveLinearTrendGenerator(TrendsGenerator):
    """
    Add a linear trend to the input time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the AdditiveLinearTrendGenerator with a random number generator.

        :param random_generator: An instance of a random number generator,
                                 default is `numpy.random.default_rng(seed=0)`.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, slope, start: int | None = None, end: int | None = None):
        """
        Add a linear trend to a specified segment of the input time-series data.

        :param X: The input signal to be transformed, expected to be a 2D array where each row represents a time step.
        :param slope: The slope of the trend (increase per time unit). Can be a single float value or a list of slopes
                      for each feature in X.
        :type slope: float or list
        :param start: The starting index of the segment to apply the trend.
                      When None (default), a starting index is drawn uniformly at random
                      in the first half of the signal.
        :param end: The ending index of the segment to apply the trend.
                    When None (default), the trend lasts until the end of the signal.
        :return: The transformed signal Xt (X with the linear trend applied to the specified segment).
        """
        if start is None:
            # when start is not given, it is chosen randomly in the first half of the signal
            start = int(self.random_generator.uniform(0, int(0.5 * len(X))))

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
        return Xt


class RandomlySpacedLinearTrends(TrendsGenerator):
    """
    Generates randomly time intervals where a linear trend is added to the signal
    Slopes, Tme intervals locations and widths are chosen randomly
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomlySpacedLinearTrends with a random number generator.

        :param random_generator: An instance of a random number generator,
                                 default is `numpy.random.default_rng(seed=0)`.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10, slope_min: float = -0.05, slope_max: float = 0.05):
        """
        Generates randomly spaced time intervals where a linear trend is added to the signal.
        Slopes, time interval locations, and widths are chosen randomly.

        :param X: The input signal to be transformed, expected to be a 2D array where each row represents a time step.
        :param n_patterns: The total number of time intervals where a linear trend is added.
        :param min_width_pattern: The minimum width of the time intervals.
        :param max_width_patterns: The maximum width of the time intervals.
        :param slope_min: The minimum value of the slope. The slope is chosen uniformly at random between `slope_min` and `slope_max` for each time interval and each column of X.
        :param slope_max: The maximum value of the slope. The slope is chosen uniformly at random between `slope_min` and `slope_max` for each time interval and each column of X.

        :return: The transformed signal Xt (X with randomly spaced linear trends added).
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
        return Xt
