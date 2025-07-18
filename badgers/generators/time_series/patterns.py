import abc
from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.interpolate import CubicSpline

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs
from badgers.generators.time_series.utils import generate_random_patterns_indices


def add_offset(values: np.array, offset: float = 0.) -> np.array:
    """
    Adds an offset to the given array of values.

    :param values: The input array of values to which the offset will be added.
    :param offset: The offset value to be added to each element in the array.
    :return: A new array with the offset added to each element.
    """
    return values + offset


def add_linear_trend(values: np.array, start_value: float = 0., end_value: float = 1.) -> np.array:
    """
    Adds a linear trend to the given array of values.

    :param values: The input array of values to which the linear trend will be added.
    :param start_value: The starting value of the linear trend.
    :param end_value: The ending value of the linear trend.
    :return: A new array with the linear trend added to each element.
    """
    return values + np.linspace(start_value - values[0], end_value - values[-1], len(values))


def scale(values: np.array, scaling_factor: float = 1.) -> np.array:
    """
    Scales the given array of values by a specified factor.

    :param values: The input array of values to be scaled.
    :param scaling_factor: The factor by which to scale the values.
    :return: A new array with each element scaled by the specified factor.
    """
    return values * scaling_factor


class Pattern:

    def __init__(self, values: np.array):
        """
        Initialize a Pattern object.

        :param values: A 1D or 2D numpy array where rows represent the time axis and columns represent features.
                       If only a single feature is provided (1D array), it is automatically reshaped into a 2D array with shape (-1, 1).
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim > 2:
            raise ValueError(
                "Values has more thant 2 dimensions where it is expected to have either 1 or 2!"
            )
        self.values = values
        self.interpolation_function = CubicSpline(np.linspace(0, 1, values.shape[0]), values, bc_type='natural')

    def resample(self, nb_point: int) -> np.array:
        return self.interpolation_function(np.linspace(0, 1, nb_point))


class PatternsGenerator(GeneratorMixin):
    """
    Base class for transformers that generate patterns in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the PatternsGenerator with a random number generator.

        :param random_generator: An instance of a random number generator from `numpy.random`, used for generating random patterns.
        """
        self.random_generator = random_generator
        self.patterns_indices_ = []

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        """
        Abstract method to inject patterns into the time-series data.
        This method should be overridden by subclasses to implement specific pattern generation logic.

        :param X: Input time-series data as a 2D numpy array or pandas DataFrame.
        :param y: Target values as a 1D numpy array or pandas Series.
        :param params: Additional parameters that might be required for pattern generation.
        :return: A tuple containing the modified time-series data and target values.
        """
        pass

    def _inject_pattern(self, X: pd.DataFrame, p: Pattern, start_index: int, end_index: int,
                        scaling_factor: Union[float, str, None] = 'auto'):
        """
        Utility function to inject a predefined pattern `p` into a signal `X`.

        :param X: The signal (time-series data) to inject the pattern into, as a pandas DataFrame.
        :param p: The pattern to be injected, represented as a `Pattern` object.
        :param start_index: The starting index in `X` where the pattern injection begins.
        :param end_index: The ending index in `X` where the pattern injection ends.
        :param scaling_factor: The factor by which to scale the pattern before injection. Can be a float, 'auto' to scale based on the signal's range, or None to apply no scaling.
        :return: The transformed signal (time-series data) as a pandas DataFrame, where the pattern has been injected.
        """

        # start and end values
        v_start = X.iloc[start_index, :].values
        v_end = X.iloc[start_index, :].values

        # number of points needed for resampling
        nb_points = end_index - start_index + 1

        if scaling_factor == 'auto':
            # compute a scaling factor to make the pattern looks realistic
            scaling_factor = (np.max(X[start_index:end_index + 1]) - np.min(X[start_index:end_index + 1])) / (
                np.max(p.values) - np.min(p.values)) * self.random_generator.normal(1, 0.2)
        elif scaling_factor is None:
            # default to 1
            scaling_factor = 1.0

        transformed_pattern = p.resample(nb_point=nb_points)
        transformed_pattern = scale(transformed_pattern, scaling_factor=scaling_factor)
        transformed_pattern = add_linear_trend(start_value=v_start, end_value=v_end, values=transformed_pattern)

        X.iloc[start_index:end_index + 1, :] = transformed_pattern
        return X


class RandomlySpacedPatterns(PatternsGenerator):
    """
    Inject given patterns with random width and indices
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomlySpacedPatterns with a random number generator.

        :param random_generator: An instance of a random number generator from `numpy.random`, used for generating random patterns.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10,
                 pattern: Pattern = Pattern(values=np.array([0, 0, 0, 0, 0])),
                 scaling_factor: Union[float, str, None] = 'auto') -> Tuple:
        """
        Inject patterns with random width and indices in the time-series data.

        :param X: Input time-series data as a 2D numpy array or pandas DataFrame.
        :param y: Target values as a 1D numpy array or pandas Series (not used in this method).
        :param n_patterns: The number of patterns to inject into the time-series data.
        :param min_width_pattern: The minimum width of the pattern to inject.
        :param max_width_patterns: The maximum width of the pattern to inject.
        :param pattern: The pattern to inject, represented as a `Pattern` object.
        :param scaling_factor: The factor by which to scale the pattern before injection. Can be a float, 'auto' to scale based on the signal's range, or None to apply no scaling.
        :return: A tuple containing the transformed time-series data and the unchanged target values.
        """
        # generate patterns indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        for (start, end) in self.patterns_indices_:
            X = self._inject_pattern(X, p=pattern, start_index=start, end_index=end, scaling_factor=scaling_factor)

        return X, y


class RandomlySpacedConstantPatterns(PatternsGenerator):
    """
    Generates constant patterns of constant value with random width and indices
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomlySpacedConstantPatterns with a random number generator.

        :param random_generator: An instance of a random number generator from `numpy.random`, used for generating random patterns.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10,
                 constant_value: float = 0) -> Tuple:
        """
        Generate constant patterns with random width and indices in the time-series data.

        :param X: Input time-series data as a 2D numpy array or pandas DataFrame.
        :param y: Target values as a 1D numpy array or pandas Series (not used in this method).
        :param n_patterns: The number of constant patterns to inject into the time-series data.
        :param min_width_pattern: The minimum width of each constant pattern to inject.
        :param max_width_patterns: The maximum width of each constant pattern to inject.
        :param constant_value: The constant value of the patterns to inject.
        :return: A tuple containing the transformed time-series data and the unchanged target values.
        """
        # generate patterns indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        for (start, end) in self.patterns_indices_:
            X.iloc[start:end, :] = constant_value

        return X, y


class RandomlySpacedLinearPatterns(PatternsGenerator):
    """
    Generates patterns of constant slope (linear) with random width and indices
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomlySpacedLinearPatterns with a random number generator.

        :param random_generator: An instance of a random number generator from `numpy.random`, used for generating random patterns.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10) -> Tuple:
        """
        Generate linear patterns with random width and indices in the time-series data.

        :param X: Input time-series data as a 2D numpy array or pandas DataFrame.
        :param y: Target values as a 1D numpy array or pandas Series (not used in this method).
        :param n_patterns: The number of linear patterns to inject into the time-series data.
        :param min_width_pattern: The minimum width of each linear pattern to inject.
        :param max_width_patterns: The maximum width of each linear pattern to inject.
        :return: A tuple containing the transformed time-series data and the unchanged target values.
        """
        # generate patterns indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        for (start, end) in self.patterns_indices_:
            for col in range(X.shape[1]):
                X.iloc[start:end, col] = np.linspace(X.iloc[start, col], X.iloc[end, col], end - start)

        return X, y
