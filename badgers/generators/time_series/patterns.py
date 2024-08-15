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
    return values + offset


def add_linear_trend(values: np.array, start_value: float = 0., end_value: float = 1.) -> np.array:
    return values + np.linspace(start_value - values[0], end_value - values[-1], len(values))


def scale(values: np.array, scaling_factor: float = 1.) -> np.array:
    return values * scaling_factor


class Pattern:

    def __init__(self, values: np.array):
        """
        Pattern constructor
        :param values: a 1D or 2D numpy array (rows = Time axis, columns = Features), if a single feature is used (1D), the values is automatically reshaped using reshape(-1,1)
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
        :param random_generator: a random number generator
        :param n_patterns: the number of patterns to generate
        """
        self.random_generator = random_generator
        self.patterns_indices_ = []

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass

    def _inject_pattern(self, X: pd.DataFrame, p: Pattern, start_index: int, end_index: int,
                        scaling_factor: Union[float,str] = 'auto'):
        """
        Utility function to inject a predefined pattern `p` into a signal `X`
        :param X: the signal to inject the pattern
        :param p: the pattern to be injected
        :param start_index:
        :param end_index:
        :param scaling_factor: float | None | "auto" (default "auto")
        :return: the transformed signal where the pattern has been injected
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
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10,
                 pattern: Pattern = Pattern(values=np.array([0, 0, 0, 0, 0]))) -> Tuple:
        # generate patterns indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        for (start, end) in self.patterns_indices_:
            X = self._inject_pattern(X, p=pattern, start_index=start, end_index=end, scaling_factor='auto')

        return X, y


class RandomlySpacedConstantPatterns(PatternsGenerator):
    """
    Generates constant patterns of constant value with random width and indices
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10,
                 constant_value: float = 0) -> Tuple:
        """

        :param X:
        :param y:
        :param n_patterns:
        :param min_width_pattern:
        :param max_width_patterns:
        :param constant_value:
        :return:
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
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5,
                 max_width_patterns: int = 10) -> Tuple:
        """

        :param X:
        :param y:
        :param n_patterns:
        :param min_width_pattern:
        :param max_width_patterns:
        :return:
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
