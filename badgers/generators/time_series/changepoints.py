import abc
from typing import Tuple

from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


class ChangePointsGenerator(GeneratorMixin):
    """
    Base class for generators that generate changepoints in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0), ):
        """
        Initialize the ChangePointsGenerator with a given random number generator.

        :param random_generator: A random number generator instance (default is numpy's default_rng with seed 0).
        """
        self.random_generator = random_generator
        self.changepoints = None

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        """
        Abstract method that generates changepoints in the given time-series data.

        This method must be overridden by subclasses.

        :param X: Input features of the time-series data.
        :param y: Target values of the time-series data.
        :param params: Additional parameters required for changepoint generation.
        :return: A tuple containing the modified time-series data and the generated changepoints.
        """
        pass


class RandomChangeInMeanGenerator(ChangePointsGenerator):
    """
    Generate randomly change in mean changepoints
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomChangeInMeanGenerator with a given random number generator.

        :param random_generator: A random number generator instance (default is numpy's default_rng with seed 0).
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_changepoints: int = 10, min_change: float = -5,
                 max_change: float = 5) -> Tuple:
        """
        Generate random changepoints in the time-series data where the mean changes at each changepoint.

        :param X: Input features of the time-series data.
        :param y: Target values of the time-series data.
        :param n_changepoints: Number of changepoints to generate.
        :param min_change: Minimum value of the change in mean.
        :param max_change: Maximum value of the change in mean.
        :return: A tuple containing the modified time-series data and the generated changepoints.
        """
        # Generate change points
        self.changepoints = list(
            zip(
                self.random_generator.integers(int(0.05 * len(X)), int(0.95 * len(X)), size=n_changepoints),
                self.random_generator.uniform(min_change, max_change, size=n_changepoints)
            )
        )

        for idx, change in self.changepoints:
            X.iloc[idx:] += change

        return X, y
