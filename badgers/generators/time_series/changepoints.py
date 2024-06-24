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
        :param random_generator: a random number generator
        """
        self.random_generator = random_generator
        self.changepoints = None

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class RandomChangeInMeanGenerator(ChangePointsGenerator):
    """
    Generate randomly change in mean changepoints
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_changepoints: int = 10, min_change: float = -5,
                 max_change: float = 5) -> Tuple:
        """

        :param X:
        :param y:
        :param max_change:
        :param min_change:
        :param n_changepoints:
        :return:
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
