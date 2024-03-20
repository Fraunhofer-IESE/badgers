import abc
from typing import Tuple

from numpy.random import default_rng

from badgers.core.base import GeneratorMixin


class ChangePointGenerator(GeneratorMixin):
    """
    Base class for generators that generate changepoints in time-series data
    """

    def __init__(self, random_generator=default_rng(seed=0), n_changepoints: int = 10):
        """
        :param random_generator: a random number generator
        :param n_outliers: the number of outliers to generate
        """
        self.random_generator = random_generator
        self.n_changepoints = n_changepoints
        self.changepoints = None

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class RandomChangeInMeanGenerator(ChangePointGenerator):
    """
    Generate randomly change in mean changepoints
    """

    def __init__(self, random_generator=default_rng(seed=0), n_changepoints: int = 10, min_change: float = -5,
                 max_change: float = 5):
        super().__init__(random_generator=random_generator, n_changepoints=n_changepoints)
        self.min_change = min_change
        self.max_change = max_change

    def generate(self, X, y, **params) -> Tuple:
        """

        :param X:
        :param y:
        :param params:
        :return:
        """
        # Generate change points
        self.changepoints = list(
            zip(
                self.random_generator.integers(int(0.05 * len(X)), int(0.95 * len(X)), size=self.n_changepoints),
                self.random_generator.uniform(self.min_change, self.max_change, size=self.n_changepoints)
            )
        )

        Xt = X.copy()

        for idx, change in self.changepoints:
            Xt[idx:] += change

        return Xt, y
