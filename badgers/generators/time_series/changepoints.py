import abc


from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


class ChangePointsGenerator(GeneratorMixin):
    """
    Base class for generators that generate changepoints in time-series data
    """

    def __init__(self, random_generator):
        """
        Initialize the ChangePointsGenerator with a given random number generator.

        :param random_generator: A random number generator instance (required; no default).
        """
        self.random_generator = random_generator
        self.changepoints = None

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Abstract method that generates changepoints in the given time-series data.

        This method must be overridden by subclasses.

        :param X: Input features of the time-series data.
        :param params: Additional parameters required for changepoint generation.
        :return: The modified time-series data.
        """
        pass


class RandomChangeInMeanGenerator(ChangePointsGenerator):
    """
    Generate randomly change in mean changepoints
    """

    def __init__(self, random_generator):
        """
        Initialize the RandomChangeInMeanGenerator with a given random number generator.

        :param random_generator: A random number generator instance (required; no default).
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, n_changepoints: int = 10, min_change: float = -5,
                 max_change: float = 5):
        """
        Generate random changepoints in the time-series data where the mean changes at each changepoint.

        The changepoint indices are distinct (no position is drawn twice). The changes are
        applied cumulatively: each change shifts the signal from its changepoint index to the
        end of the signal. The reported ``changepoints`` list is sorted by index (ascending
        position) and each entry's change value IS the realized level jump at that position
        (the level immediately before vs. immediately after the changepoint index).

        :param X: Input features of the time-series data.
        :param n_changepoints: Number of changepoints to generate.
        :param min_change: Minimum value of the change in mean.
        :param max_change: Maximum value of the change in mean.
        :return: The modified time-series data.
        """
        # Generate change points (distinct indices, sorted by position)
        n_candidates = int(0.95 * len(X)) - int(0.05 * len(X))
        idx = self.random_generator.choice(
            range(int(0.05 * len(X)), int(0.95 * len(X))), size=min(n_changepoints, n_candidates),
            replace=False)
        changes = self.random_generator.uniform(min_change, max_change, size=len(idx))
        self.changepoints = sorted(zip(idx, changes), key=lambda t: t[0])
        self.changepoints = sorted(zip(idx, changes), key=lambda t: t[0])

        for idx, change in self.changepoints:
            X[idx:] += change

        return X
