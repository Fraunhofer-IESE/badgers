import abc
from typing import Tuple

from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin


class NoiseGenerator(GeneratorMixin):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class LocalGaussianNoiseGenerator(NoiseGenerator):

    def __init__(self, random_generator=default_rng(seed=0), n_patterns: int = 10, patterns_width: int = 10,
                 noise_std: float = 0.1):
        super().__init__(random_generator=random_generator)
        self.n_patterns = n_patterns
        self.patterns_width = patterns_width
        self.noise_std = noise_std

    def generate(self, X, y, **params) -> Tuple:
        # TODO input validation!
        if X.ndim < 2:
            raise ValueError(
                "Expected 2D array. "
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )
        # generate extreme values indices and values
        self.patterns_indices_ = [(x, x + self.patterns_width) for x in
                                  self.random_generator.choice(X.shape[0] - self.patterns_width,
                                                               size=self.n_patterns,
                                                               replace=False, p=None)]

        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        for (start, end) in self.patterns_indices_:
            Xt[start:end, :] += self.random_generator.normal(loc=0, scale=self.noise_std, size=(self.patterns_width, Xt.shape[1]))

        # inverse standardization
        return scaler.inverse_transform(Xt), y


class GlobalGaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, random_generator=default_rng(seed=0), noise_std: float = 0.1):
        """

        :param random_generator: A random generator
        :param noise_std: The standard deviation of the noise to be added
        """
        super().__init__(random_generator=random_generator)
        self.noise_std = noise_std

    def generate(self, X, y, **params):
        """
        Add Gaussian white noise to the data.
        the data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X:
        :return:
        """
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise
        Xt = Xt + self.random_generator.normal(loc=0, scale=self.noise_std, size=Xt.shape)
        # inverse standardization
        return scaler.inverse_transform(Xt), y
