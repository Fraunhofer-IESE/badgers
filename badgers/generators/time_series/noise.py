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


class GaussianNoiseGenerator(NoiseGenerator):
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
        # inverse pca
        return scaler.inverse_transform(Xt), y
