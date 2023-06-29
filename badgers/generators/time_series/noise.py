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
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple:
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, random_generator=default_rng(seed=0), signal_to_noise_ratio: float = 0.1):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param signal_to_noise_ratio: float, default 0.1
            The standard deviation of the noise to be added
        """
        super().__init__(random_generator=random_generator)
        self.signal_to_noise_ratio = signal_to_noise_ratio

    def generate(self, X, y, **params):
        """
        Add Gaussian white noise to te data.
        te data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to te data.

        :param X:
        :return:
        """
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise
        Xt = Xt + self.random_generator.normal(loc=0, scale=self.signal_to_noise_ratio, size=Xt.shape)
        # inverse pca
        return scaler.inverse_transform(Xt), y
