import abc
from typing import Tuple

import pandas as pd
from badgers.generators.time_series.utils import generate_random_patterns_indices
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.time_series import preprocess_inputs


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

    def __init__(self, random_generator=default_rng(seed=0), ):
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5, max_width_patterns: int=10,
                 noise_std: float = 0.1) -> Tuple:
        # generate extreme values indices and values
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        for (start, end) in self.patterns_indices_:
            Xt[start:end, :] += self.random_generator.normal(loc=0, scale=noise_std, size=(end-start, Xt.shape[1]))

        # inverse standardization
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y


class GlobalGaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std: float = 0.1):
        """
        Add Gaussian white noise to the data.
        the data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param noise_std: The standard deviation of the noise to be added
        :param X:
        :return:
        """
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise
        Xt = Xt + self.random_generator.normal(loc=0, scale=noise_std, size=Xt.shape)
        # inverse standardization
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y
