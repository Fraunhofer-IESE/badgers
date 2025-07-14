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
        Initializes the NoiseGenerator with a specified random generator.
        :param random_generator: An instance of a random number generator from `numpy.random`.
                                 Default is `default_rng(seed=0)`.
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method to be implemented by subclasses. Adds noise to the input data.

        :param X: Input features DataFrame.
        :param y: Target Series.
        :param params: Additional parameters that might be required for noise generation.
        :return: A tuple containing the modified features DataFrame and the target Series.
        """
        pass


class LocalGaussianNoiseGenerator(NoiseGenerator):

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the LocalGaussianNoiseGenerator with a specified random generator.

        :param random_generator: An instance of a random number generator from `numpy.random`.
                                 Default is `default_rng(seed=0)`.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, n_patterns: int = 10, min_width_pattern: int = 5, max_width_patterns: int = 10,
                 noise_std: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Adds Gaussian noise to randomly selected local patterns within the input data.

        :param X: Input features DataFrame.
        :param y: Target Series.
        :param n_patterns: Number of local patterns to add noise to.
        :param min_width_pattern: Minimum width of each pattern.
        :param max_width_patterns: Maximum width of each pattern.
        :param noise_std: Standard deviation of the Gaussian noise.
        :return: A tuple containing the modified features DataFrame and the original target Series.
        """
        # Generate indices for random patterns
        self.patterns_indices_ = generate_random_patterns_indices(
            random_generator=self.random_generator,
            n_patterns=n_patterns,
            signal_size=len(X),
            min_width_pattern=min_width_pattern,
            max_width_patterns=max_width_patterns)

        scaler = StandardScaler()
        # Fit and transform the data
        scaler.fit(X)
        Xt = scaler.transform(X)

        # Add Gaussian noise to each pattern
        for (start, end) in self.patterns_indices_:
            Xt[start:end, :] += self.random_generator.normal(loc=0, scale=noise_std, size=(end-start, Xt.shape[1]))

        # Inverse standardize the data
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y


class GlobalGaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the GlobalGaussianNoiseGenerator with a specified random generator.
        :param random_generator: An instance of a random number generator from `numpy.random`.
                                 Default is `default_rng(seed=0)`.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std: float = 0.1):
        """
        Adds Gaussian white noise to the entire dataset.
        The data is first standardized (each feature has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is then added to the standardized data, and the result is inverse-standardized to restore the original scale.

        :param X: Input features DataFrame.
        :param y: Target Series.
        :param noise_std: The standard deviation of the noise to be added.
        :return: A tuple containing the modified features DataFrame and the original target Series.
        """
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise
        Xt = Xt + self.random_generator.normal(loc=0, scale=noise_std, size=Xt.shape)
        # inverse standardization
        return pd.DataFrame(data=scaler.inverse_transform(Xt), columns=X.columns, index=X.index), y
