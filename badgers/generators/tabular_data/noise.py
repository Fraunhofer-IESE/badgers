import abc

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin


class NoiseGenerator(GeneratorMixin):
    """
    Base class for generators that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0), repeat=1):
        """
        :param random_generator: A random generator
        :param repeat: number of times a noisy point is generated from the original.
            repeat = 1 means that Xt.shape[0] == X.shape[0], repeat = 10 means that Xt.shape[0] == 10 * X.shape[0]
        """
        self.random_generator = random_generator
        self.repeat = repeat

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to the tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0), noise_std: float = 0.1, repeat=1):
        """

        :param random_generator: A random generator
        :param noise_std: The standard deviation of the noise to be added
        """
        super().__init__(random_generator=random_generator, repeat=repeat)
        self.noise_std = noise_std

    def generate(self, X, y, **params):
        """
        Adds Gaussian white noise to the data.
        The data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X: the input
        :param y: the target
        :param params: optional parameters
        :return: Xt, yt
        """
        # standardize X
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise and repeat
        Xt = np.concatenate(
            [
                Xt + self.random_generator.normal(loc=0, scale=self.noise_std, size=Xt.shape)
                for _ in range(self.repeat)
            ], axis=0
        )
        if y is not None:
            yt = np.concatenate([y] * self.repeat, axis=0)
        else:
            yt = None
        # inverse pca
        return scaler.inverse_transform(Xt), yt


class GaussianNoiseClassesGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to each class separately.
    """

    def __init__(self, random_generator=default_rng(seed=0), repeat=1, noise_std_per_class: dict = None):
        """

        :param random_generator: A random generator
        :param noise_std_per_class: A dictionary giving the standard deviation of the noise to be added for each class
            key = class labels, values = noise std for this given class
        """
        super().__init__(random_generator=random_generator, repeat=repeat)
        self.noise_std_per_class = noise_std_per_class

    def generate(self, X, y, **params):
        """
        Add Gaussian white noise to the data.
        the data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X: the input
        :param y: the target
        :param params: optional parameters
        :return: Xt, yt
        """
        # standardize X
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # add noise and repeat for each class

        tmp_Xt = []
        tmp_yt = []
        for label, noise_std in self.noise_std_per_class.items():
            data_class = np.array(Xt[y == label])
            noisy_data_class = np.concatenate(
                [
                    data_class + self.random_generator.normal(loc=0, scale=noise_std, size=data_class.shape)
                    for _ in range(self.repeat)
                ],
                axis=0
            )
            labels = [label] * self.repeat * data_class.shape[0]
            tmp_Xt.append(noisy_data_class.copy())
            tmp_yt.append(labels)

        Xt = np.concatenate(tmp_Xt, axis=0)
        yt = np.concatenate(tmp_yt, axis=0)
        # inverse pca
        return scaler.inverse_transform(Xt), yt
