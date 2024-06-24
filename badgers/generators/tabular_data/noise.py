import abc

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs


class NoiseGenerator(GeneratorMixin):
    """
    Base class for generators that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params):
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to the tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: A random generator

        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std):
        """
        Adds Gaussian white noise to the data.
        The data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X: the input
        :param y: the target
        :param noise_std: The standard deviation of the noise to be added
        :return: Xt, yt
        """
        # standardize X
        scaler = StandardScaler()
        # fit, transform
        Xt = scaler.fit_transform(X)
        # add noise
        Xt += self.random_generator.normal(loc=0, scale=noise_std, size=Xt.shape)
        # inverse standardization
        Xt = scaler.inverse_transform(Xt)
        return pd.DataFrame(data=Xt, columns=X.columns, index=X.index), y


class GaussianNoiseClassesGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to each class separately.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: A random generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std_per_class=dict()):
        """
        Add Gaussian white noise to the data.
        the data is first standardized (each column has a mean = 0 and variance = 1).
        The noise is generated from a normal distribution with standard deviation = `noise_std`.
        The noise is added to the data.

        :param X: the input
        :param y: the target
                :param noise_std_per_class: A dictionary giving the standard deviation of the noise to be added for each class
            key = class labels, values = noise std for this given class
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
        for label, noise_std in noise_std_per_class.items():
            data_class = np.array(Xt[y == label])
            noisy_data_class = data_class + self.random_generator.normal(loc=0, scale=noise_std, size=data_class.shape)
            labels = np.array([label] * data_class.shape[0])
            tmp_Xt.append(noisy_data_class.copy())
            tmp_yt.append(labels)

        Xt = np.concatenate(tmp_Xt, axis=0)
        yt = np.concatenate(tmp_yt, axis=0)
        # inverse standardization
        Xt = scaler.inverse_transform(Xt)
        return pd.DataFrame(data=Xt, columns=X.columns, index=X.index), pd.Series(yt)
