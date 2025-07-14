import abc

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs


class NoiseGenerator(GeneratorMixin):
    """
    Base class for generators that add noise to tabular data.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the NoiseGenerator with a specified random number generator.

        :param random_generator: A random number generator instance from numpy.random,
                                 used to introduce randomness in the noise generation process.
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params):
        """
        Abstract method to generate noisy data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array).
        :param params: Additional parameters required for noise generation.
        """
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to the tabular data.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """

        :param random_generator: A random generator

        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std):
        """
        Adds Gaussian white noise to the input data.
        The data is first standardized such that each feature (column) has a mean of 0 and a variance of 1.
        Gaussian noise is then generated from a normal distribution with a standard deviation equal to `noise_std`.
        This noise is added to the standardized data.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array), which remains unchanged.
        :param noise_std: Standard deviation of the Gaussian noise to be added.
        :return: Xt, yt where Xt is the noisy input features and yt is the unchanged target variable y.
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
        Initialize the GaussianNoiseClassesGenerator with a specified random number generator.

        :param random_generator: A random number generator instance from numpy.random,
                                 used to introduce randomness in the noise generation process.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, noise_std_per_class=dict()):
        """
        Add Gaussian white noise to the data separately for each class.
        The data is first standardized such that each feature (column) has a mean of 0 and a variance of 1.
        Gaussian noise is then generated from a normal distribution with a standard deviation specified in `noise_std_per_class` for each class.
        This noise is added to the standardized data for each class separately.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array).
        :param noise_std_per_class: A dictionary specifying the standard deviation of the noise to be added for each class.
            Keys are class labels, and values are the noise standard deviations for the corresponding classes.
        :return: Xt, yt where Xt is the noisy input features and yt is the unchanged target variable y.
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
