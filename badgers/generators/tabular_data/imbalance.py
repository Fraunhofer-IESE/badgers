import abc

import numpy as np
import pandas as pd
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.utils import normalize_proba


class ImbalanceGenerator(GeneratorMixin):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the ImbalanceGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        """
        Abstract method to generate imbalanced data from the input data.
        This should be overridden

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, can be a pandas Series or a numpy array.
                  If None, it is assumed that the target is not provided.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param params: Additional keyword arguments that might be required for specific implementations.
        :type params: dict
        """
        pass


class RandomSamplingFeaturesGenerator(ImbalanceGenerator):

    def __init__(self, random_generator=default_rng(seed=0), ):
        """
        Initialize the RandomSamplingFeaturesGenerator with a specified random number generator.
        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, sampling_proba_func=lambda X: normalize_proba(X.iloc[:, 0])):
        """
        Randomly samples instances based on the feature values in X using a specified sampling probability function.

        The sampling probability function is applied to the input features X to determine the probability of each instance being sampled.
        By default, the first column of X is used to compute the normalized sampling probabilities.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, can be a pandas Series or a numpy array.
                  If None, it is assumed that the target is not provided.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param sampling_proba_func: A function that takes as input data (X) and returns a series of sampling probabilities.
                                    The function should ensure that the probabilities are normalized.
        :type sampling_proba_func: callable
        :return: A tuple containing the sampled features (Xt) and the corresponding target values (yt).
                 If y is None, only the sampled features (Xt) are returned.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
        """
        # total number of instances that will be missing
        # sampling
        sampling_proba = sampling_proba_func(X)
        sampling_mask = self.random_generator.choice(X.shape[0], p=sampling_proba, size=X.shape[0], replace=True)
        Xt = X.iloc[sampling_mask,:]
        yt = y[sampling_mask] if y is not None else y
        return Xt, yt


class RandomSamplingClassesGenerator(ImbalanceGenerator):
    """
    Randomly samples data points within predefined classes
    """

    def __init__(self, random_generator=default_rng(seed=0), ):
        """
        Initialize the RandomSamplingClassesGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)
        self.transformed_labels_ = None

    @preprocess_inputs
    def generate(self, X, y, proportion_classes: dict = None):
        """
        Randomly samples instances for each class based on the specified proportions.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, must be a pandas Series or a numpy array.
        :type y: Union[pandas.Series, numpy.ndarray]
        :param proportion_classes: A dictionary specifying the desired proportion of each class.
                                   The keys are class labels and the values are the desired proportions.
                                   For example, to have 50% of class 'A', 30% of class 'B', and 20% of class 'C',
                                   use `proportion_classes={'A': 0.5, 'B': 0.3, 'C': 0.2}`.
        :type proportion_classes: dict, optional
        :return: A tuple containing the sampled features (Xt) and the corresponding target values (yt).
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray]]
        """
        # local variables
        Xt = []
        transformed_labels = []

        for label, prop in proportion_classes.items():
            size = int(prop * X.shape[0])
            Xt.append(self.random_generator.choice(X[y == label], size=size, replace=True))
            transformed_labels += [label] * size

        Xt = pd.DataFrame(
            data=np.vstack(Xt),
            columns=X.columns
        )

        yt = pd.Series(data=transformed_labels)

        return Xt, yt


class RandomSamplingTargetsGenerator(ImbalanceGenerator):
    """
    Randomly samples data points
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomSamplingTargetsGenerator with a specified random number generator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)
        self.transformed_labels_ = None

    @preprocess_inputs
    def generate(self, X, y, sampling_proba_func=lambda y: normalize_proba(y)):
        """
        Randomly samples instances based on the target values in y using a specified sampling probability function.

        The sampling probability function is applied to the target values y to determine the probability of each instance being sampled.
        By default, the target values are used to compute the normalized sampling probabilities.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, must be a pandas Series or a numpy array.
        :type y: Union[pandas.Series, numpy.ndarray]
        :param sampling_proba_func: A function that takes as input target values (y) and returns a series of sampling probabilities.
                                    The function should ensure that the probabilities are normalized.
        :type sampling_proba_func: callable
        :return: A tuple containing the sampled features (Xt) and the corresponding target values (yt).
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray]]
        """
        sampling_probabilities_ = sampling_proba_func(y)
        sampling_mask = self.random_generator.choice(X.shape[0], p=sampling_probabilities_, size=X.shape[0],
                                                     replace=True)

        Xt = X.iloc[sampling_mask, :]
        yt = y[sampling_mask]

        return Xt, yt
