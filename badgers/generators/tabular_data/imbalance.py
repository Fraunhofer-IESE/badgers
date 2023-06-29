import abc

import numpy as np
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators import numpy_API
from badgers.core.utils import normalize_proba


class ImbalanceGenerator(GeneratorMixin):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        pass


class RandomSamplingFeaturesGenerator(ImbalanceGenerator):

    def __init__(self, random_generator=default_rng(seed=0), sampling_proba_func=lambda X: normalize_proba(X[:, 0])):
        """

        :param random_generator: A random generator
        :param sampling_proba_func: A function that takes as input data and returns a sampling probability
        """
        super().__init__(random_generator=random_generator)
        self.sampling_proba_func = sampling_proba_func

    @numpy_API
    def generate(self, X, y=None, **params):
        """
        Randomly samples instances based on the features values in X

        :param X:
        :param y:
        :return: Xt, yt
        """
        # total number of instances that will be missing
        # sampling
        sampling_proba = self.sampling_proba_func(X)
        sampling_mask = self.random_generator.choice(X.shape[0], p=sampling_proba, size=X.shape[0], replace=True)
        Xt = X[sampling_mask]
        yt = y[sampling_mask] if y is not None else y
        return Xt, yt


class RandomSamplingClassesGenerator(ImbalanceGenerator):
    """
    Randomly samples data points within predefined classes
    """

    def __init__(self, random_generator=default_rng(seed=0), proportion_classes: dict = None):
        """

        :param random_generator: A random generator
        :param proportion_classes: Example for having in total 50% of class 'A', 30% of class 'B', and 20% of class 'C'
            proportion_classes={'A':0.5, 'B':0.3, 'C':0.2}
        """
        super().__init__(random_generator=random_generator)
        self.transformed_labels_ = None
        self.proportion_classes = proportion_classes

    @numpy_API
    def generate(self, X, y, **params):
        """
        Randomly samples instances for each classes

        :param X:
        :param y:
        :param params:
        :return:
        """
        # local variables
        Xt = []
        transformed_labels = []

        for label, prop in self.proportion_classes.items():
            size = int(prop * X.shape[0])
            Xt.append(self.random_generator.choice(X[y == label], size=size, replace=True))
            transformed_labels += [label] * size

        Xt = np.vstack(Xt)
        yt = np.array(transformed_labels)

        return Xt, yt


class RandomSamplingTargetsGenerator(ImbalanceGenerator):
    """
    Randomly samples data points
    """

    def __init__(self, random_generator=default_rng(seed=0), sampling_proba_func=lambda y: normalize_proba(y)):
        """

        :param random_generator: A random generator
        :param sampling_proba_func: A function that takes y as input and returns a sampling probability
        """
        super().__init__(random_generator=random_generator)
        self.transformed_labels_ = None
        self.sampling_proba_func = sampling_proba_func

    @numpy_API
    def generate(self, X, y, **params):
        """
        Randomly samples instances for each classes

        :param X:
        :param y:
        :return:
        """
        sampling_probabilities_ = self.sampling_proba_func(y)
        sampling_mask = self.random_generator.choice(X.shape[0], p=sampling_probabilities_, size=X.shape[0],
                                                     replace=True)

        Xt = X[sampling_mask, :]
        yt = y[sampling_mask]

        return Xt, yt
