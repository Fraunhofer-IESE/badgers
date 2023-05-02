import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array


class ImbalanceTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        self.random_generator = random_generator


class RandomSamplingClassesTransformer(ImbalanceTransformer):

    def __init__(self, random_generator=default_rng(seed=0), min: int = None):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param min: int, default None
            The minimum number of instance per class.
            If `None` defaults to 10% of the number of instance in the smallest class
        """
        super().__init__(random_generator=random_generator)
        self.min = min

    def transform(self, X, y):
        """
        Randomly samples instances for each classes

        :param X:
        :param y:
        :return:
        """
        X = check_array(X)
        Xt = []
        yt = []
        # get the unique classes names and the number of unique classes
        classes, instances_per_class = np.unique(y, return_counts=True)
        # compute boundary for the sampling
        low = self.min if self.min is not None else int(0.1 * min(instances_per_class))
        high = max(instances_per_class)
        # compute the number of instances to be samples for each class
        samples_per_class = self.random_generator.integers(low, high, size=len(classes))
        # sampling with replacement
        for c, n in zip(classes, samples_per_class):
            Xt.append(self.random_generator.choice(X[y == c], size=n, replace=True))
            yt += [c] * n

        Xt = np.vstack(Xt)
        yt = np.array(yt)

        return Xt, yt
