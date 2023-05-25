import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from badgers.core.utils import normalize_proba


class ImbalanceTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that makes tabular data imbalanced
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator


class RandomSamplingFeaturesTransformer(ImbalanceTransformer):

    def __init__(self, random_generator=default_rng(seed=0), percentage_missing: int = 10,
                 sampling_proba_func=lambda X: normalize_proba(X[:, 0])):
        """

        :param random_generator: A random generator
        :param estimator:
        """
        assert 0 < percentage_missing < 100
        super().__init__(random_generator=random_generator)
        self.sampling_proba_func = sampling_proba_func
        self.percentage_missing = percentage_missing

    def transform(self, X):
        """
        Randomly samples instances based on the features values in X

        :param X:
        :return:
        """
        X = check_array(X)
        # total number of instances that will be missing
        size = int(X.shape[0] * (100 - self.percentage_missing) / 100)
        # sampling
        sampling_proba = self.sampling_proba_func(X)
        Xt = self.random_generator.choice(X, p=sampling_proba, size=size)
        return Xt


class RandomSamplingClassesTransformer(ImbalanceTransformer):

    def __init__(self, random_generator=default_rng(seed=0), min_instances: int = None):
        """

        :param random_generator: A random generator
        :param min_instances: The minimum number of instance per class. If `None` defaults to 10% of the number of instance in the smallest class
        """
        super().__init__(random_generator=random_generator)
        self.min = min_instances

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = check_array(X)
        self.original_labels_ = y
        return self

    def transform(self, X):
        """
        Randomly samples instances for each classes

        :param X:
        :param y:
        :return:
        """
        # input validation
        check_is_fitted(self, ['original_labels_'])
        X = check_array(X)
        # local variables
        Xt = []
        transformed_labels = []
        # get the unique classes names and the number of unique classes
        classes, instances_per_class = np.unique(self.original_labels_, return_counts=True)
        # compute boundary for the sampling
        low = self.min if self.min is not None else int(0.1 * min(instances_per_class))
        high = max(instances_per_class)
        # compute the number of instances to be samples for each class
        samples_per_class = self.random_generator.integers(low, high, size=len(classes))
        # sampling with replacement
        for c, n in zip(classes, samples_per_class):
            Xt.append(self.random_generator.choice(X[self.original_labels_ == c], size=n, replace=True))
            transformed_labels += [c] * n

        Xt = np.vstack(Xt)
        self.labels_ = np.array(transformed_labels)

        return Xt
