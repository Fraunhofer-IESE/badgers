import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


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
        self.n_features_in_ = X.shape[1]
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
