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

    def __init__(self, random_generator=default_rng(seed=0), sampling_proba_func=lambda X: normalize_proba(X[:, 0])):
        """

        :param random_generator: A random generator
        :param sampling_proba_func: A function that takes as input X and returns a sampling probability
        """
        super().__init__(random_generator=random_generator)
        self.sampling_proba_func = sampling_proba_func

    def transform(self, X):
        """
        Randomly samples instances based on the features values in X

        :param X:
        :return:
        """
        X = check_array(X)
        # total number of instances that will be missing
        # sampling
        sampling_proba = self.sampling_proba_func(X)
        Xt = self.random_generator.choice(X, p=sampling_proba, size=X.shape[0], replace=True)
        return Xt


class RandomSamplingClassesTransformer(ImbalanceTransformer):
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
        self.proportion_classes = proportion_classes

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = check_array(X)
        if set(y) != set(self.proportion_classes.keys()):
            raise ValueError(f'The proportion_classes attribute should have the same keys as the classes in y\n'
                             f'Keys in proportion_classes: {set(self.proportion_classes.keys())} are different from'
                             f'classes listed in y: {set(y)}')
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

        for label, prop in self.proportion_classes.items():
            size = int(prop * X.shape[0])
            Xt.append(self.random_generator.choice(X[self.original_labels_ == label], size=size, replace=True))
            transformed_labels += [label] * size

        Xt = np.vstack(Xt)
        self.labels_ = np.array(transformed_labels)

        return Xt


class RandomSamplingTargetsTransformer(ImbalanceTransformer):
    """
    Randomly samples data points
    """

    def __init__(self, random_generator=default_rng(seed=0), sampling_proba_func=lambda y: normalize_proba(y)):
        """

        :param random_generator: A random generator
        :param sampling_proba_func: A function that takes y as input and returns a sampling probability
        """
        super().__init__(random_generator=random_generator)
        self.sampling_proba_func = sampling_proba_func

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        self.sampling_probabilities_ = self.sampling_proba_func(y)
        return self

    def transform(self, X):
        """
        Randomly samples instances for each classes

        :param X:
        :param y:
        :return:
        """
        # input validation
        check_is_fitted(self, ['sampling_probabilities_'])
        X = check_array(X)

        Xt = self.random_generator.choice(X, p=self.sampling_probabilities_, size=X.shape[0], replace=True)

        return Xt
