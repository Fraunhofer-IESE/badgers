import abc

import numpy
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted


class MissingNodesTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for missing nodes transformer
    """
    def __init__(self, percentage_missing: int = 10, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """

        :param percentage_missing: int, default 10
            The percentage of missing nodes (int value between 0 and 100 included)
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        """
        assert 0 <= percentage_missing <= 100
        self.percentage_missing = percentage_missing
        self.random_generator = random_generator

    @abc.abstractmethod
    def fit(self, X, y=None, **fit_params):
        """
        Generates a list of indices for the missing nodes (`self.missing_nodes_indices_`).
        The list of indices takes the form of a list of nodes_ids.



        :param X: Graph object
        :param y:
        :param fit_params:
        :return:
        """
        pass

    def fit_transform(self, X, y=None, **fit_params):
        """

        :param X:
        :param y:
        :param fit_params:
        :return:
        """
        self.fit(X, y=None)
        return self.transform(X)

    def transform(self, X):
        """

        :param X: {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        :return X_transformed: array, shape (n_samples, n_features)
            The array containing missing values.
        """
        check_is_fitted("missing_nodes_indices_")
        raise NotImplementedError()
