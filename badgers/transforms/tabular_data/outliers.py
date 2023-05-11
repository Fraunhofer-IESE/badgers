import numpy as np
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from badgers.core.utils import random_sign


class OutliersTransformer(TransformerMixin, BaseEstimator):
    """
    Base class for transformers that add outliers to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10):
        """
        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_outliers: int, default 10
            The percentage of outliers to generate
        """
        assert 0 <= percentage_outliers <= 100

        self.random_generator = random_generator
        self.percentage_extreme_values = percentage_outliers
        self.outliers_indices_ = None


class ZScoreTransformer(OutliersTransformer):
    """
    Randomly generates outliers as data points with a z-score > 3.
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10):
        super().__init__(random_generator, percentage_outliers)

    def transform(self, X):
        """
        Randomly generates outliers as data points with a z-score > 3.

        1. Standardize the input data (mean = 0, variance = 1)
        2. Randomly selects indices for the outliers (uniformly at random)
        3. Replace the value of the data points marked as outliers as follows:
            - the sign is randomly chosen
            - the value is equal to 3 + a random number following an exponential distribution function with default parameters (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
        4. Inverse the standardization transformation

        :param X:
        :return:
        """
        X = check_array(X, accept_sparse=False)
        # standardize data
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute number of outliers
        n_outliers = int(X.shape[0] * self.percentage_extreme_values / 100)
        # generate outliers
        self.outliers_indices_ = self.random_generator.choice(X.shape[0], size=n_outliers, replace=False, p=None)

        # computing outliers
        for row in self.outliers_indices_:
            value = random_sign(self.random_generator) * (3. + self.random_generator.exponential(size=X.shape[1]))
            # updating with new outliers
            Xt[row, :] = value

        return scaler.inverse_transform(Xt)


class PCATransformer(OutliersTransformer):
    """
    Randomly generate outliers by first applying a PCA and a z-score transformer
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10, n_components: int = None):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_outliers: int, default 10
            The percentage of outliers to generate
        :param n_components: int, default None
            The number of components to be used by the PCA transformation.
            If not set, it will default to log2(X.shape[1]) (if X.shape[1] > 2) or 1 (if X.shape[1] <= 2)
        """
        super().__init__(random_generator, percentage_outliers)
        self.n_components = n_components
        self._z_score_transformer = ZScoreTransformer(random_generator=random_generator,
                                                      percentage_outliers=percentage_outliers)

    def transform(self, X):
        """
        Randomly generate outliers by first applying a PCA and a z-score transformer.

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply a PCA
        3. Generates outliers by appyling the ZScoreTransformer
        4. Inverse the PCA and the standardization transformation

        :param X:
        :return:
        """
        X = check_array(X)

        # check the number of components for the PCA
        if self.n_components is None:
            if X.shape[1] <= 2:
                n_components = 1
            else:
                n_components = int(np.log2(X.shape[1]))
        else:
            n_components = self.n_components

        # sctandardize the data and apply a PCA
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        Xt = pipeline.fit_transform(X)
        # add outliers using the zscore_transformer
        Xt = self._z_score_transformer.transform(Xt)
        # update outliers indices
        self.outliers_indices_ = self._z_score_transformer.outliers_indices_
        # inverse the PCA and standardization transformations
        return pipeline.inverse_transform(Xt)
