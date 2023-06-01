import abc

import numpy as np
import sklearn.base
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from badgers.core.utils import random_sign, random_spherical_coordinate
from core.base import GeneratorMixin


class OutliersTransformer(GeneratorMixin):
    """
    Base class for transformers that add outliers to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10):
        """
        :param random_generator: A random generator
        :param percentage_outliers: The percentage of outliers to generate
        """
        assert 0 <= percentage_outliers <= 100

        self.random_generator = random_generator
        self.percentage_extreme_values = percentage_outliers

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        pass


class ZScoreSampling(OutliersTransformer):
    """
    Randomly generates outliers as data points with a z-score > 3.
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10):
        super().__init__(random_generator, percentage_outliers)

    def generate(self, X, y=None, **params):
        """
        Randomly generates outliers as data points with a z-score > 3.

        1. Standardize the input data (mean = 0, variance = 1)
        3. Generate outliers as follows:
            - the sign is randomly chosen
            - for each dimension: the value is equal to 3 + a random number following an exponential distribution function
            with default parameters (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
        4. Inverse the standardization transformation

        :param X:
        :param y:
        :param params:
        :return:
        """

        # standardize data
        scaler = StandardScaler()

        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute number of outliers
        n_outliers = int(Xt.shape[0] * self.percentage_extreme_values / 100)

        # generate outliers
        outliers = np.array([
            random_sign(self.random_generator, size=Xt.shape[1]) * (
                3. + self.random_generator.exponential(size=Xt.shape[1]))
            for _ in range(n_outliers)
        ])

        # in case we only have 1 outlier, reshape the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        return scaler.inverse_transform(outliers), None


class HypersphereSampling(OutliersTransformer):
    """
    Generates outliers by sampling points from a hypersphere with radius at least 3 sigma
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10):
        super().__init__(random_generator, percentage_outliers)

    def generate(self, X, y=None, **params):
        """
        Randomly generates outliers as data points with a z-score > 3.

        1. Standardize the input data (mean = 0, variance = 1)
        3. Generate outliers on a hypersphere (see https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates):
            - angles are chosen uniformly at random
            - radius is = 3 + a random number following an exponential distribution function with default parameters (see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
        4. Inverse the standardization transformation

        :param X:
        :param y:
        :param params:
        :return:
        """

        # standardize data
        scaler = StandardScaler()

        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute number of outliers
        n_outliers = int(Xt.shape[0] * self.percentage_extreme_values / 100)

        # computing outliers
        outliers = np.array([
            random_spherical_coordinate(
                random_generator=self.random_generator,
                size=Xt.shape[1],
                radius=3. + self.random_generator.exponential()
            )
            for _ in range(n_outliers)
        ])

        # in case we only have 1 outlier, rehspe the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        return scaler.inverse_transform(outliers), None


class HistogramSampling(OutliersTransformer):
    """
    Randomly generates outliers from low density regions. Low density regions are estimated through histograms

    Should only be used with low dimensionality data!
    It will raise an error if the number of dimensions is greater than 5
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10,
                 threshold_low_density: float = 0.1, bins: int = 10):
        """

        :param random_generator: A random generator
        :param percentage_outliers: The percentage of outliers to generate
        :param threshold_low_density: the threshold that defines a low density region (must be between 0 and 1)
        :param bins: number of bins for the histogram
        """
        assert 0 < threshold_low_density < 1
        super().__init__(random_generator, percentage_outliers)
        self.threshold_low_density = threshold_low_density
        self.bins = bins

    def generate(self, X, y=None, **params):
        """
        Randomly generates outliers from low density regions. Low density regions are estimated through histograms

        1. Standardize the input data (mean = 0, variance = 1)
        2. Compute and normalize histogram for the data
        3. Sample datapoint uniformly at random within bins of low density
        4. Inverse the standardization transformation

        :param X:
        :return:
        """
        if X.shape[1] > 5:
            raise NotImplementedError('So far this transformer only supports tabular data with at most 5 columns')
        # standardize data
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)

        # compute number of outliers
        n_outliers = int(Xt.shape[0] * self.percentage_extreme_values / 100)

        # compute the histogram of the data
        hist, edges = np.histogramdd(Xt, density=False, bins=self.bins)
        # normalize
        norm_hist = hist / (np.max(hist) - np.min(hist))
        # get coordinates of the histogram where the density is low (below a certain threshold)
        hist_coords_low_density = np.where(norm_hist <= self.threshold_low_density)
        # randomly pick some coordinates in the histogram where the density is low
        hist_coords_random = self.random_generator.choice(list(zip(*hist_coords_low_density)), n_outliers, replace=True)

        # computing outliers values
        outliers = np.array([
            [
                self.random_generator.uniform(low=edges[i][c], high=edges[i][c + 1])
                for i, c in enumerate(h_coords)
            ]
            for h_coords in hist_coords_random
        ])

        # in case we only have 1 outlier, rehspe the array to match sklearn convention
        if outliers.shape[0] == 1:
            outliers = outliers.reshape(1, -1)

        return scaler.inverse_transform(outliers), None


class DecompositionAndSamplingTransformer(GeneratorMixin):

    def __init__(self, decomposition_transformer: sklearn.base.TransformerMixin = PCA(n_components=2),
                 outlier_transformer: OutliersTransformer = ZScoreSampling(default_rng(0), percentage_outliers=10)):
        """

        :param decomposition_transformer: The dimensionality reduction transformer to be used before the outlier transformer
        :param outlier_transformer: The outlier transformer to be used after the dimensionality has been reduced
        """
        assert hasattr(
            decomposition_transformer,
            'inverse_transform'), \
            f'the decomposition transformer class must implement the inverse_transform function.' \
            f'\nUnfortunately the class {decomposition_transformer} does not'

        self.decomposition_transformer = decomposition_transformer
        if hasattr(self.decomposition_transformer, 'fit_inverse_transform'):
            setattr(self.decomposition_transformer, 'fit_inverse_transform', True)
        self.outlier_transformer = outlier_transformer

    def generate(self, X, y=None, **params):
        """
        Randomly generate outliers by first applying a dimensionality reduction technique (sklearn.decomposition)
        and a outlier transformer.

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply the dimensionality reduction transformer
        3. Generates outliers by applying the outlier transformer
        4. Inverse the dimensionality reduction and the standardization transformations

        :param X:
        :return:
        """

        # standardize the data and apply the dimensionality reduction transformer
        pipeline = make_pipeline(
            StandardScaler(),
            self.decomposition_transformer,
        )
        Xt = pipeline.fit_transform(X)
        # add outliers using the zscore_transformer
        Xt = self.outlier_transformer.transform(Xt)
        # inverse the manifold and standardization transformations
        return pipeline.inverse_transform(Xt), None
