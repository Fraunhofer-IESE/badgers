import numpy as np
import sklearn.base
from numpy.random import default_rng
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA, KernelPCA
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
        :param random_generator: A random generator
        :param percentage_outliers: The percentage of outliers to generate
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
            value = random_sign(self.random_generator, shape=X.shape[1]) * (
                3. + self.random_generator.exponential(size=X.shape[1]))
            # updating with new outliers
            Xt[row, :] = value

        return scaler.inverse_transform(Xt)


# class HistogramTransformer(OutliersTransformer):
#     """
#     Randomly generates outliers from low density regions. Low density regions are estimated through histograms
#     """
#
#     def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10,
#                  threshold_low_density: float = 0.1, bins: int = 10):
#         assert 0 < threshold_low_density < 1
#         super().__init__(random_generator, percentage_outliers)
#         self.threshold_low_density = threshold_low_density
#         self.bins = bins
#
#     def transform(self, X):
#         """
#         Randomly generates outliers from low density regions. Low density regions are estimated through histograms
#
#         1. Standardize the input data (mean = 0, variance = 1)
#         2. Compute histograms for the data
#         3. Replace the value of the data points marked as outliers as follows:
#             -
#         4. Inverse the standardization transformation
#
#         :param X:
#         :return:
#         """
#         X = check_array(X, accept_sparse=False)
#         # standardize data
#         scaler = StandardScaler()
#         # fit, transform
#         scaler.fit(X)
#         Xt = scaler.transform(X)
#
#         # compute number of outliers
#         n_outliers = int(X.shape[0] * self.percentage_extreme_values / 100)
#         # generate outliers indices
#         self.outliers_indices_ = self.random_generator.choice(X.shape[0], shape=n_outliers, replace=False, p=None)
#
#         # compute the histogram of the data
#         hist, edges = np.histogramdd(Xt, density=False, bins=self.bins)
#         # normalize
#         norm_hist = hist / (np.max(hist) - np.min(hist))
#         # get coordinates of the histogram where the density is low (below a certain threshold)
#         hist_coords_low_density = np.where(norm_hist <= self.threshold_low_density)
#         # randomly pick some coordinates in the histogram where the density is low
#         hist_coords_random = self.random_generator.choice(list(zip(*hist_coords_low_density)), n_outliers, replace=True)
#
#         # computing outliers values
#         for row, hcoords in zip(self.outliers_indices_, hist_coords_random):
#             value = [
#                 self.random_generator.uniform(low=edges[i][c], high=edges[i][c + 1]) for i, c in enumerate(hcoords)
#             ]
#             # updating with new outliers
#             Xt[row, :] = value
#
#         return scaler.inverse_transform(Xt)


class DecompositionZScoreTransformer(OutliersTransformer):

    def __init__(self, random_generator=default_rng(seed=0),
                 percentage_outliers: int = 10,
                 n_components: int = None,
                 decomposition_transformer_class: sklearn.base.TransformerMixin = PCA,
                 **decomposition_transformer_kwargs):
        """

        :param random_generator: A random generator
        :param percentage_outliers: The percentage of outliers to generate
        :param n_components: The number of components to be used by the decomposition transformation. If not set, it will default to log2(X.shape[1]) (if X.shape[1] > 2) or 1 (if X.shape[1] <= 2)
        :param decomposition_transformer_class: The class of the dimensionality reduction transformer to be used before the ZScoreTransformer. It needs to implement a `inverse_transform()` function and have the attribute `n_components`
        :param decomposition_transformer_kwargs: The parameters to be passed to the decomposition_transformer_class
        """
        assert hasattr(
            decomposition_transformer_class,
            'inverse_transform'), \
            f'the decomposition transformer class must implement the inverse_transform function.' \
            f'\nUnfortunately the class {decomposition_transformer_class} does not'
        super().__init__(random_generator, percentage_outliers)
        self.n_components = n_components
        self.decomposition_transformer_class = decomposition_transformer_class
        self.decomposition_transformer_kwargs = decomposition_transformer_kwargs
        self._z_score_transformer = ZScoreTransformer(random_generator=random_generator,
                                                      percentage_outliers=percentage_outliers)

    def transform(self, X):
        """
        Randomly generate outliers by first applying a dimensionality reduction technique (sklearn.decomposition)
        and a z-score transformer.

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply the dimensionality reduction transformer
        3. Generates outliers by applying the ZScoreTransformer
        4. Inverse the dimensionality reduction and the standardization transformations

        :param X:
        :return:
        """
        X = check_array(X)

        # check the number of components for the dimensionality reduction transformer
        if self.n_components is None:
            if X.shape[1] <= 2:
                n_components = 1
            else:
                n_components = int(np.log2(X.shape[1]))
        else:
            n_components = self.n_components

        # sctandardize the data and apply the manifold transformer
        pipeline = make_pipeline(
            StandardScaler(),
            self.decomposition_transformer_class(n_components=n_components, **self.decomposition_transformer_kwargs),
        )
        Xt = pipeline.fit_transform(X)
        # add outliers using the zscore_transformer
        Xt = self._z_score_transformer.transform(Xt)
        # update outliers indices
        self.outliers_indices_ = self._z_score_transformer.outliers_indices_
        # inverse the manifold and standardization transformations
        return pipeline.inverse_transform(Xt)


class PCATransformer(DecompositionZScoreTransformer):
    """
    Randomly generate outliers by first applying a PCA and a z-score transformer

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply a PCA
        3. Generates outliers by applying the ZScoreTransformer
        4. Inverse the PCA and the standardization transformation
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10, n_components: int = None,
                 **pca_kwargs):
        """

        :param random_generator: numpy.random.Generator, default default_rng(seed=0)
            A random generator
        :param percentage_outliers: int, default 10
            The percentage of outliers to generate
        :param n_components: int, default None
            The number of components to be used by the PCA transformation.
            If not set, it will default to log2(X.shape[1]) (if X.shape[1] > 2) or 1 (if X.shape[1] <= 2)
        """
        super().__init__(
            random_generator=random_generator,
            percentage_outliers=percentage_outliers,
            n_components=n_components,
            decomposition_transformer_class=PCA, **pca_kwargs
        )


class KernelPCATransformer(DecompositionZScoreTransformer):
    """
    Randomly generate outliers by first applying a Kernel PCA and a z-score transformer

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply a Kernel PCA
        3. Generates outliers by applying the ZScoreTransformer
        4. Inverse the Kernel PCA and the standardization transformation
    """

    def __init__(self, random_generator=default_rng(seed=0), percentage_outliers: int = 10, n_components: int = None,
                 **kernel_pca_kwargs):
        """

        :param random_generator: A random generator
        :param percentage_outliers: The percentage of outliers to generate
        :param n_components: The number of components to be used by the PCA transformation. If not set, it will default to log2(X.shape[1]) (if X.shape[1] > 2) or 1 (if X.shape[1] <= 2)
        """
        # forcing the fit_inverse_transform argument to be True
        kernel_pca_kwargs['fit_inverse_transform'] = True
        super().__init__(
            random_generator=random_generator,
            percentage_outliers=percentage_outliers,
            n_components=n_components,
            decomposition_transformer_class=KernelPCA,
            **kernel_pca_kwargs
        )
