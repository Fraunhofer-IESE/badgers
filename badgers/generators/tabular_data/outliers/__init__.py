import abc

import numpy as np
import sklearn.base
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.outliers.distribution_sampling import ZScoreSamplingGenerator


class OutliersGenerator(GeneratorMixin):
    """
    Base class for transformers that add outliers to tabular data
    """

    def __init__(self, random_generator: np.random.Generator=default_rng(seed=0)):
        """
        Initialize the OutliersGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        """
        Abstract method to generate outliers data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array).
        :param params: Additional parameters required for noise generation.
        """
        pass


class DecompositionAndOutlierGenerator(OutliersGenerator):

    def __init__(self, decomposition_transformer: sklearn.base.TransformerMixin = PCA(n_components=2),
                 outlier_generator: OutliersGenerator = ZScoreSamplingGenerator(default_rng(0))):
        """
        Initialize the DecompositionAndOutlierGenerator with a decomposition transformer and an outlier generator.

        :param decomposition_transformer: The dimensionality reduction transformer to be applied to the data before generating outliers.
        :param outlier_generator: The outlier generator to be used after the data has been transformed.
        """
        assert hasattr(
            decomposition_transformer,
            'inverse_transform'), \
            f'the decomposition transformer class must implement the inverse_transform function.' \
            f'\nUnfortunately the class {decomposition_transformer} does not'
        super().__init__(random_generator=outlier_generator.random_generator)

        self.decomposition_transformer = decomposition_transformer
        self.outlier_generator = outlier_generator

    @preprocess_inputs
    def generate(self, X, y=None, **params):
        """
        Randomly generate outliers by first applying a dimensionality reduction technique (sklearn.decomposition)
        and an outlier transformer.

        1. Standardize the input data (mean = 0, variance = 1)
        2. Apply the dimensionality reduction transformer
        3. Generates outliers by applying the outlier transformer
        4. Inverse the dimensionality reduction and the standardization transformations

        :param X: the input features
        :param y: the regression target, class labels, or None
        :param params:
        :return:
        """

        # standardize the data and apply the dimensionality reduction transformer
        pipeline = make_pipeline(
            StandardScaler(),
            self.decomposition_transformer,
        )
        Xt = pipeline.fit_transform(X)
        # add outliers using the zscore_transformer
        Xt, yt = self.outlier_generator.generate(Xt, y, **params)
        # inverse the manifold and standardization transformations
        return pipeline.inverse_transform(Xt), yt
