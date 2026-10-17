import abc

import numpy as np
import sklearn.base
from numpy.random import default_rng
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs


class OutliersGenerator(GeneratorMixin):
    """
    Base class for transformers that add outliers to tabular data.

    After each ``generate()`` call the generator stores a boolean label
    vector describing ITS OWN output: ``get_outliers_labels()`` returns a
    per-row array aligned with the generated features, ``True`` where the
    row is an outlier (the deficit), ``False`` for an original row. The
    target ``y`` is only a pass-through -- deficit labels never ride in
    ``yt``.
    """

    def __init__(self, random_generator: np.random.Generator = default_rng(seed=0)):
        """
        Initialize the OutliersGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (default is a new generator with seed 0).
        """
        self.random_generator = random_generator
        self.outlier_labels_: np.ndarray | None = None

    def get_outliers_labels(self) -> np.ndarray:
        """
        Per-row outlier labels of the LAST generated output.

        :return: A boolean array aligned with the rows of the last
            generated features; ``True`` marks an outlier row (the
            deficit), ``False`` an original row.
        :raises RuntimeError: if ``generate()`` has not been called yet.
        """
        if self.outlier_labels_ is None:
            raise RuntimeError(
                "no labels available: call generate() first "
                "(the labels describe the LAST generated output)"
            )
        return self.outlier_labels_

    def _set_outlier_labels(self, n_original: int, n_outliers: int):
        """Record the label vector of an output built as original rows
        followed by outlier rows."""
        self.outlier_labels_ = np.concatenate([
            np.zeros(n_original, dtype=bool),
            np.ones(n_outliers, dtype=bool),
        ])

    @abc.abstractmethod
    def generate(self, X, y=None, **params):
        """
        Abstract method to generate outliers data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array, or None; passed through unchanged).
        :param params: Additional parameters required for the generation.
        """
        pass


class DecompositionAndOutlierGenerator(OutliersGenerator):

    def __init__(self, dimension_reduction_transformer: sklearn.base.TransformerMixin, outlier_generator: OutliersGenerator):
        """
        Initialize the DecompositionAndOutlierGenerator with a decomposition transformer and an outlier generator.

        :param decomposition_transformer: The dimensionality reduction transformer to be applied to the data before generating outliers.
        :param outlier_generator: The outlier generator to be used after the data has been transformed.
        """
        assert hasattr(
            dimension_reduction_transformer,
            'inverse_transform'), \
            f'the decomposition transformer class must implement the inverse_transform function.' \
            f'\nUnfortunately the class {dimension_reduction_transformer} does not'
        super().__init__(random_generator=outlier_generator.random_generator)

        self.decomposition_transformer = dimension_reduction_transformer
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
        # propagate the inner generator's labels (aligned with Xt's rows)
        self.outlier_labels_ = self.outlier_generator.get_outliers_labels()
        # inverse the manifold and standardization transformations
        return pipeline.inverse_transform(Xt), yt
