import abc

import numpy as np
import sklearn.base
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
    row is an outlier (the deficit), ``False`` for an original row.
    """

    def __init__(self, random_generator):
        """
        Initialize the OutliersGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (required; no default).
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
    def generate(self, X, **params):
        """
        Abstract method to generate outliers data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param params: Additional parameters required for the generation.
        """
        pass


class StructuredOutlierGenerator(OutliersGenerator):
    """Generates outliers in a reduced (structured) space.

    The input is first mapped to a lower-dimensional representation
    (standardization + a decomposition by default), outliers are sampled
    there — where the data really has little support — and the result is
    mapped back to the original feature space. Preprocessing is not part
    of the generator's responsibility: pass a custom ``preprocessor``
    (e.g. an already configured / fitted sklearn pipeline with
    ``transform``/``inverse_transform``) or leave it ``None`` to use the
    built-in standardization + decomposition.

    Parameters
    ----------
    dimension_reduction_transformer : sklearn transformer with
        ``inverse_transform``, optional
        The decomposition applied after standardization (default path).
        Ignored when ``preprocessor`` is given.
    outlier_generator : OutliersGenerator
        The outlier sampler used in the reduced space.
    preprocessor : sklearn pipeline-like with ``transform`` and
        ``inverse_transform``, or None, default=None
        Optional custom preprocessing step used instead of the built-in
        standardization + decomposition. It is (re)fit on ``X`` at each
        ``generate()`` call.
    """

    def __init__(
        self,
        dimension_reduction_transformer: sklearn.base.TransformerMixin = None,
        outlier_generator: OutliersGenerator = None,
        preprocessor=None,
    ):
        if preprocessor is None:
            if dimension_reduction_transformer is None:
                raise ValueError(
                    "either dimension_reduction_transformer or preprocessor "
                    "must be given")
            if not hasattr(dimension_reduction_transformer, 'inverse_transform'):
                raise ValueError(
                    f'the decomposition transformer class must implement the inverse_transform function.'
                    f'\nUnfortunately the class {dimension_reduction_transformer} does not')
        else:
            if not hasattr(preprocessor, 'inverse_transform'):
                raise ValueError(
                    f'the preprocessor must implement the inverse_transform function.'
                    f'\nUnfortunately the class {preprocessor} does not')
        if outlier_generator is not None:
            super().__init__(random_generator=outlier_generator.random_generator)
        else:
            super().__init__()

        self.decomposition_transformer = dimension_reduction_transformer
        self.outlier_generator = outlier_generator
        self.preprocessor = preprocessor

    @preprocess_inputs
    def generate(self, X, **params):
        """
        Randomly generate outliers by first applying a dimensionality reduction technique (sklearn.decomposition)
        and an outlier transformer.

        1. Standardize the input data (mean = 0, variance = 1) -- or apply the custom ``preprocessor``
        2. Apply the dimensionality reduction transformer (default path only)
        3. Generates outliers by applying the outlier transformer
        4. Inverse the dimensionality reduction and the standardization transformations (or the custom ``preprocessor``)

        :param X: the input features
        :param params:
        :return: The generated features in the original feature space.
        """

        # standardize the data and apply the dimensionality reduction transformer
        if self.preprocessor is not None:
            pipeline = self.preprocessor
        else:
            pipeline = make_pipeline(
                StandardScaler(),
                self.decomposition_transformer,
            )
        Xt = pipeline.fit_transform(X)
        # add outliers using the zscore_transformer
        Xt = self.outlier_generator.generate(Xt, **params)
        # propagate the inner generator's labels (aligned with Xt's rows)
        self.outlier_labels_ = self.outlier_generator.get_outliers_labels()
        # inverse the manifold and standardization transformations
        return pipeline.inverse_transform(Xt)


# Backwards-compatible alias (renamed in 0.0.15)
DecompositionAndOutlierGenerator = StructuredOutlierGenerator
