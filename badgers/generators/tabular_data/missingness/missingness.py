import abc

import numpy as np
import numpy.random

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import ExactCountSelector
from badgers.core.utils import normalize_proba


def _mar_weights(X, rng):
    """Per-column selection probabilities for classic MAR: for each column
    i a random OTHER column j is drawn (uniform, one rng.choice per column,
    in column order) and p[:, i] = normalized (reversed) values of j.
    Single-column input: p = normalized own values, no driver draw.
    Constant columns get uniform probabilities."""
    k = X.shape[1]
    ranges = X.max(axis=0) - X.min(axis=0)
    X_norm = (X.max(axis=0) - X) / np.where(ranges > 0, ranges, 1.0)
    # A constant column has no defined ordering: uniform weights keep the
    # selection for that column data-independent.
    X_norm[:, ranges == 0] = 1.0
    p = np.zeros_like(X)
    if k > 1:
        for i in range(k):
            j = rng.choice([x for x in range(k) if x != i])
            p[:, i] = X_norm[:, j]
    else:
        p = X_norm
    return normalize_proba(p)


def _mnar_weights(X):
    """Per-column selection probabilities for classic MNAR: normalized
    own-column values (no randomness, no driver). Selection probability
    grows with the own value (docstring: a cell at the column maximum has
    missing probability 1, one at the minimum 0).
    Constant columns get uniform probabilities."""
    ranges = X.max(axis=0) - X.min(axis=0)
    weights = (X - X.min(axis=0)) / np.where(ranges > 0, ranges, 1.0)
    # A constant column has no defined ordering: uniform weights.
    weights[:, ranges == 0] = 1.0
    return normalize_proba(weights)


class MissingValueGenerator(GeneratorMixin):
    """
    Base class for missing values transformer
    """

    def __init__(self, random_generator):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator
        self.missing_values_indices_ = None

    def _write_missing(self, X, masks):
        """Write NaNs from ``{column_position: bool-mask}``; populate
        ``missing_values_indices_``.

        Column-major order matches the legacy index list. The caller
        guarantees exactly one mask per column, in column order.
        """
        indices = []
        for col, mask in masks.items():
            X[mask, col] = np.nan
            indices.extend(
                zip(np.where(mask)[0].tolist(), [col] * int(mask.sum()))
            )
        self.missing_values_indices_ = indices
        return X

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Abstract method to generate missing values in the input data.
        This should be overridden by subclasses.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param params: Additional keyword arguments that might be required for specific implementations.
        :type params: dict
        """
        pass


class MissingCompletelyAtRandom(MissingValueGenerator):
    """
    A generator that removes values completely at random (MCAR [1]) (uniform distribution over all data).

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html

    Routes the row selection through the shared ``ExactCountSelector``
    preset (exact-count, optional soft probabilities).
    """

    def __init__(self, random_generator):
        """
        Initializes the MissingCompletelyAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Required (no default): pass a numpy.random.Generator.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` completely at random according to a specified percentage.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: The modified input features `Xt` with introduced missing values.
        """
        if not 0 <= percentage_missing <= 1:
            raise ValueError(
                f"percentage_missing must be in [0, 1], got {percentage_missing}")
        # Route the row selection through the shared ExactCountSelector
        # preset (uniform draws, exact count per column) and write the
        # NaNs from the returned {column: bool-mask} dict.
        selector = ExactCountSelector(
            list(range(X.shape[1])), rate=percentage_missing,
            rng=self.random_generator,
        )
        masks = selector(X)
        self._write_missing(X, masks)

        return X


class DummyMissingAtRandom(MissingValueGenerator):
    """
    A generator that removes values at random (MAR [1]),
    where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
    where j is randomly chosen.

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html

    Routes the row selection through the shared ``ExactCountSelector``
    preset (exact-count, optional soft probabilities).
    """

    def __init__(self, random_generator):
        """
        Initializes the DummyMissingAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Required (no default): pass a numpy.random.Generator.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` at random based on another feature,
        where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
        and j is randomly chosen.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: The modified input features `Xt` with introduced missing values.
        """
        if not 0 <= percentage_missing <= 1:
            raise ValueError(
                f"percentage_missing must be in [0, 1], got {percentage_missing}")
        # Route the row selection through the shared ExactCountSelector
        # preset. The soft coupling (per-column driver drawn uniformly
        # among the OTHER columns) enters via per-row weights; the driver
        # draws happen inside the weights callable, BEFORE the row draws
        # -- exactly the legacy RNG call order.
        selector = ExactCountSelector(
            list(range(X.shape[1])), rate=percentage_missing,
            weights=lambda X_: _mar_weights(X_, self.random_generator),
            rng=self.random_generator,
        )
        masks = selector(X)
        self._write_missing(X, masks)

        return X


class DummyMissingNotAtRandom(MissingValueGenerator):
    """
    A generator that removes values not at random (MNAR [1]),
    where the probability of a data instance X[i,j] missing depends linearly upon its own value.
    A data point X[i,j] = max(X[:,j]) has a missing probability of 1.
    A data point X[i,j] = min(X[:,j]) has a missing probability of 0.
    A constant column X[:,j] has no defined ordering: values are selected uniformly.

    Routes the row selection through the shared ``ExactCountSelector``
    preset (exact-count, optional soft probabilities).
    """

    def __init__(self, random_generator):
        """
        Initializes the DummyMissingNotAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Required (no default): pass a numpy.random.Generator.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, percentage_missing):
        """
        Introduces missing values into the input features `X` not at random, where the probability of a data instance X[i,j] missing
        depends linearly upon its own value. Specifically, a data point X[i,j] = max(X[:,j]) has a missing probability of 1, and a
        data point X[i,j] = min(X[:,j]) has a missing probability of 0.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: The modified input features `Xt` with introduced missing values.
        """
        if not 0 <= percentage_missing <= 1:
            raise ValueError(
                f"percentage_missing must be in [0, 1], got {percentage_missing}")
        # Route the row selection through the shared ExactCountSelector
        # preset. The self-coupling (probability of missing grows with the
        # own value) enters via per-row weights.
        selector = ExactCountSelector(
            list(range(X.shape[1])), rate=percentage_missing,
            weights=lambda X_: _mnar_weights(X_),
            rng=self.random_generator,
        )
        masks = selector(X)
        self._write_missing(X, masks)

        return X
