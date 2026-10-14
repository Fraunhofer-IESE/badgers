import abc

import numpy as np
import numpy.random
from numpy.random import default_rng

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import ExactCountSelector
from badgers.core.utils import normalize_proba


def _mar_weights(X, rng):
    """Per-column selection probabilities for classic MAR: for each column
    i a random OTHER column j is drawn (uniform, one rng.choice per column,
    in column order) and p[:, i] = normalized (reversed) values of j.
    Single-column input: p = normalized own values, no driver draw."""
    k = X.shape[1]
    X_norm = (X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))
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
    (reversed) own-column values (no randomness, no driver)."""
    return normalize_proba(
        (X.max(axis=0) - X) / (X.max(axis=0) - X.min(axis=0))
    )


class MissingValueGenerator(GeneratorMixin):
    """
    Base class for missing values transformer
    """

    def __init__(self, random_generator: numpy.random.Generator = default_rng(seed=0)):
        """
        :param random_generator: A random generator
        """
        self.random_generator = random_generator
        self.missing_values_indices_ = None

    def _positional_names(self, X):
        """Synthesized variable names for graph-less data:
        ("x0", ..., "x{k-1}")."""
        return tuple(f"x{i}" for i in range(X.shape[1]))

    def _write_missing(self, X, masks):
        """Write NaNs from ``{name: bool-mask}``; populate
        ``missing_values_indices_``.

        Column-major order matches the legacy index list. The caller
        guarantees exactly one mask per column, in column order.
        """
        indices = []
        for col, (name, mask) in enumerate(masks.items()):
            X[mask, col] = np.nan
            indices.extend(
                zip(np.where(mask)[0].tolist(), [col] * int(mask.sum()))
            )
        self.missing_values_indices_ = indices
        return X

    @abc.abstractmethod
    def generate(self, X, y, **params):
        """
        Abstract method to generate missing values in the input data.
        This should be overridden by subclasses.

        :param X: Input features, can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: Target variable, can be a pandas Series or a numpy array.
                  If None, it is assumed that the target is not provided and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
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

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the MissingCompletelyAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` completely at random according to a specified percentage.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
        """
        assert 0 <= percentage_missing <= 1
        # Route the row selection through the shared ExactCountSelector
        # preset (uniform draws, exact count per column) and write the
        # NaNs from the returned {variable: bool-mask} dict.
        selector = ExactCountSelector(
            percentage_missing, self._positional_names(X),
            rng=self.random_generator,
        )
        masks = selector(X, y)
        self._write_missing(X, masks)

        return X, y


class DummyMissingAtRandom(MissingValueGenerator):
    """
    A generator that removes values at random (MAR [1]),
    where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
    where j is randomly chosen.

    See also [1] https://stefvanbuuren.name/fimd/sec-MCAR.html

    Routes the row selection through the shared ``ExactCountSelector``
    preset (exact-count, optional soft probabilities).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the DummyMissingAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing: float = 0.1):
        """
        Introduces missing values into the input features `X` at random based on another feature,
        where the probability of a data instance X[_,i] missing depends upon another feature X[_,j],
        and j is randomly chosen.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
        """
        assert 0 <= percentage_missing <= 1
        # Route the row selection through the shared ExactCountSelector
        # preset. The soft coupling (per-column driver drawn uniformly
        # among the OTHER columns) enters via per-row weights; the driver
        # draws happen inside the weights callable, BEFORE the row draws
        # -- exactly the legacy RNG call order.
        selector = ExactCountSelector(
            percentage_missing, self._positional_names(X),
            weights=lambda X_, y_=None: _mar_weights(X_, self.random_generator),
            rng=self.random_generator,
        )
        masks = selector(X, y)
        self._write_missing(X, masks)

        return X, y


class DummyMissingNotAtRandom(MissingValueGenerator):
    """
    A generator that removes values not at random (MNAR [1]),
    where the probability of a data instance X[i,j] missing depends linearly upon its own value.
    A data point X[i,j] = max(X[:,j]) has a missing probability of 1.
    A data point X[i,j] = min(X[:,j]) has a missing probability of 0.

    Routes the row selection through the shared ``ExactCountSelector``
    preset (exact-count, optional soft probabilities).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initializes the DummyMissingNotAtRandom class with a specified random generator.

        :param random_generator: A NumPy random number generator instance. Defaults to a new instance of `default_rng` with seed 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y, percentage_missing):
        """
        Introduces missing values into the input features `X` not at random, where the probability of a data instance X[i,j] missing
        depends linearly upon its own value. Specifically, a data point X[i,j] = max(X[:,j]) has a missing probability of 1, and a
        data point X[i,j] = min(X[:,j]) has a missing probability of 0.

        :param X: The input features, which can be a pandas DataFrame or a numpy array.
        :type X: Union[pandas.DataFrame, numpy.ndarray]
        :param y: The target variable, which can be a pandas Series or a numpy array.
                  If not provided, it is assumed that the target is not needed and will be ignored.
        :type y: Union[pandas.Series, numpy.ndarray, None], optional
        :param percentage_missing: The proportion of values to be replaced with missing values, expressed as a float between 0 and 1.
        :type percentage_missing: float
        :return: A tuple containing the modified input features `Xt` with introduced missing values and the original target `y`.
        :rtype: Tuple[Union[pandas.DataFrame, numpy.ndarray], Union[pandas.Series, numpy.ndarray, None]]
        """
        assert 0 <= percentage_missing <= 1
        # Route the row selection through the shared ExactCountSelector
        # preset. The self-coupling (probability of missing grows with the
        # own value) enters via per-row weights.
        selector = ExactCountSelector(
            percentage_missing, self._positional_names(X),
            weights=lambda X_, y_=None: _mnar_weights(X_),
            rng=self.random_generator,
        )
        masks = selector(X, y)
        self._write_missing(X, masks)

        return X, y
