import abc
from typing import Union

import numpy as np
from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.utils import resolve_class_column
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler


class DriftGenerator(GeneratorMixin):
    """
    Base class for transformers that add noise to tabular data
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the drift generator.
        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, y, **params):
        pass


class RandomShiftGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) values of each column independently of one another.
    Data are first standardized (mean = 0, var = 1) and a random number is added to each column.
    The ith columns is simply translated: `$x_i \left arrow x_i + \epsilon_i$`
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomShiftGenerator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, shift_std: Union[float, np.array, list] = 0.1):
        """
        Randomly shift (geometrical translation) values of each column independently of one another.
        Data are first standardized (mean = 0, var = 1), and a random number drawn from a normal distribution
        with mean 0 and standard deviation `shift_std` is added to each column.
        The ith column is simply translated: `$x_i \leftarrow x_i + \epsilon_i$`, where $\epsilon_i \sim \mathcal{N}(0, \text{shift\_std})$.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable, a 1D array-like object (optional). Not used in this implementation.
        :param shift_std: Standard deviation of the normal distribution from which the random shifts are drawn.
                          Can be a single float (applied to all columns) or an array of floats (one per column).
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        # check that if shift_std is an array, it is of length == X.shape[1]
        if hasattr(shift_std, '__len__'):
            assert len(shift_std) == X.shape[
                1], f"shift_std length ({len(shift_std)}) must equal number of features in X ({X.shape[1]})"
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift for each column
        shift = self.random_generator.normal(loc=0, scale=shift_std, size=X.shape[1])
        # add shift
        Xt += shift
        # inverse transform
        return scaler.inverse_transform(Xt), y


class RandomShiftClassesGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) the rows fired by a selector,
    independently per class.

    Data are first standardized (mean = 0, var = 1); for each class value
    of the class column a random shift is drawn and added to the FIRED rows
    of that class. The class data lives in a column of ``X`` (``y`` is not
    used for selection).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the RandomShiftClassesGenerator.

        :param random_generator: A NumPy random number generator used to generate random numbers.
                                 Defaults to a default random number generator seeded with 0.
        :type random_generator: numpy.random.Generator
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, *, selector=None, shift_std=0.1):
        """
        Randomly shift the rows fired by the selector, one independent
        shift per class value of the class column.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable (optional, passed through unchanged; not used for selection).
        :param selector: A selector deciding which rows are shifted. A
                         :class:`badgers.core.selectors.ClassSelector` over
                         the class column (typical), or any selector whose
                         masks OR to the fired rows. The per-class lookup
                         requires ``selector.column_mapping`` to resolve the
                         class column position. Required (a per-class rule
                         has no class-free default).
        :param shift_std: Standard deviation of the normal distribution from
                          which the shifts are drawn. A single float (same
                          magnitude for every class, one independent draw
                          per class) or a dict ``{class_value: float}``.
                          Fired rows whose class value is absent from the
                          dict are not shifted.
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        if selector is None:
            raise ValueError(
                "RandomShiftClassesGenerator needs a selector deciding "
                "which rows shift (e.g. ClassSelector over the class "
                "column): classes are columns of X, y is not used for "
                "selection."
            )
        masks = list(selector(X, y).values())
        fired = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
        if not fired.any():
            return X, y
        # resolve the class column from the selector's column_mapping
        class_col = resolve_class_column(selector, X)
        stds = shift_std if isinstance(shift_std, dict) else None
        if stds is not None and class_col is None:
            raise ValueError(
                "a dict shift_std requires selector.column_mapping to "
                "resolve the class column position"
            )
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # one independent shift per class value among the fired rows
        if class_col is not None:
            values = np.unique(X[fired, class_col])
        else:
            values = np.array([None])
        for value in values:
            if stds is not None:
                if value not in stds:
                    continue
                std = stds[value]
            else:
                std = shift_std
            rows = fired if class_col is None else (fired & (X[:, class_col] == value))
            shift = self.random_generator.normal(loc=0, scale=std, size=X.shape[1])
            Xt[rows] += shift
        # inverse transform
        return scaler.inverse_transform(Xt), y