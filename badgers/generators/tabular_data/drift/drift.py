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
    r"""
    Randomly shift (geometrical translation) values of each column independently of one another.
    Data are first standardized (mean = 0, var = 1) and a random number is added to each column.
    The ith columns is simply translated: `$x_i \leftarrow x_i + \epsilon_i$`
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
        r"""
        Randomly shift (geometrical translation) values of each column independently of one another.
        Data are first standardized (mean = 0, var = 1), and a random number drawn from a normal distribution
        with mean 0 and standard deviation 1. `shift_std` is then added to each column.
        The ith column is simply translated: `$x_i \leftarrow x_i + \epsilon_i$`, where $\epsilon_i \sim \mathcal{N}(0, \text{shift\_std})$.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable, a 1D array-like object (optional). Not used in this implementation.
        :param shift_std: Magnitude of the random shift applied to each column. Can be a single float
                          (applied to all columns) or an array of floats (one per column). The values
                          are interpreted as standard deviations: a random number drawn from a normal
                          distribution is multiplied by ``shift_std`` (absolute value used as the
                          scale, the sign used as the shift direction). A negative entry shifts the
                          column in the negative direction.
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        # check that if shift_std is an array, it is of length == X.shape[1]
        shift_std = np.asarray(shift_std, dtype=float)
        if shift_std.ndim == 1 and shift_std.size > 1:
            assert shift_std.size == X.shape[
                1], f"shift_std length ({shift_std.size}) must equal number of features in X ({X.shape[1]})"
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # generate random values for the shift for each column
        # scale must be non-negative: use |shift_std| as the scale and sign(shift_std) as the direction
        scale = np.broadcast_to(np.abs(shift_std), (X.shape[1],))
        sign = np.broadcast_to(np.sign(shift_std), (X.shape[1],))
        shift = self.random_generator.normal(loc=0, scale=scale, size=X.shape[1]) * sign
        # add shift
        Xt += shift
        # inverse transform
        return scaler.inverse_transform(Xt), y


class FixedShiftGenerator(DriftGenerator):
    r"""
    Deterministically shift (geometrical translation) values of each column
    independently of one another — the non-random counterpart of
    :class:`RandomShiftGenerator`.

    Data are first standardized (mean = 0, var = 1), then each column is
    translated by exactly ``shift_std``: the ith column becomes
    `$x_i \leftarrow x_i + \text{shift\_std}_i$`. No random number is drawn,
    so repeated calls with the same input return the same output.
    """

    def __init__(self):
        """Initialize the FixedShiftGenerator (no random generator needed)."""
        super().__init__(random_generator=None)

    @preprocess_inputs
    def generate(self, X, y=None, shift_std: Union[float, np.array, list] = 0.1):
        r"""
        Deterministically shift each column by exactly ``shift_std``.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable (optional, passed through unchanged; not used in this implementation).
        :param shift_std: Shift applied to each column. A single float (applied to all columns)
                          or an array of floats (one per column). The value is the shift in
                          units of column standard deviation (data are standardized first):
                          a positive entry moves the column up, a negative entry moves it
                          down. The same input always produces the same output.
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        shift_std = np.asarray(shift_std, dtype=float)
        if shift_std.ndim == 1 and shift_std.size > 1:
            assert shift_std.size == X.shape[
                1], f"shift_std length ({shift_std.size}) must equal number of features in X ({X.shape[1]})"
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # exact shift per column: |shift_std| magnitude, sign(shift_std) direction
        shift = np.broadcast_to(shift_std, (X.shape[1],))
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
    def generate(self, X, y=None, *, selector=None, shift_std=0.1, variables=None, class_column=None):
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
        :param variables: Columns to shift. Either a list of column positions
                          (ints) or a list of variable names resolvable through
                          ``selector.column_mapping``. Defaults to all columns
                          except the class column — the class column is
                          identification metadata and is never shifted by
                          default (shifting it would move rows away from
                          their class values).
        :param class_column: The column holding the class values: either its
                          position (int) or a variable name resolvable through
                          ``selector.column_mapping``. The class column groups
                          the fired rows (one shift per class value) and is
                          never shifted itself. If omitted, it is recovered
                          from the selector's ``column_mapping``; when neither
                          is available, all fired rows are treated as one
                          class and every column may be shifted.
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
        # resolve the class column: explicit parameter first, selector sniffing as fallback
        class_col = class_column
        if class_col is not None and isinstance(class_col, str):
            mapping = getattr(selector, "column_mapping", None)
            pos = mapping.get(class_col) if mapping is not None else None
            if pos is None or not (0 <= pos < X.shape[1]):
                raise ValueError(f"cannot resolve class column {class_col!r} in X")
            class_col = pos
        elif class_col is None:
            class_col = resolve_class_column(selector, X)
        elif not (0 <= class_col < X.shape[1]):
            raise ValueError(f"class_column {class_col} is not a valid column position in X")
        stds = shift_std if isinstance(shift_std, dict) else None
        if stds is not None and class_col is None:
            raise ValueError(
                "a dict shift_std requires a class column: pass "
                "class_column (int or name) or use a selector with "
                "column_mapping to resolve the class column position"
            )
        # resolve which columns are shifted
        if variables is not None:
            mapping = getattr(selector, "column_mapping", None)
            cols = []
            for var in variables:
                pos = mapping.get(var) if mapping is not None and isinstance(var, str) else var
                if pos is None or not (0 <= pos < X.shape[1]):
                    raise ValueError(f"cannot resolve shift column {var!r} in X")
                cols.append(pos)
            cols = np.array(sorted(set(cols)), dtype=int)
        else:
            # all columns except the class column
            cols = np.array([i for i in range(X.shape[1]) if i != class_col], dtype=int)
        if cols.size == 0:
            return X, y
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
            shift = self.random_generator.normal(loc=0, scale=std, size=cols.size)
            Xt[np.ix_(rows, cols)] += shift
        # inverse transform
        return scaler.inverse_transform(Xt), y


class FixedShiftClassesGenerator(DriftGenerator):
    """
    Deterministically shift the rows fired by a selector, independently per
    class — the non-random counterpart of :class:`RandomShiftClassesGenerator`.

    Data are first standardized (mean = 0, var = 1); each class value of the
    class column gets an exact shift (no draw), applied to the rows fired by
    the selector. The class data lives in a column of ``X`` (``y`` is not
    used for selection). Repeated calls with the same input return the same
    output.
    """

    def __init__(self):
        """Initialize the FixedShiftClassesGenerator (no random generator needed)."""
        super().__init__(random_generator=None)

    @preprocess_inputs
    def generate(self, X, y=None, *, selector=None, shift=0.1, variables=None, class_column=None):
        """
        Deterministically shift the rows fired by the selector, one exact
        shift per class value of the class column.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param y: Target variable (optional, passed through unchanged; not used for selection).
        :param selector: A selector deciding which rows are shifted (required; its masks
                         are OR-ed to the affected rows).
        :param shift: The exact shift applied to the fired rows of each class, in units of
                      column standard deviation (data are standardized first). A single
                      float (same shift for every class and column), a vector (one entry
                      per shifted column, same for every class), or a dict
                      ``{class_value: float or vector}`` — fired rows whose class value
                      is absent from the dict are not shifted.
        :param variables: Columns to shift. Either a list of column positions (ints) or a
                          list of variable names resolvable through
                          ``selector.column_mapping``. Defaults to all columns except the
                          class column.
        :param class_column: The column holding the class values: either its position
                             (int) or a variable name resolvable through
                             ``selector.column_mapping``. The class column groups the
                             fired rows (one shift per class value) and is never shifted
                             itself. If omitted, it is recovered from the selector's
                             ``column_mapping``; when neither is available, all fired
                             rows are treated as one class and every column may be
                             shifted.
        :return: A tuple containing the modified feature matrix `X'` and the original target `y`.
        """
        if selector is None:
            raise ValueError(
                "FixedShiftClassesGenerator needs a selector deciding "
                "which rows shift (e.g. ClassSelector over the class "
                "column): classes are columns of X, y is not used for "
                "selection."
            )
        masks = list(selector(X, y).values())
        fired = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
        if not fired.any():
            return X, y
        # resolve the class column: explicit parameter first, selector sniffing as fallback
        class_col = class_column
        if class_col is not None and isinstance(class_col, str):
            mapping = getattr(selector, "column_mapping", None)
            pos = mapping.get(class_col) if mapping is not None else None
            if pos is None or not (0 <= pos < X.shape[1]):
                raise ValueError(f"cannot resolve class column {class_col!r} in X")
            class_col = pos
        elif class_col is None:
            class_col = resolve_class_column(selector, X)
        elif not (0 <= class_col < X.shape[1]):
            raise ValueError(f"class_column {class_col} is not a valid column position in X")
        shifts = shift if isinstance(shift, dict) else None
        if shifts is not None and class_col is None:
            raise ValueError(
                "a dict shift requires a class column: pass "
                "class_column (int or name) or use a selector with "
                "column_mapping to resolve the class column position"
            )
        # resolve which columns are shifted
        if variables is not None:
            mapping = getattr(selector, "column_mapping", None)
            cols = []
            for var in variables:
                pos = mapping.get(var) if mapping is not None and isinstance(var, str) else var
                if pos is None or not (0 <= pos < X.shape[1]):
                    raise ValueError(f"cannot resolve shift column {var!r} in X")
                cols.append(pos)
            cols = np.array(sorted(set(cols)), dtype=int)
        else:
            # all columns except the class column
            cols = np.array([i for i in range(X.shape[1]) if i != class_col], dtype=int)
        if cols.size == 0:
            return X, y
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # one exact shift per class value among the fired rows
        if class_col is not None:
            values = np.unique(X[fired, class_col])
        else:
            values = np.array([None])
        for value in values:
            if shifts is not None:
                if value not in shifts:
                    continue
                shift_value = np.broadcast_to(np.asarray(shifts[value], dtype=float), (cols.size,))
            elif isinstance(shift, np.ndarray) and shift.ndim == 1:
                shift_value = np.broadcast_to(np.asarray(shift, dtype=float), (cols.size,))
            else:
                shift_value = np.full(cols.size, float(shift))
            rows = fired if class_col is None else (fired & (X[:, class_col] == value))
            Xt[np.ix_(rows, cols)] += shift_value
        # inverse transform
        return scaler.inverse_transform(Xt), y