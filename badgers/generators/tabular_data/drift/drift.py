import abc
from typing import Union

import numpy as np
from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import fired, resolve_class_column
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
    def generate(self, X, **params):
        pass


class RandomShiftGenerator(DriftGenerator):
    r"""
    Randomly shift (geometrical translation) the rows fired by a selector,
    independently per column.
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
    def generate(
        self,
        X,
        shift_std: Union[float, np.array, list] = 0.1,
        *,
        selector=None,
    ):
        r"""
        Randomly shift (geometrical translation) the rows fired by the selector, independently per column.
        Data are first standardized (mean = 0, var = 1), and a random number drawn from a normal distribution
        with mean 0 and standard deviation 1. `shift_std` is then added to each column.
        The ith column is simply translated: `$x_i \leftarrow x_i + \epsilon_i$`, where $\epsilon_i \sim \mathcal{N}(0, \text{shift\_std})$.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param shift_std: Magnitude of the random shift applied to each column. Can be a single float
                          (applied to all columns) or an array of floats (one per column). The values
                          are interpreted as standard deviations: a random number drawn from a normal
                          distribution is multiplied by ``shift_std`` (absolute value used as the
                          scale, the sign used as the shift direction). A negative entry shifts the
                          column in the negative direction.
        :param selector: A selector deciding which rows are shifted. A preset from
                         :mod:`badgers.core.selectors` (``AlwaysSelector``,
                         ``BernoulliSelector``, ``ExactCountSelector``,
                         ``TopQuantileSelector``, ``ClassSelector``, ...) or
                         any callable accepting X and returning
                         ``dict[int, spec]`` (keys are column positions;
                         value-free presets return bool masks --
                         ``AlwaysSelector`` the scalar ``1.0`` -- value-driven
                         presets ``[0, 1]`` firing specs as ``(n,)`` arrays),
                         realized internally via ``fired`` (row-level
                         union ``.any(axis=1)``). ``None`` =
                         every row is shifted (backward compatible).
        :return: The modified feature matrix `X'`.
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
        # add shift to the FIRED rows only (masks computed on the un-mutated input)
        rows = np.flatnonzero(fired(X, selector, self.random_generator).any(axis=1))
        Xt[rows] += shift
        # inverse transform
        return scaler.inverse_transform(Xt)


class FixedShiftGenerator(DriftGenerator):
    r"""
    Deterministically shift (geometrical translation) the rows fired by a
    selector, independently per column — the non-random counterpart of
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
    def generate(self, X, shift_std: Union[float, np.array, list] = 0.1, *, selector=None):
        r"""
        Deterministically shift the rows fired by the selector, by exactly
        ``shift_std`` per column.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param shift_std: Shift applied to each column. A single float (applied to all columns)
                          or an array of floats (one per column). The value is the shift in
                          units of column standard deviation (data are standardized first):
                          a positive entry moves the column up, a negative entry moves it
                          down. The same input always produces the same output.
        :param selector: A selector deciding which rows are shifted. A preset from
                         :mod:`badgers.core.selectors` (``AlwaysSelector``,
                         ``BernoulliSelector``, ``ExactCountSelector``,
                         ``TopQuantileSelector``, ``ClassSelector``, ...) or
                         any callable accepting X and returning
                         ``dict[int, spec]`` (keys are column positions;
                         value-free presets return bool masks --
                         ``AlwaysSelector`` the scalar ``1.0`` -- value-driven
                         presets ``[0, 1]`` firing specs as ``(n,)`` arrays),
                         realized internally via ``fired`` (row-level
                         union ``.any(axis=1)``). ``None`` =
                         every row is shifted (backward compatible).
        :return: The modified feature matrix `X'`.
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
        # add shift to the FIRED rows only (masks computed on the un-mutated input).
        # Deterministic generator: realizing the selector's specs consumes an
        # rng, but the shift itself draws nothing -- bind a fixed private
        # stream so repeated calls return the same output.
        rng = self.random_generator if self.random_generator is not None else default_rng(0)
        rows = np.flatnonzero(fired(X, selector, rng).any(axis=1))
        Xt[rows] += shift
        # inverse transform
        return scaler.inverse_transform(Xt)


class RandomShiftClassesGenerator(DriftGenerator):
    """
    Randomly shift (geometrical translation) the rows fired by a selector,
    independently per class.

    Data are first standardized (mean = 0, var = 1); for each class value
    of the class column a random shift is drawn and added to the FIRED rows
    of that class. The class data lives in a column of ``X``.
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
    def generate(self, X, *, selector=None, shift_std=0.1, variables=None, class_column=None):
        """
        Randomly shift the rows fired by the selector, one independent
        shift per class value of the class column.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param selector: A selector deciding which rows are shifted. A
                         :class:`badgers.core.selectors.ClassSelector` over
                         the class column (typical), or any selector whose
                         masks OR to the fired rows. The per-class lookup
                         reads the selector's ``class_column`` to resolve
                         the class column position. Required (a per-class
                         rule has no class-free default).
        :param shift_std: Standard deviation of the normal distribution from
                          which the shifts are drawn. Entries are SIGNED: the
                          magnitude (``abs``) is the scale of the draw and the
                          sign is the shift direction — a negative entry drifts
                          the class downward. A single float (same magnitude for
                          every class, one independent draw per class) or a dict
                          ``{class_value: float}``. Every class value present
                          among the fired rows must have an entry (a missing
                          class value raises ``ValueError``); the keys are
                          normalized against the class column's dtype, so
                          plain-int keys work for a float-encoded class column.
        :param variables: Columns to shift, as a list of column positions
                          (ints). Defaults to all columns
                          except the class column — the class column is
                          identification metadata and is never shifted by
                          default (shifting it would move rows away from
                          their class values).
        :return: The modified feature matrix `X'`.
        """
        if selector is None:
            raise ValueError(
                "RandomShiftClassesGenerator needs a selector deciding "
                "which rows shift (e.g. ClassSelector over the class "
                "column): classes are columns of X."
            )
        firing = fired(X, selector, self.random_generator).any(axis=1)
        if not firing.any():
            return X
        # the selector owns the class information: resolve the class column
        # from the selector's class_column (never a generator parameter)
        class_col = resolve_class_column(selector, X)
        if class_col is None:
            # distinguish the two "None" causes: no class_column/column
            # attribute at all, or an out-of-range position (silent None
            # from resolve_class_column)
            col = getattr(selector, "class_column", None)
            if col is None:
                col = getattr(selector, "column", None)
            if col is None:
                raise ValueError(
                    "a per-class shift requires a selector exposing "
                    "class_column (e.g. ClassSelector) to resolve the "
                    "class column position"
                )
            raise ValueError(
                f"the selector's class_column {col} is not a valid column "
                f"position in X with {X.shape[1]} columns"
            )
        stds = shift_std if isinstance(shift_std, dict) else None
        # normalize the dict keys against the class column's dtype (a
        # plain-int key works for a float-encoded class column and vice
        # versa; wrong-kind keys, e.g. strings for numeric classes, raise)
        if stds:
            keys = list(stds.keys())
            try:
                normalized = np.asarray(keys, dtype=X[:, class_col].dtype)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"shift_std keys {keys} cannot be compared "
                    f"with the class column values (dtype "
                    f"{X[:, class_col].dtype})"
                ) from exc
            stds = dict(zip(normalized, stds.values()))
        # resolve which columns are shifted
        if variables is not None:
            cols = []
            for var in variables:
                if not (isinstance(var, (int, np.integer))
                        and 0 <= var < X.shape[1]):
                    raise ValueError(f"cannot resolve shift column {var!r} in X")
                cols.append(var)
            cols = np.array(sorted(set(cols)), dtype=int)
        else:
            # all columns except the class column
            cols = np.array([i for i in range(X.shape[1]) if i != class_col], dtype=int)
        if cols.size == 0:
            return X
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # one independent shift per class value among the fired rows
        values = np.unique(X[firing, class_col])
        for value in values:
            if stds is not None:
                if value not in stds:
                    raise ValueError(
                        f"shift_std has no entry for the fired class "
                        f"value {value!r} (class column {class_col}); "
                        f"entries: {sorted(stds)}"
                    )
                std = stds[value]
            else:
                std = shift_std
            rows = firing & (X[:, class_col] == value)
            # signed magnitude: |std| is the scale, sign(std) the direction
            shift = self.random_generator.normal(loc=0, scale=abs(std), size=cols.size)
            Xt[np.ix_(rows, cols)] += np.sign(std) * shift
        # inverse transform
        return scaler.inverse_transform(Xt)


class FixedShiftClassesGenerator(DriftGenerator):
    """
    Deterministically shift the rows fired by a selector, independently per
    class — the non-random counterpart of :class:`RandomShiftClassesGenerator`.

    Data are first standardized (mean = 0, var = 1); each class value of the
    class column gets an exact shift (no draw), applied to the rows fired by
    the selector. The class data lives in a column of ``X``. Repeated calls
    with the same input return the same
    output.
    """

    def __init__(self):
        """Initialize the FixedShiftClassesGenerator (no random generator needed)."""
        super().__init__(random_generator=None)

    @preprocess_inputs
    def generate(self, X, *, selector=None, shift=0.1, variables=None):
        """
        Deterministically shift the rows fired by the selector, one exact
        shift per class value of the class column.

        :param X: Input features, a 2D array-like object (e.g., a Pandas DataFrame or a NumPy array).
        :param selector: A selector deciding which rows are shifted (required; its masks
                         are OR-ed to the affected rows). The per-class lookup
                         reads the selector's ``class_column`` to resolve
                         the class column position.
        :param shift: The exact shift applied to the fired rows of each class, in units of
                      column standard deviation (data are standardized first). A single
                      float (same shift for every class and column), a vector (one entry
                      per shifted column, same for every class), or a dict
                      ``{class_value: float or vector}``. Every class value present
                      among the fired rows must have an entry (a missing
                      class value raises ``ValueError``); the keys are
                      normalized against the class column's dtype, so
                      plain-int keys work for a float-encoded class column.
        :param variables: Columns to shift, as a list of column positions (ints).
                          Defaults to all columns except the
                          class column.
        :return: The modified feature matrix `X'`.
        """
        if selector is None:
            raise ValueError(
                "FixedShiftClassesGenerator needs a selector deciding "
                "which rows shift (e.g. ClassSelector over the class "
                "column): classes are columns of X."
            )
        # Deterministic generator: realizing the selector's specs consumes an
        # rng, but the shift itself draws nothing -- bind a fixed private
        # stream so repeated calls return the same output.
        rng = self.random_generator if self.random_generator is not None else default_rng(0)
        firing = fired(X, selector, rng).any(axis=1)
        if not firing.any():
            return X
        # the selector owns the class information: resolve the class column
        # from the selector's class_column (never a generator parameter)
        class_col = resolve_class_column(selector, X)
        if class_col is None:
            # distinguish the two "None" causes: no class_column/column
            # attribute at all, or an out-of-range position (silent None
            # from resolve_class_column)
            col = getattr(selector, "class_column", None)
            if col is None:
                col = getattr(selector, "column", None)
            if col is None:
                raise ValueError(
                    "a per-class shift requires a selector exposing "
                    "class_column (e.g. ClassSelector) to resolve the "
                    "class column position"
                )
            raise ValueError(
                f"the selector's class_column {col} is not a valid column "
                f"position in X with {X.shape[1]} columns"
            )
        shifts = shift if isinstance(shift, dict) else None
        # normalize the dict keys against the class column's dtype (a
        # plain-int key works for a float-encoded class column and vice
        # versa; wrong-kind keys, e.g. strings for numeric classes, raise)
        if shifts:
            keys = list(shifts.keys())
            try:
                normalized = np.asarray(keys, dtype=X[:, class_col].dtype)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"shift keys {keys} cannot be compared "
                    f"with the class column values (dtype "
                    f"{X[:, class_col].dtype})"
                ) from exc
            shifts = dict(zip(normalized, shifts.values()))
        # resolve which columns are shifted
        if variables is not None:
            cols = []
            for var in variables:
                if not (isinstance(var, (int, np.integer))
                        and 0 <= var < X.shape[1]):
                    raise ValueError(f"cannot resolve shift column {var!r} in X")
                cols.append(var)
            cols = np.array(sorted(set(cols)), dtype=int)
        else:
            # all columns except the class column
            cols = np.array([i for i in range(X.shape[1]) if i != class_col], dtype=int)
        if cols.size == 0:
            return X
        # normalize X
        scaler = StandardScaler()
        scaler.fit(X)
        Xt = scaler.transform(X)
        # one exact shift per class value among the fired rows
        values = np.unique(X[firing, class_col])
        for value in values:
            if shifts is not None:
                if value not in shifts:
                    raise ValueError(
                        f"shift has no entry for the fired class "
                        f"value {value!r} (class column {class_col}); "
                        f"entries: {sorted(shifts)}"
                    )
                shift_value = np.broadcast_to(np.asarray(shifts[value], dtype=float), (cols.size,))
            elif isinstance(shift, np.ndarray) and shift.ndim == 1:
                shift_value = np.broadcast_to(np.asarray(shift, dtype=float), (cols.size,))
            else:
                shift_value = np.full(cols.size, float(shift))
            rows = firing & (X[:, class_col] == value)
            Xt[np.ix_(rows, cols)] += shift_value
        # inverse transform
        return scaler.inverse_transform(Xt)