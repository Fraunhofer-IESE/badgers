import abc

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import AlwaysSelector, fired, resolve_class_column


class NoiseGenerator(GeneratorMixin):
    """
    Base class for generators that add noise to tabular data.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the NoiseGenerator with a specified random number generator.

        :param random_generator: A random number generator instance from numpy.random,
                                 used to introduce randomness in the noise generation process.
        """
        self.random_generator = random_generator

    @abc.abstractmethod
    def generate(self, X, **params):
        """
        Abstract method to generate noisy data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param params: Additional parameters required for noise generation.
        """
        pass


class GaussianNoiseClassesGenerator(NoiseGenerator):
    """
    A generator that adds Gaussian white noise to the rows fired by a
    selector, with a per-class magnitude.

    The data is first standardized (each column mean 0, variance 1); noise
    is drawn per fired row's class value from a normal distribution whose
    standard deviation is looked up in ``noise_std_per_class``, then the
    data is transformed back. The class data lives in a column of ``X``.
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the GaussianNoiseClassesGenerator with a specified random number generator.

        :param random_generator: A random number generator instance from numpy.random,
                                 used to introduce randomness in the noise generation process.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, *, selector=None, noise_std_per_class=dict()):
        """
        Add Gaussian white noise to the rows fired by the selector, with a
        standard deviation looked up per class value.

        :param X: Input features (pandas DataFrame or numpy array).
        :param selector: A selector deciding which rows receive noise. A
                         :class:`badgers.core.selectors.ClassSelector` over
                         the class column (typical), or any selector whose
                         masks OR to the fired rows. The per-class lookup
                         reads the selector's ``class_column`` to resolve
                         the class column position. Required (``None``
                         raises -- a per-class rule has no relevant
                         default: the selector owns the class column).
        :param noise_std_per_class: A dictionary mapping class values to the
            noise standard deviation for that class. Every class value
            present among the fired rows must have an entry (a missing
            class value raises ``ValueError``); the keys are normalized
            against the class column's dtype, so plain-int keys work for a
            float-encoded class column.
        :return: The noisy input features.
        """
        if selector is None:
            raise ValueError(
                "GaussianNoiseClassesGenerator needs a selector deciding "
                "which rows receive noise (e.g. ClassSelector over the "
                "class column): classes are columns of X with discrete values."
            )
        firing = fired(X, selector, self.random_generator).any(axis=1)
        if not firing.any():
            return X
        # resolve the class column from the selector's class_column
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
                    "a per-class noise lookup requires a selector exposing "
                    "class_column (e.g. ClassSelector) to resolve the "
                    "class column position"
                )
            raise ValueError(
                f"the selector's class_column {col} is not a valid column "
                f"position in X with {X.shape[1]} columns"
            )
        # standardize X
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # normalize the dict keys against the class column's dtype (a
        # plain-int key works for a float-encoded class column and vice
        # versa; wrong-kind keys, e.g. strings for numeric classes, raise)
        if noise_std_per_class:
            keys = list(noise_std_per_class.keys())
            try:
                normalized = np.asarray(keys, dtype=X[:, class_col].dtype)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"noise_std_per_class keys {keys} cannot be compared "
                    f"with the class column values (dtype "
                    f"{X[:, class_col].dtype})"
                ) from exc
            noise_std_per_class = dict(zip(normalized, noise_std_per_class.values()))
        # noise per class value among the fired rows
        for value in np.unique(X[firing, class_col]):
            if value not in noise_std_per_class:
                raise ValueError(
                    f"noise_std_per_class has no entry for the fired class "
                    f"value {value!r} (class column {class_col}); entries: "
                    f"{sorted(noise_std_per_class)}"
                )
            noise_std = noise_std_per_class[value]
            rows = firing & (X[:, class_col] == value)
            noise = self.random_generator.normal(
                loc=0, scale=noise_std, size=(int(rows.sum()), X.shape[1])
            )
            Xt[rows] += noise
        # inverse standardization
        Xt = scaler.inverse_transform(Xt)
        return Xt


class AdditiveNoiseGenerator(NoiseGenerator):
    """
    A generator that adds noise drawn from an arbitrary distribution to the
    rows fired by a selector (shape unchanged).

    The distribution is specified as a *frozen* scipy distribution object
    (e.g. ``scipy.stats.laplace(scale=1)``, ``scipy.stats.t(df=3)``). What you
    pass is what is added — the generator never rescales it. ``noise_std`` is
    a global magnitude multiplier on the drawn sample (1.0 = pure raw draw).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the AdditiveNoiseGenerator.

        :param random_generator: A random number generator instance from numpy.random.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, distribution, noise_std=1.0, *, selector=None):
        """
        Add noise drawn from an arbitrary distribution to the cells fired
        by the selector.

        The selector is the single source of truth: its returned dict
        ``{column_position: spec}`` defines BOTH which columns receive
        noise (the keys) and which rows within each column are affected
        (the realized spec). Columns absent from the selector's output
        are left untouched.

        :param X: Input features (pandas DataFrame or numpy array).
        :param distribution: A frozen scipy distribution object (e.g.
            ``scipy.stats.laplace(scale=1)``) that exposes
            ``.rvs(size=..., random_state=...)``. Drawn samples are added
            as-is.
        :param noise_std: Global multiplier applied to the drawn noise
            (default 1.0 = raw draw).
        :param selector: A selector deciding which rows and columns
            receive noise. A preset from :mod:`badgers.core.selectors`
            (``AlwaysSelector``, ``BernoulliSelector``,
            ``ExactCountSelector``, ``TopQuantileSelector``,
            ``ClassSelector``, ``ProportionalSelector``, or any callable
            returning ``dict[int, spec]``). ``None`` = the relevant
            default selector ``AlwaysSelector`` over every column (noise
            everywhere).
        :return: The noisy input features.
        """
        rvs = getattr(distribution, "rvs", None)
        if not callable(rvs):
            raise TypeError(
                "distribution must be a frozen scipy distribution object "
                "(e.g. scipy.stats.laplace(scale=1)) exposing "
                ".rvs(size=..., random_state=...); "
                f"got {distribution!r}"
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        # Draw the full noise matrix (RNG consumed once regardless of masking)
        noise = rvs(size=X.shape, random_state=self.random_generator) * noise_std

        if selector is None:
            # the relevant default: noise on every cell
            selector = AlwaysSelector(columns=range(X.shape[1]))

        # Realized firing per the selector's own column scope; cells not
        # fired (or in columns the selector does not cover) get no noise.
        # A bare user callable without a ``columns`` attribute covers
        # every column of X (fired() resolves call-time targets the same
        # way -- keep in sync).
        field = fired(X, selector, self.random_generator)
        covered = getattr(selector, "columns", None)
        if covered is None:
            # same default as _resolve_targets in badgers/core/selectors/base.py -- keep in sync
            covered = range(X.shape[1])
        full = np.zeros(X.shape, dtype=bool)
        full[:, list(covered)] = field
        noise[~full] = 0.0
        return X + noise
