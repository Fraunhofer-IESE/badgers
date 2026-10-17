import abc

import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

from badgers.core.base import GeneratorMixin
from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.utils import resolve_class_column


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
    def generate(self, X, y, **params):
        """
        Abstract method to generate noisy data. Must be implemented by subclasses.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array).
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
    data is transformed back. The class data lives in a column of ``X``
    (``y`` is not used for selection and is passed through unchanged).
    """

    def __init__(self, random_generator=default_rng(seed=0)):
        """
        Initialize the GaussianNoiseClassesGenerator with a specified random number generator.

        :param random_generator: A random number generator instance from numpy.random,
                                 used to introduce randomness in the noise generation process.
        """
        super().__init__(random_generator=random_generator)

    @preprocess_inputs
    def generate(self, X, y=None, *, selector=None, noise_std_per_class=dict()):
        """
        Add Gaussian white noise to the rows fired by the selector, with a
        standard deviation looked up per class value.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (optional, passed through unchanged; not used for selection).
        :param selector: A selector deciding which rows receive noise. A
                         :class:`badgers.core.selectors.ClassSelector` over
                         the class column (typical), or any selector whose
                         masks OR to the fired rows. The per-class lookup
                         requires ``selector.column_mapping`` to resolve the
                         class column position. Required (a per-class rule
                         has no class-free default).
        :param noise_std_per_class: A dictionary mapping class values to the
            noise standard deviation for that class. Fired rows whose class
            value is absent from the dict receive no noise.
        :return: Xt, yt where Xt is the noisy input features and yt is the unchanged target variable y.
        """
        if selector is None:
            raise ValueError(
                "GaussianNoiseClassesGenerator needs a selector deciding "
                "which rows receive noise (e.g. ClassSelector over the "
                "class column): classes are columns of X, y is not used "
                "for selection."
            )
        masks = list(selector(X, y).values())
        fired = masks[0] if len(masks) == 1 else np.logical_or.reduce(masks)
        if not fired.any():
            return X, y
        # resolve the class column from the selector's column_mapping
        class_col = resolve_class_column(selector, X)
        if class_col is None:
            raise ValueError(
                "a per-class noise lookup requires selector.column_mapping "
                "to resolve the class column position"
            )
        # standardize X
        scaler = StandardScaler()
        # fit, transform
        scaler.fit(X)
        Xt = scaler.transform(X)
        # noise per class value among the fired rows
        for value in np.unique(X[fired, class_col]):
            if value not in noise_std_per_class:
                continue
            noise_std = noise_std_per_class[value]
            rows = fired & (X[:, class_col] == value)
            noise = self.random_generator.normal(
                loc=0, scale=noise_std, size=(int(rows.sum()), X.shape[1])
            )
            Xt[rows] += noise
        # inverse standardization
        Xt = scaler.inverse_transform(Xt)
        return Xt, y


class AdditiveNoiseGenerator(NoiseGenerator):
    """
    A generator that adds noise drawn from an arbitrary distribution to the
    tabular data (in-place, shape and labels unchanged).

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
    def generate(self, X, y, distribution, noise_std=1.0, perturbation_nodes=None):
        """
        Add noise drawn from an arbitrary distribution to the input data.

        :param X: Input features (pandas DataFrame or numpy array).
        :param y: Target variable (pandas Series or numpy array), which remains unchanged.
        :param distribution: A frozen scipy distribution object (e.g. ``scipy.stats.laplace()``)
            that exposes ``.rvs(size=..., random_state=...)``. Drawn samples are added as-is.
        :param noise_std: Global multiplier applied to the drawn noise (default 1.0 = raw draw).
        :param perturbation_nodes: Optional list of column indices to perturb.
            If None, noise is added to all columns.
        :return: Xt, yt where Xt is the noisy input features and yt is the unchanged target variable y.
        """
        rvs = getattr(distribution, "rvs", None)
        if not callable(rvs):
            raise TypeError(
                "distribution must be a frozen scipy distribution object (e.g. "
                "scipy.stats.laplace(scale=1)) exposing .rvs(size=..., random_state=...); "
                f"got {distribution!r}"
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        noise = rvs(size=X.shape, random_state=self.random_generator) * noise_std
        if perturbation_nodes is not None:
            perturbation_nodes = np.asarray(perturbation_nodes, dtype=int)
            if perturbation_nodes.size and (
                perturbation_nodes.min() < 0 or perturbation_nodes.max() >= X.shape[1]
            ):
                raise ValueError(
                    f"perturbation_nodes out of range for X with {X.shape[1]} columns"
                )
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[perturbation_nodes] = True
            noise[:, ~mask] = 0.0
        return X + noise, y
