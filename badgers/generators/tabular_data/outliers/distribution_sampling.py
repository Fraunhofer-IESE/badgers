import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.sampling import (
    HypersphereSampler,
    UniformOutOfDistributionSampler,
    ZScoreSampler,
)
from badgers.core.selectors import CountSelector
from badgers.generators.tabular_data.outliers import OutliersGenerator
from badgers.generators.tabular_data.outliers._selection import (
    select_and_seed,
    iter_selection_groups,
)


class HyperCubeSampling(OutliersGenerator):
    """
    Sampling uniformly at random within a hypercube encapsulating all the instances


    See section 6.1.1 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator):
        """
        Initialize the HyperCubeSampling with a random number generator.

        :param random_generator: An instance of numpy's random number generator (required; no default).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int | None = None, expansion: float = 0.0,
                 *, selector=None):
        """
        Randomly generates outliers by sampling uniformly at random within a
        hypercube encapsulating all the instances.

        The count comes from a selector: ``n_outliers`` is only the DEFAULT
        when no selector is passed (an internal ``CountSelector`` selects that
        many rows, or 10). The selected rows are the SEEDS: each appended
        outlier row starts as a copy of a selected row of ``X``, and every cell
        selected by the selector is overwritten with a value uniform in the
        (possibly expanded) data bounding box
        (:class:`badgers.core.sampling.UniformOutOfDistributionSampler`).
        A column not selected keeps the seed row's own value. The requested
        count must not exceed ``len(X)`` (a larger count raises).

        How to set the values for expansion.
        Per default expansion = 0, this means the hypercube will cover all the instances using min and max as boundaries
        It is possible to make the hypercube bigger, as proposed in [1] section 6.1.1

            Instances from Data usually determine the bounds a, b ∈ IRd . For this reason, this approach
            needs them as input. Tax and Duin [51] and Fan et al. [21] state only that these bounds should be
            chosen so that the hyper-rectangle encapsulates all genuine instances. [ 48] uses the minimum and
            maximum for each attribute obtained from Data. Theiler and Michael Cai [52] mention that the
            boundary does not need to be far beyond these boundaries. Abe et al. [1] propose the rule that the
            boundary should expand the minimum and maximum by 10%. Désir et al. [17] propose to expand
            the boundary by 20%.

        For expanding the hypercube by 10% use expansion = 0.1, for 20% use 0.2, etc.

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The default number of outliers when no selector is
            passed (ignored when a selector is given; must not exceed the
            number of rows of X).
        :param expansion: how much the hypercube shall be expanded beyond (min,max) range, in percent (0.1 == 10%)
        :param selector: optional selector from ``badgers.core.selectors``;
            its selected rows are the seeds and its columns the perturbed cells
            (``n_outliers`` must then not be passed).
        :return: The augmented feature matrix with added outliers.
        """
        if selector is not None and n_outliers is not None:
            raise ValueError(
                "n_outliers is only the default count (no selector): pass "
                "n_outliers without a selector, or give the count inside "
                "the selector")
        if expansion < 0:
            raise ValueError(f"expansion must be >= 0, got {expansion}")

        if selector is None:
            selector = CountSelector(
                columns=range(X.shape[1]),
                count=10 if n_outliers is None else n_outliers,
                rng=self.random_generator,
            )
        rng = self.random_generator

        field, targets, seed_rows = select_and_seed(X, selector, rng)
        outliers = X[seed_rows].astype(np.float64, copy=True)
        for group_rows, selected_cols in iter_selection_groups(field, targets, seed_rows):
            outliers[np.ix_(group_rows, selected_cols)] = (
                UniformOutOfDistributionSampler(expansion=expansion).sample(
                    rng, X, selected_cols, group_rows.size
                ))

        self._set_outlier_labels(len(X), seed_rows.size)
        return np.vstack([X, outliers])


class ZScoreSamplingGenerator(OutliersGenerator):
    """
    Randomly generates outliers as data points with a z-score > 3.

    Very similar to "GaussTail" in section 6.1.5 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator):
        """
        Initialize the ZScoreSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (required; no default).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int | None = None, scale: float = 1.0,
                 *, selector=None):
        r"""
        Randomly generates outliers as data points with a z-score > 3.

        The count comes from a selector: ``n_outliers`` is only the DEFAULT
        when no selector is passed (an internal ``CountSelector`` selects that
        many rows, or 10). The selected rows are the SEEDS: each appended
        outlier row starts as a copy of a selected row of ``X``, and every cell
        selected by the selector is overwritten with an out-of-distribution
        value from :class:`badgers.core.sampling.ZScoreSampler` (a random
        sign and ``3 + exponential(scale)`` standard deviations from the
        column mean). A column not selected keeps the seed row's own value.
        The requested count must not exceed ``len(X)`` (a larger count
        raises).

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The default number of outliers when no selector
            is passed (ignored when a selector is given; must not exceed the
            number of rows of X).
        :param scale: float (the scale parameter from
                    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
                    The scale parameter, :math:`\beta = 1/\lambda`. Must be
                    positive.
        :param selector: optional selector from ``badgers.core.selectors``;
            its selected rows are the seeds and its columns the perturbed cells
            (``n_outliers`` must then not be passed).
        :return: The augmented feature matrix with added outliers.
        """
        if selector is not None and n_outliers is not None:
            raise ValueError(
                "n_outliers is only the default count (no selector): pass "
                "n_outliers without a selector, or give the count inside "
                "the selector")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")

        if selector is None:
            selector = CountSelector(
                columns=range(X.shape[1]),
                count=10 if n_outliers is None else n_outliers,
                rng=self.random_generator,
            )
        rng = self.random_generator

        field, targets, seed_rows = select_and_seed(X, selector, rng)
        outliers = X[seed_rows].astype(np.float64, copy=True)
        for group_rows, selected_cols in iter_selection_groups(field, targets, seed_rows):
            outliers[np.ix_(group_rows, selected_cols)] = ZScoreSampler(scale=scale).sample(
                rng, X, selected_cols, group_rows.size
            )

        self._set_outlier_labels(len(X), seed_rows.size)
        return np.vstack([X, outliers])


class HypersphereSamplingGenerator(OutliersGenerator):
    """
    Generates outliers by sampling points from a hypersphere with radius at least 3 sigma

    Very similar to "GaussTail" in section 6.1.5 in [1]

    [1] Georg Steinbuss and Klemens Böhm. 2021.
        Generating Artificial Outliers in the Absence of Genuine Ones — A Survey.
        ACM Trans. Knowl. Discov. Data 15, 2, Article 30 (April 2021), 37 pages.
        https://doi.org/10.1145/3447822
    """

    def __init__(self, random_generator):
        """
        Initialize the HypersphereSamplingGenerator with a random number generator.

        :param random_generator: An instance of numpy's random number generator (required; no default).
        """
        super().__init__(random_generator)

    @preprocess_inputs
    def generate(self, X, n_outliers: int | None = None, scale: float = 1.0,
                 *, selector=None):
        r"""
        Generates outliers by sampling points from a hypersphere with radius at least 3 sigma.

        The count comes from a selector: ``n_outliers`` is only the DEFAULT
        when no selector is passed (an internal ``CountSelector`` selects that
        many rows, or 10). The selected rows are the SEEDS: each appended
        outlier row starts as a copy of a selected row of ``X``, and every cell
        selected by the selector is overwritten with a hypersphere point
        (:class:`badgers.core.sampling.HypersphereSampler`, radius
        ``3 + exponential(scale)`` in standardized space). A column not
        selected keeps the seed row's own value. The requested count must not
        exceed ``len(X)`` (a larger count raises).

        :param X: the input features (pandas DataFrame or numpy array).
        :param n_outliers: The default number of outliers when no selector
            is passed (ignored when a selector is given; must not exceed the
            number of rows of X).
        :param scale: float (the scale parameter from
                    https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html)
                    The scale parameter, :math:`\beta = 1/\lambda`. Must be
                    positive.
        :param selector: optional selector from ``badgers.core.selectors``;
            its selected rows are the seeds and its columns the perturbed cells
            (``n_outliers`` must then not be passed).
        :return: The augmented feature matrix with added outliers.
        """
        if selector is not None and n_outliers is not None:
            raise ValueError(
                "n_outliers is only the default count (no selector): pass "
                "n_outliers without a selector, or give the count inside "
                "the selector")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")

        if selector is None:
            selector = CountSelector(
                columns=range(X.shape[1]),
                count=10 if n_outliers is None else n_outliers,
                rng=self.random_generator,
            )
        rng = self.random_generator

        field, targets, seed_rows = select_and_seed(X, selector, rng)
        outliers = X[seed_rows].astype(np.float64, copy=True)
        for group_rows, selected_cols in iter_selection_groups(field, targets, seed_rows):
            outliers[np.ix_(group_rows, selected_cols)] = HypersphereSampler(scale=scale).sample(
                rng, X, selected_cols, group_rows.size
            )

        self._set_outlier_labels(len(X), seed_rows.size)
        return np.vstack([X, outliers])
