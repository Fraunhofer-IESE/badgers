"""Sampling strategies for generating values within or outside observed distributions.

WithinDistributionSampler subclasses generate plausible values from the
observed data distribution (used for exogenous nodes in causal generation,
and for within-distribution value synthesis: ``InstanceSampler`` copies
real values per cell).

OutOfDistributionSampler subclasses generate anomalous values outside the
observed distribution (used for perturbation nodes in causal generation).
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from badgers.core.utils import random_sign, random_spherical_coordinates


class Sampler:
    """Abstract base class for all samplers.

    Subclasses implement ``sample(rng, X, columns, n_samples)`` to generate
    values for the given columns.
    """

    def sample(self, rng, X, columns, n_samples):
        """Sample ``n_samples`` values for the given columns.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        X : np.ndarray of shape (n_original, n_features)
            Full original data (sampler may use it to compute statistics
            or fit density estimators).
        columns : list of int
            Column indices to sample.
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        np.ndarray of shape (n_samples, len(columns))
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Within-distribution samplers (for exogenous / ancestor nodes)
# ---------------------------------------------------------------------------


class WithinDistributionSampler(Sampler):
    """Base class for samplers that generate values within the observed
    distribution.

    Used for exogenous nodes in causal outlier generation — ancestors
    that are sampled from their natural distributions.
    """


class NormalSampler(WithinDistributionSampler):
    """Sample from N(mean, std) for each column independently."""

    def sample(self, rng, X, columns, n_samples):
        means = X[:, columns].mean(axis=0)
        stds = X[:, columns].std(axis=0)
        stds[stds == 0] = 1.0
        return rng.normal(loc=means, scale=stds, size=(n_samples, len(columns)))


class UniformSampler(WithinDistributionSampler):
    """Sample uniformly within [min, max] for each column.

    This sampler stays strictly within the observed range — no expansion.
    For out-of-distribution uniform sampling with expansion, use
    :class:`UniformOutOfDistributionSampler`.
    """

    def sample(self, rng, X, columns, n_samples):
        mins = X[:, columns].min(axis=0)
        maxs = X[:, columns].max(axis=0)
        return rng.uniform(low=mins, high=maxs, size=(n_samples, len(columns)))


class InstanceSampler(WithinDistributionSampler):
    """Sample each cell by copying the value of a random real row.

    Per-cell bootstrap: for every (sample, column) pair an independent
    row is drawn from the full data and that cell's value is copied.
    Every returned value is a real value of its column, but the cells of
    one row come from different rows, so each row is a plausible-looking
    recombination — within-distribution per cell, unlikely as a row.
    """

    def sample(self, rng, X, columns, n_samples):
        row_indices = rng.integers(0, len(X), size=(n_samples, len(columns)))
        return X[row_indices, np.asarray(columns)]


_WITHIN_DISTRIBUTION_REGISTRY = {
    "normal": NormalSampler,
    "uniform": UniformSampler,
}


def create_within_distribution_sampler(name="normal", **kwargs):
    """Create a configured WithinDistributionSampler instance.

    Parameters
    ----------
    name : str, default="normal"
        One of ``"normal"``, ``"uniform"``.
    **kwargs
        Passed to the sampler constructor.

    Returns
    -------
    WithinDistributionSampler

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    if name not in _WITHIN_DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown within-distribution sampler '{name}'. "
            f"Available: {list(_WITHIN_DISTRIBUTION_REGISTRY)}"
        )
    return _WITHIN_DISTRIBUTION_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Out-of-distribution samplers (for perturbation nodes)
# ---------------------------------------------------------------------------


class OutOfDistributionSampler(Sampler):
    """Base class for samplers that generate anomalous values outside the
    observed distribution.

    Used for perturbation nodes in causal outlier generation — nodes
    where the intervention is applied directly.
    """


class ZScoreSampler(OutOfDistributionSampler):
    """Sample using z-score strategy: ±(3 + exponential) * sigma per column.

    Each column independently gets a random sign, then a value of
    ``3 + exponential(scale)`` standard deviations away from the mean.

    Parameters
    ----------
    scale : float, default=1.0
        Scale parameter for the exponential distribution.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def sample(self, rng, X, columns, n_samples):
        n_features = len(columns)
        means = X[:, columns].mean(axis=0)
        stds = X[:, columns].std(axis=0)
        stds[stds == 0] = 1.0
        signs = random_sign(rng, size=(n_samples, n_features))
        exponentials = rng.exponential(
            scale=self.scale, size=(n_samples, n_features)
        )
        standardized = signs * (3.0 + exponentials)
        return means + standardized * stds


class HypersphereSampler(OutOfDistributionSampler):
    """Sample points on a hypersphere with radius >= 3 sigma.

    Points are generated on a hypersphere with radius
    ``3 + exponential(scale)`` in standardized space, then transformed
    back to the original scale.

    Parameters
    ----------
    scale : float, default=1.0
        Scale parameter for the exponential distribution.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def sample(self, rng, X, columns, n_samples):
        n_features = len(columns)
        means = X[:, columns].mean(axis=0)
        stds = X[:, columns].std(axis=0)
        stds[stds == 0] = 1.0
        radii = 3.0 + rng.exponential(scale=self.scale, size=n_samples)
        standardized = random_spherical_coordinates(
            random_generator=rng,
            size=n_features,
            radii=radii,
        )
        return means + standardized * stds


class UniformOutOfDistributionSampler(OutOfDistributionSampler):
    """Sample uniformly around the observed range.

    Samples from [min - expansion*range, max + expansion*range] for each
    column independently.

    Parameters
    ----------
    expansion : float
        Fraction to expand beyond [min, max]. Must be >= 0.
        ``0`` samples the exact data bounding box [min, max]
        (in-distribution by construction; the out-of-distribution-ness is
        the caller's choice).
    """

    def __init__(self, expansion):
        if expansion < 0:
            raise ValueError(
                f"expansion must be >= 0 for uniform sampling, got {expansion}"
            )
        self.expansion = expansion

    def sample(self, rng, X, columns, n_samples):
        mins = X[:, columns].min(axis=0)
        maxs = X[:, columns].max(axis=0)
        ranges = maxs - mins
        low = mins - self.expansion * ranges
        high = maxs + self.expansion * ranges
        return rng.uniform(low=low, high=high, size=(n_samples, len(columns)))


class IndependentHistogramSampler(OutOfDistributionSampler):
    """Sample each requested column from its low-density histogram bins.

    Parameters
    ----------
    bins : int, default=10
        Number of histogram bins per column. Must be positive.
    """

    def __init__(self, bins=10):
        if bins <= 0:
            raise ValueError(f"bins must be > 0, got {bins}")
        self.bins = bins

    def sample(self, rng, X, columns, n_samples):
        result = np.empty((n_samples, len(columns)), dtype=float)
        for output_column, input_column in enumerate(columns):
            counts, edges = np.histogram(X[:, input_column], bins=self.bins)
            probabilities = 1.0 - counts / counts.max()
            probability_sum = probabilities.sum()
            if probability_sum == 0:
                probabilities = np.full(self.bins, 1.0 / self.bins)
            else:
                probabilities /= probability_sum
            bin_indices = rng.choice(
                self.bins, size=n_samples, p=probabilities
            )
            result[:, output_column] = rng.uniform(
                low=edges[bin_indices], high=edges[bin_indices + 1]
            )
        return result


class JointHistogramSampler(OutOfDistributionSampler):
    """Sample requested columns jointly from low-density histogram bins.

    Parameters
    ----------
    bins : int, default=10
        Number of bins along each standardized dimension. Must be positive.
    threshold_low_density : float, default=0.1
        Maximum normalized bin count considered low density. Must be
        strictly between zero and one.
    """

    def __init__(self, bins=10, threshold_low_density=0.1):
        if bins <= 0:
            raise ValueError(f"bins must be > 0, got {bins}")
        if not 0 < threshold_low_density < 1:
            raise ValueError(
                "threshold_low_density must be between 0 and 1, "
                f"got {threshold_low_density}"
            )
        self.bins = bins
        self.threshold_low_density = threshold_low_density

    def sample(self, rng, X, columns, n_samples):
        if len(columns) > 5:
            raise NotImplementedError(
                "Joint histogram sampling supports at most 5 columns"
            )

        target = X[:, columns]
        scaler = StandardScaler()
        standardized = scaler.fit_transform(target)
        counts, edges = np.histogramdd(
            standardized, bins=self.bins
        )
        normalized_counts = counts / counts.max()
        low_density_bins = np.argwhere(
            normalized_counts <= self.threshold_low_density
        )
        if len(low_density_bins) == 0:
            raise ValueError(
                "threshold_low_density found no low-density region"
            )

        selected_bins = low_density_bins[
            rng.integers(0, len(low_density_bins), size=n_samples)
        ]
        samples = np.column_stack([
            rng.uniform(
                low=edges[column][selected_bins[:, column]],
                high=edges[column][selected_bins[:, column] + 1],
            )
            for column in range(len(columns))
        ])
        return scaler.inverse_transform(samples)


# Cap on the number of grid cells scored by the KDE grid samplers.
_MAX_GRID_CELLS = 65536


class KernelDensitySampler(OutOfDistributionSampler):
    """Sample from low-density regions of a kernel density estimate.

    The requested columns are standardized, a ``KernelDensity`` is fit on
    them (ball-tree accelerated), and the standardized bounding box padded
    by one standard deviation per side is discretized into a grid of at
    most ``grid_budget`` cells. Every cell is scored once, and the
    requested samples are drawn from a categorical over the cells at or
    below the ``threshold_low_density`` percentile of the data's
    log-densities — each point uniform inside its cell. There is no
    rejection step, so exactly ``n_samples`` points are always returned.

    The grid keeps at most ``grid_budget`` cells per dimension root
    (``g = ceil(grid_budget ** (1 / d))``) and at most 65536 cells in
    total, so the sampler is intended for low dimensionality (d <= ~16
    with the default budget).

    Parameters
    ----------
    threshold_low_density : float, default=0.1
        Percentile (in [0, 1)) of the data's log-densities that defines
        the low-density boundary.
    grid_budget : int, default=2048
        Approximate number of grid cells (``g ** d``). Must be >= 1.
    """

    def __init__(self, threshold_low_density=0.1, grid_budget=2048):
        if not 0 <= threshold_low_density < 1:
            raise ValueError(
                "threshold_low_density must be in [0, 1), "
                f"got {threshold_low_density}")
        if grid_budget < 1:
            raise ValueError(f"grid_budget must be >= 1, got {grid_budget}")
        self.threshold_low_density = threshold_low_density
        self.grid_budget = grid_budget

    def sample(self, rng, X, columns, n_samples):
        target = X[:, columns]
        d = target.shape[1]
        if n_samples == 0:
            return np.empty((0, d))
        scaler = StandardScaler()
        standardized = scaler.fit_transform(target)

        estimator = KernelDensity(
            bandwidth="scott", algorithm="ball_tree").fit(standardized)
        low_density_threshold = np.percentile(
            estimator.score_samples(standardized), self.threshold_low_density)

        g = min(int(np.ceil(self.grid_budget ** (1.0 / d))), 32)
        while g > 1 and g ** d > _MAX_GRID_CELLS:
            g -= 1
        if g < 2:
            raise NotImplementedError(
                f"a grid with >= 2 cells per dimension is infeasible for "
                f"{d} dimensions with grid_budget={self.grid_budget}; "
                f"use fewer columns or a larger budget")
        edges = [
            np.linspace(standardized[:, j].min() - 1.0,
                        standardized[:, j].max() + 1.0, g + 1)
            for j in range(d)
        ]
        total = g ** d
        idx = np.unravel_index(np.arange(total), (g,) * d)
        cell_lo = np.column_stack([edges[j][idx[j]] for j in range(d)])
        cell_hi = np.column_stack([edges[j][idx[j] + 1] for j in range(d)])
        centers = (cell_lo + cell_hi) / 2

        f_grid = estimator.score_samples(centers)
        weights = (f_grid <= low_density_threshold).astype(float)
        if weights.sum() == 0:
            # Pathological (tiny grid / very peaked data): fall back to
            # the least dense cells.
            weights[np.argsort(f_grid)[: min(len(f_grid), self.grid_budget)]] = 1.0
        cells = rng.choice(total, size=n_samples, p=weights / weights.sum())
        samples = np.column_stack([
            rng.uniform(cell_lo[:, j], cell_hi[:, j])[cells] for j in range(d)
        ])
        return scaler.inverse_transform(samples)


def _kd_partition(points, lo, hi, leaf_size):
    """Yield ``(count, lo, hi)`` leaf cells of a recursive median split.

    Splits the hyperrectangle on its longest axis at the data median until
    every cell holds at most ``leaf_size`` points (or is degenerate).
    Median splits balance the counts, so the depth is O(log n).
    """
    n = points.shape[0]
    if n <= leaf_size:
        yield n, lo, hi
        return
    axis = int(np.argmax(hi - lo))
    if hi[axis] <= lo[axis]:
        yield n, lo, hi
        return
    order = np.argsort(points[:, axis], kind="stable")
    values = points[order, axis]
    med = values[n // 2]
    pos = int(np.searchsorted(values, med, side="right"))
    if pos <= 0 or pos >= n:
        yield n, lo, hi
        return
    left_hi = hi.copy()
    right_lo = lo.copy()
    left_hi[axis] = med
    right_lo[axis] = med
    yield from _kd_partition(points[order[:pos]], lo, left_hi, leaf_size)
    yield from _kd_partition(points[order[pos:]], right_lo, hi, leaf_size)


class TreeCellSampler(OutOfDistributionSampler):
    """Sample from low-density cells of an adaptive kd partition.

    The requested columns are standardized and partitioned by recursive
    median splits into cells holding at most ``leaf_size`` points each:
    the resolution adapts (fine in dense regions, coarse in sparse ones),
    so the cell density ``count / volume`` is a data-driven histogram
    with the bin size chosen by the data. Every cell is then extended
    axis-aligned to the midplanes with the neighbouring points (an
    axis-aligned Voronoi approximation), so the empty space between the
    cells is covered. A cell is low density when its density is at most
    ``threshold_low_density`` times the global mean density; the
    requested samples are drawn from a categorical over the qualifying
    cells (no rejection), each point uniform inside its extended cell,
    so exactly ``n_samples`` points are always returned.

    Parameters
    ----------
    threshold_low_density : float, default=0.1
        Maximum cell density, as a share of the global mean density,
        for the cell to count as low density (in (0, 1]).
    leaf_size : int, default=8
        Maximum number of points per leaf cell (must be >= 1).
    """

    def __init__(self, threshold_low_density=0.1, leaf_size=8):
        if not 0 < threshold_low_density <= 1:
            raise ValueError(
                "threshold_low_density must be in (0, 1], "
                f"got {threshold_low_density}")
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size}")
        self.threshold_low_density = threshold_low_density
        self.leaf_size = leaf_size

    def sample(self, rng, X, columns, n_samples):
        target = X[:, columns]
        d = target.shape[1]
        if n_samples == 0:
            return np.empty((0, d))
        scaler = StandardScaler()
        standardized = scaler.fit_transform(target)
        n = standardized.shape[0]

        lo0 = standardized.min(axis=0)
        hi0 = standardized.max(axis=0)
        leaves = list(_kd_partition(standardized, lo0, hi0, self.leaf_size))
        counts = np.array([c for c, _, _ in leaves], dtype=float)
        cell_lo = np.array([lo for _, lo, _ in leaves])
        cell_hi = np.array([hi for _, _, hi in leaves])

        # Density proxy: points per cell volume, relative to the global
        # mean density (degenerate zero-volume cells are infinitely dense)
        volumes = np.prod(cell_hi - cell_lo, axis=1)
        mean_volume = float(np.prod(hi0 - lo0))
        mean_density = n / mean_volume if mean_volume > 0 else 0.0
        density = np.full(len(volumes), np.inf)
        np.divide(counts, volumes, out=density, where=volumes > 0)

        # Extend each cell axis-aligned to the midplane with the
        # neighbouring points (empty space is shared with neighbours)
        for j in range(d):
            values = np.sort(standardized[:, j])
            k = np.searchsorted(values, cell_hi[:, j])
            gap_plus = np.where(
                k < n, values[np.minimum(k, n - 1)] - cell_hi[:, j],
                (hi0[j] + 1.0) - cell_hi[:, j])
            k2 = np.searchsorted(values, cell_lo[:, j], side="left")
            gap_minus = np.where(
                k2 > 0, cell_lo[:, j] - values[k2 - 1],
                cell_lo[:, j] - (lo0[j] - 1.0))
            cell_hi[:, j] = np.minimum(cell_hi[:, j] + 0.5 * gap_plus, hi0[j] + 1.0)
            cell_lo[:, j] = np.maximum(cell_lo[:, j] - 0.5 * gap_minus, lo0[j] - 1.0)

        weights = (
            density <= self.threshold_low_density * mean_density).astype(float)
        if weights.sum() == 0:
            # Pathological (e.g. very small data): fall back to the least
            # dense cells.
            weights[np.argsort(density)[: min(len(density), 10)]] = 1.0
        cells = rng.choice(len(weights), size=n_samples, p=weights / weights.sum())
        samples = np.column_stack([
            rng.uniform(cell_lo[:, j], cell_hi[:, j])[cells] for j in range(d)
        ])
        return scaler.inverse_transform(samples)


_OUT_OF_DISTRIBUTION_REGISTRY = {
    "zscore": ZScoreSampler,
    "hypersphere": HypersphereSampler,
    "uniform": UniformOutOfDistributionSampler,
    "independent_histogram": IndependentHistogramSampler,
    "histogram": JointHistogramSampler,
    "low_density": KernelDensitySampler,
    "tree": TreeCellSampler,
}


def create_out_of_distribution_sampler(name="zscore", **kwargs):
    """Create a configured OutOfDistributionSampler instance.

    Parameters
    ----------
    name : str, default="zscore"
        One of ``"zscore"``, ``"hypersphere"``, ``"uniform"``,
        ``"independent_histogram"``, ``"histogram"``, ``"low_density"``,
        ``"tree"``.
    **kwargs
        Passed to the sampler constructor (e.g., ``scale=2.0`` for
        ``"zscore"``, ``expansion=0.2`` for ``"uniform"``, or ``bins=8``
        for either histogram sampler).

    Returns
    -------
    OutOfDistributionSampler

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    if name not in _OUT_OF_DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown out-of-distribution sampler '{name}'. "
            f"Available: {list(_OUT_OF_DISTRIBUTION_REGISTRY)}"
        )
    return _OUT_OF_DISTRIBUTION_REGISTRY[name](**kwargs)
