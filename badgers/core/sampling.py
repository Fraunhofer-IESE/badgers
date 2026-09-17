"""Sampling strategies for generating values within or outside observed distributions.

WithinDistributionSampler subclasses generate plausible values from the
observed data distribution (used for exogenous nodes in causal generation).

OutOfDistributionSampler subclasses generate anomalous values outside the
observed distribution (used for perturbation nodes in causal generation).
"""

import numpy as np

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
    """Sample uniformly beyond the observed range.

    Samples from [min - expansion*range, max + expansion*range] for each
    column independently.

    Parameters
    ----------
    expansion : float
        Fraction to expand beyond [min, max]. Must be > 0.
    """

    def __init__(self, expansion):
        if expansion <= 0:
            raise ValueError(
                f"expansion must be > 0 for out-of-distribution sampling, "
                f"got {expansion}"
            )
        self.expansion = expansion

    def sample(self, rng, X, columns, n_samples):
        mins = X[:, columns].min(axis=0)
        maxs = X[:, columns].max(axis=0)
        ranges = maxs - mins
        low = mins - self.expansion * ranges
        high = maxs + self.expansion * ranges
        return rng.uniform(low=low, high=high, size=(n_samples, len(columns)))


_OUT_OF_DISTRIBUTION_REGISTRY = {
    "zscore": ZScoreSampler,
    "hypersphere": HypersphereSampler,
    "uniform": UniformOutOfDistributionSampler,
}


def create_out_of_distribution_sampler(name="zscore", **kwargs):
    """Create a configured OutOfDistributionSampler instance.

    Parameters
    ----------
    name : str, default="zscore"
        One of ``"zscore"``, ``"hypersphere"``, ``"uniform"``.
    **kwargs
        Passed to the sampler constructor (e.g., ``scale=2.0`` for
        ``"zscore"``, ``expansion=0.2`` for ``"uniform"``).

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
