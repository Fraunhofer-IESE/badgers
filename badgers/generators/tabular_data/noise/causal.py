"""Causal noise generator.

Injects Gaussian measurement noise at perturbation nodes and recomputes the
descendants of the affected rows through the structural equations (Pearl
surgery: each descendant is re-evaluated from its perturbed parents with
fresh noise).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import (
    CausalGenerator,
    _column_stds,
)


class CausalNoiseGenerator(CausalGenerator):
    """Generates causally-propagated Gaussian noise.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes : list of str
        Nodes where the noise is injected (required).
    n_copies : int, default=1
        Number of noisy copies per seed row (``n -> n * n_copies``).
        ``1`` degenerates to in-place.
    selector : Selector or None, default=None
        Row-selection rule from ``badgers.core.selectors``, evaluated on
        the ORIGINAL (un-expanded) rows. ``None`` = every row fires.
    noise_std : float or np.ndarray, default=0.1
        Std of Gaussian noise, relative to column std. Scalar or per-column
        (aligned to perturbation_nodes order).
    """

    deficit_name = "noise"

    def get_noise_labels(self) -> np.ndarray:
        """Return the noise labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the noise deficit, False for
        original rows.
        """
        if self.deficit_labels_ is None:
            raise RuntimeError(
                "Noise labels are only available after calling generate()."
            )
        return self.deficit_labels_

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        n_copies=1,
        selector=None,
        noise_std=0.1,
    ):
        """Generate causally-propagated noise (expands each seed row)."""
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}")
        self._require_fitted()
        self._require_dgp(X)
        self._validate_structure(X)
        self._validate_perturbation_nodes(perturbation_nodes)
        node_to_col = self.column_mapping
        n_features = X.shape[1]
        col_stds = _column_stds(X)

        # noise_std scalar or per-perturbation-node array
        if np.isscalar(noise_std):
            noise_scale = np.full(len(perturbation_nodes), float(noise_std))
        else:
            noise_scale = np.asarray(noise_std, dtype=float)
            if noise_scale.shape != (len(perturbation_nodes),):
                raise ValueError(
                    "noise_std array must have one entry per perturbation node"
                )

        # Expand: each seed row -> n_copies copies (n_copies=1 keeps the shape).
        rows = np.repeat(X, n_copies, axis=0)
        # Selector gating: masks evaluated on the ORIGINAL (un-expanded)
        # rows, then replicated to the expansion (each seed row's copies
        # either all fire or all stay clean). Quantile thresholds are
        # computed on the un-expanded input by construction.
        fires = self._fire_masks(
            X, perturbation_nodes, selector=selector,
        )
        fires = {
            node: np.repeat(mask, n_copies) for node, mask in fires.items()
        }
        affected = np.zeros(rows.shape[0], dtype=bool)
        for j, node in enumerate(perturbation_nodes):
            mask = fires[node]
            col = node_to_col[node]
            if mask.any():
                rows[mask, col] += self.random_generator.normal(
                    0, noise_scale[j] * col_stds[col], size=int(mask.sum())
                )
            affected |= mask

        # Recompute the descendants of the AFFECTED rows through the
        # structural equations (Pearl surgery): descendant = f(new parents)
        # + fresh noise. Unaffected rows are left untouched.
        affected_rows = np.flatnonzero(affected)
        if affected_rows.size:
            sub = rows[affected_rows]
            self.dgp.intervene(
                sub, perturbation_nodes, rng=self.random_generator
            )
            rows[affected_rows] = sub

        self.deficit_labels_ = affected.copy()
        return rows, None


class CausalAdditiveNoiseGenerator(CausalGenerator):
    """Generates causally-propagated additive noise with an arbitrary distribution.

    Like :class:`CausalNoiseGenerator`, but the injected noise is drawn from
    a user-chosen distribution (a *frozen* scipy distribution object, e.g.
    ``scipy.stats.laplace(scale=1)``, ``scipy.stats.t(df=3)``) instead of
    Gaussian white noise. Every row fires (no mechanism gating, no row
    expansion — strictly in-place); the affected rows' descendants are
    recomputed through the structural equations (Pearl surgery).

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    distribution : frozen scipy distribution
        Frozen scipy distribution object exposing ``.rvs(size=...,
        random_state=...)`` (e.g. ``scipy.stats.laplace(scale=1)``). Drawn
        samples are added as-is.
    noise_std : float, default=1.0
        Global multiplier applied to the drawn noise (1.0 = raw draw).
    perturbation_nodes : list of str
        Nodes where the noise is injected (required).
    selector : Selector or None, default=None
        Row-selection rule from ``badgers.core.selectors``. ``None`` =
        every row fires.
    """

    deficit_name = "noise"

    def get_noise_labels(self) -> np.ndarray:
        """Return the noise labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the noise deficit, False for
        original rows.
        """
        if self.deficit_labels_ is None:
            raise RuntimeError(
                "Noise labels are only available after calling generate()."
            )
        return self.deficit_labels_

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes,
        distribution,
        noise_std=1.0,
        selector=None,
    ):
        """Generate causally-propagated additive noise (in-place)."""
        rvs = getattr(distribution, "rvs", None)
        if not callable(rvs):
            raise TypeError(
                "distribution must be a frozen scipy distribution object (e.g. "
                "scipy.stats.laplace(scale=1)) exposing .rvs(size=..., random_state=...); "
                f"got {distribution!r}"
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self._require_fitted()
        self._require_dgp(X)
        self._validate_structure(X)
        self._validate_perturbation_nodes(perturbation_nodes)
        node_to_col = self.column_mapping

        fires = self._fire_masks(
            X, perturbation_nodes, selector=selector,
        )
        affected = np.zeros(X.shape[0], dtype=bool)
        for mask in fires.values():
            affected |= mask

        # Inject noise ONLY at the fired rows of each perturbation node.
        noise = np.zeros(X.shape)
        for j, node in enumerate(perturbation_nodes):
            col = node_to_col[node]
            mask = fires[node]
            if mask.any():
                noise[mask, col] = (
                    rvs(size=int(mask.sum()), random_state=self.random_generator)
                    * noise_std
                )
        rows = X + noise

        # Recompute the AFFECTED rows through the structural equations
        # (Pearl surgery): descendant = f(new parents) + fresh noise.
        affected_rows = np.flatnonzero(affected)
        if affected_rows.size:
            sub = rows[affected_rows]
            self.dgp.intervene(
                sub, perturbation_nodes, rng=self.random_generator
            )
            rows[affected_rows] = sub

        self.deficit_labels_ = affected.copy()
        return rows, None
