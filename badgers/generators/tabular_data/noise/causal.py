"""Causal noise generator.

Injects Gaussian measurement noise at perturbation nodes and propagates it to
descendants via the structural equations (each descendant adds its own
per-row residual).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.generators.tabular_data.causal._base import CausalGenerator


class CausalNoiseGenerator(CausalGenerator):
    """Generates causally-propagated Gaussian noise.

    Parameters
    ----------
    random_generator : numpy.random.Generator, default=rng(seed=0)
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (attached via fit() or use_dgp()).

    generate() parameters (passed via **params)
    ----------
    perturbation_nodes, mechanism, mechanism_cause, mechanism_rate :
        see CausalGenerator.
    n_copies : int, default=1
        Number of noisy copies per seed row (``n -> n * n_copies``).
        ``1`` degenerates to in-place.
    noise_std : float or np.ndarray, default=0.1
        Std of Gaussian noise, relative to column std. Scalar or per-column
        (aligned to perturbation_nodes order).
    """

    deficit_name = "noise"

    @preprocess_inputs
    def generate(
        self,
        X,
        y,
        perturbation_nodes=None,
        n_copies=1,
        mechanism="always",
        mechanism_cause=None,
        mechanism_rate=0.5,
        noise_std=0.1,
    ):
        """Generate causally-propagated noise (expands each seed row)."""
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}")
        self._require_fitted()
        self._require_dgp(X)
        order, node_to_col, descendants, exogenous, perturb_cols = self._validate(
            X, perturbation_nodes
        )
        n_features = X.shape[1]
        col_stds = np.std(X, axis=0)
        col_stds[col_stds == 0] = 1.0

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
        rows_original = np.repeat(X, n_copies, axis=0)
        # Inject noise at the perturbation nodes (gated by the mechanism:
        # fire masks for ALL expanded rows at once; quantile thresholds are
        # computed on the UN-EXPANDED input so expanded rows are gated
        # against the original columns' statistics).
        fires = self._fire_masks(
            rows, perturbation_nodes, node_to_col, mechanism, mechanism_cause,
            mechanism_rate, stats_rows=X,
        )
        affected = np.zeros(rows.shape[0], dtype=bool)
        for j, node in enumerate(perturbation_nodes):
            mask = fires[node]
            col = node_to_col[node]
            if mask.any():
                rows[mask, col] += self.random_generator.normal(
                    0, noise_scale[j] * col_stds[col], size=int(mask.sum())
                )
            affected |= mask

        # Propagate the delta to the descendants via the structural
        # equations: descendant_new = descendant_orig + beta @ delta_parents.
        # Each expanded row's "original" is the seed row it was copied from
        # (so the descendants keep the seed's exogenous noise); only the
        # injected noise delta propagates.
        self.propagate(rows, perturbation_nodes, self.random_generator,
                       rows_original=rows_original)

        if y is not None:
            y = np.repeat(y, n_copies)
        yt = self._labels_expanded(y, affected, self.deficit_name)
        return rows, yt
