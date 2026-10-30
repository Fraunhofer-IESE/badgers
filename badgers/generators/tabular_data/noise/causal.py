"""Causal noise generator.

Injects Gaussian measurement noise at perturbation nodes and recomputes the
descendants of the affected rows through the structural equations (Pearl
surgery: each descendant is re-evaluated from its perturbed parents with
fresh noise).
"""

import numpy as np

from badgers.core.decorators.tabular_data import preprocess_inputs
from badgers.core.selectors import get_selected_fields
from badgers.core.selectors.base import _resolve_targets
from badgers.generators.tabular_data.causal._base import (
    CausalGenerator,
    _column_stds,
    _param_by_target,
)


class CausalNoiseGenerator(CausalGenerator):
    """Generates causally-propagated Gaussian noise.

    A selector is required and owns the intervention targets: the
    selector's ``columns`` (column positions) are the nodes where the
    noise is injected, and its selection rule decides which rows are
    perturbed. The selector sees the ORIGINAL (un-expanded) rows:
    selection masks are evaluated before the ``n_copies`` expansion, then
    replicated so every copy of a seed row shares the same state.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    selector : Selector
        Row/column selection rule from ``badgers.core.selectors``. Its
        ``columns`` own the perturbation targets; required (``None``
        raises -- there is no relevant default: the selector owns the
        perturbation targets). Evaluated on the ORIGINAL (un-expanded)
        rows.
    n_copies : int, default=1
        Number of noisy copies per seed row (``n -> n * n_copies``).
        ``1`` degenerates to in-place.
    noise_std : float or dict, default=0.1
        Std of Gaussian noise, relative to column std. Scalar (same for
        all targets) or dict keyed by column position
        (``{col_position: std}``) — must cover every target.
    """

    deficit_name = "noise"

    def get_noise_labels(self) -> np.ndarray:
        """Return the noise labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the noise deficit, False for
        original rows.
        """
        return self._get_deficit_labels()

    @preprocess_inputs
    def generate(
        self,
        X,
        selector,
        n_copies=1,
        noise_std=0.1,
    ):
        """Generate causally-propagated noise (expands each seed row)."""
        if n_copies < 1:
            raise ValueError(f"n_copies must be >= 1, got {n_copies}")
        self._require_fitted()
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets"
            )
        self._require_dgp(X)
        self._validate_structure(X)
        targets = list(_resolve_targets(selector, X, None))
        scale = _param_by_target(noise_std, "noise_std", targets)
        col_stds = _column_stds(X)
        rng = self.random_generator

        # Selector realization: masks evaluated on the ORIGINAL (un-expanded)
        # rows, then replicated to the expansion (each seed row's copies
        # either all are selected or all stay clean). Quantile thresholds are
        # computed on the un-expanded input by construction.
        field = get_selected_fields(X, selector, rng)
        field_expanded = np.repeat(field, n_copies, axis=0)

        # Expand: each seed row -> n_copies copies (n_copies=1 keeps the shape).
        rows = np.repeat(X, n_copies, axis=0)

        # Inject noise ONLY at the selected rows of each target.
        affected = np.zeros(rows.shape[0], dtype=bool)
        for i, col in enumerate(targets):
            mask = field_expanded[:, i]
            if mask.any():
                rows[mask, col] += rng.normal(
                    scale=scale[col] * col_stds[col],
                    size=int(mask.sum()),
                )
            affected |= mask

        # Recompute the descendants of the AFFECTED rows through the
        # structural equations (Pearl surgery): descendant = f(new parents)
        # + fresh noise. Intervene receives ALL target names: unselected
        # target nodes stay pinned at their (possibly noisy) current
        # values while their strict descendants recompute from the pinned
        # parents. Unaffected rows are left untouched.
        affected_rows = np.flatnonzero(affected)
        if affected_rows.size:
            sub = rows[affected_rows]
            self.dgp.intervene(sub, self._node_names(targets), rng=rng)
            rows[affected_rows] = sub

        self.deficit_labels_ = affected.copy()
        return rows


class CausalAdditiveNoiseGenerator(CausalGenerator):
    """Generates causally-propagated additive noise with an arbitrary distribution.

    Like :class:`CausalNoiseGenerator`, but the injected noise is drawn from
    a user-chosen distribution (a *frozen* scipy distribution object, e.g.
    ``scipy.stats.laplace(scale=1)``, ``scipy.stats.t(df=3)``) instead of
    Gaussian white noise. The affected rows are noised in place
    (``rows = X + noise``) and their descendants are recomputed through
    the structural equations (Pearl surgery).

    A selector is required and owns the intervention targets: the
    selector's ``columns`` (column positions) are the nodes where the
    noise is injected, and its selection rule decides which rows are
    perturbed.

    Parameters
    ----------
    random_generator : numpy.random.Generator, required
        Random number generator instance.
    dgp : DGP or None, default=None
        DGP instance (passed via the constructor or use_dgp(); fitted
        explicitly by the caller, or lazily inside generate() if
        unfitted).

    generate() parameters (passed via **params)
    ----------
    selector : Selector
        Row/column selection rule from ``badgers.core.selectors``. Its
        ``columns`` own the perturbation targets; required (``None``
        raises -- there is no relevant default: the selector owns the
        perturbation targets).
    distribution : frozen scipy distribution
        Frozen scipy distribution object exposing ``.rvs(size=...,
        random_state=...)`` (e.g. ``scipy.stats.laplace(scale=1)``). Drawn
        samples are added as-is.
    noise_std : float, default=1.0
        Global multiplier applied to the drawn noise (1.0 = raw draw).
    """

    deficit_name = "noise"

    def get_noise_labels(self) -> np.ndarray:
        """Return the noise labels of the last generated output.

        A boolean vector aligned with the rows of the last generated
        output: True for rows that carry the noise deficit, False for
        original rows.
        """
        return self._get_deficit_labels()

    @preprocess_inputs
    def generate(
        self,
        X,
        selector,
        distribution=None,
        noise_std=1.0,
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
        if selector is None:
            raise ValueError(
                "selector is required: it owns the intervention targets"
            )
        self._require_dgp(X)
        self._validate_structure(X)
        targets = list(_resolve_targets(selector, X, None))
        col_stds = _column_stds(X)
        rng = self.random_generator

        field = get_selected_fields(X, selector, rng)
        affected = field.any(axis=1)

        # Inject noise ONLY at the selected rows of each target.
        noise = np.zeros(X.shape)
        for i, col in enumerate(targets):
            mask = field[:, i]
            if mask.any():
                noise[mask, col] = (
                    rvs(size=int(mask.sum()), random_state=rng)
                    * noise_std
                    * col_stds[col]
                )
        rows = X + noise

        # Recompute the AFFECTED rows through the structural equations
        # (Pearl surgery): descendant = f(new parents) + fresh noise.
        # Intervene receives ALL target names: every selected row is
        # affected (additive model), so every target node is noisy.
        affected_rows = np.flatnonzero(affected)
        if affected_rows.size:
            sub = rows[affected_rows]
            self.dgp.intervene(sub, self._node_names(targets), rng=rng)
            rows[affected_rows] = sub

        self.deficit_labels_ = affected.copy()
        return rows
