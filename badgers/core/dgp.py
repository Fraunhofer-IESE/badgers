"""Data generation process (DGP) protocol and PyMC-based implementation.

The DGP models the structural equations (DGP layer) and the measurement
process (observation layer). ``PyMCDGP`` fits a Bayesian linear model with
PyMC and exposes a fast forward pass (``forward_pass``) and sampling (``sample``)
for the causal defect generators.
"""

from typing import Protocol

import networkx as nx
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from badgers.core.causal_graph import get_parents, topological_order


class DGP(Protocol):
    """Protocol for a data generation process."""

    def fit(self, X: np.ndarray) -> None:
        """Fit the DGP to the observed data."""
        ...

    def forward_pass(
        self,
        rows: np.ndarray,
        nodes: list,
        rng: np.random.Generator,
        add_residual: bool = True,
    ) -> np.ndarray:
        """Compute values for ``nodes`` from their parents (structural eq).

        This is the generic, defect-agnostic primitive: the caller decides
        *what* to compute (``nodes``), the DGP only knows *how* (the
        structural equation).

        Args:
            rows: Array of shape (n_rows, n_features). The parent columns of
                each node in ``nodes`` must already be filled. Filled in
                place for the columns of ``nodes``.
            nodes: Explicit, topologically-ordered list of non-root nodes to
                compute. Root nodes are skipped.
            rng: NumPy random generator (used to draw per-row residuals).
            add_residual: If True, add a per-row residual (noise-like). If
                False, the forward pass is deterministic (drift-like).

        Returns:
            The same ``rows`` array with the ``nodes`` columns filled.
        """
        ...

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Sample ``n_samples`` rows from the DGP's joint distribution."""
        ...


class PyMCDGP(DGP):
    """PyMC-based DGP.

    Two-layer model, one block per node in topological order:

    - DGP layer (structural equation):
      - root: ``X_i = mu_i + sigma_i * eps_i``
      - non-root: ``X_i = beta_i @ X_parents + sigma_i * eps_i``
    - Observation layer (measurement process):
      - ``Observed_i = X_i + sigma_obs_i * eps_obs_i``

    ``fit`` builds the ``pm.Model`` and runs MCMC. ``forward_pass`` uses the
    posterior mean of ``beta`` for a fast forward pass. ``sample`` uses the
    posterior-mean parameters to generate new rows.
    """

    def __init__(self, graph: nx.DiGraph, column_mapping: dict,
                 n_samples_mcmc: int = 1000):
        self.graph = graph
        self.column_mapping = column_mapping
        self.n_samples_mcmc = n_samples_mcmc
        self._fitted = False
        self._beta_mean: dict = {}
        self._mu_mean: dict = {}
        self._sigma_mean: dict = {}
        self._residual_std: dict = {}
        self._model = None
        self._trace = None

    def fit(self, X: np.ndarray) -> None:
        """Build the pm.Model and run MCMC (Bayesian inference)."""
        n = X.shape[0]
        order = topological_order(self.graph)
        with pm.Model() as self._model:
            X_latent = {}
            for node in order:
                col = self.column_mapping[node]
                parents = get_parents(self.graph, node)
                if not parents:
                    mu_i = pm.Normal(f"mu_{node}", 0, 10)
                    sigma_i = pm.HalfNormal(f"sigma_{node}", 1)
                    X_latent[node] = pm.Normal(
                        f"X_{node}", mu=mu_i, sigma=sigma_i, shape=n
                    )
                else:
                    parent_cols = [self.column_mapping[p] for p in parents]
                    beta_i = pm.Normal(f"beta_{node}", 0, 10, shape=len(parents))
                    sigma_i = pm.HalfNormal(f"sigma_{node}", 1)
                    X_latent[node] = pm.Normal(
                        f"X_{node}",
                        mu=pt.dot(X[:, parent_cols], beta_i),
                        sigma=sigma_i,
                        shape=n,
                    )
                sigma_obs_i = pm.HalfNormal(f"sigma_obs_{node}", 1)
                pm.Normal(
                    f"obs_{node}",
                    mu=X_latent[node],
                    sigma=sigma_obs_i,
                    observed=X[:, col],
                )
            self._trace = pm.sample(
                self.n_samples_mcmc,
                tune=self.n_samples_mcmc // 2,
                chains=1,
                cores=1,
                return_inferencedata=True,
                progressbar=False,
            )
        # Extract posterior means
        for node in order:
            parents = get_parents(self.graph, node)
            if not parents:
                self._mu_mean[node] = float(
                    np.asarray(
                        self._trace.posterior[f"mu_{node}"]
                        .mean(dim=("chain", "draw")).values
                    )
                )
                self._sigma_mean[node] = float(
                    np.asarray(
                        self._trace.posterior[f"sigma_{node}"]
                        .mean(dim=("chain", "draw")).values
                    )
                )
            else:
                self._beta_mean[node] = np.asarray(
                    self._trace.posterior[f"beta_{node}"]
                    .mean(dim=("chain", "draw")).values
                )
                self._sigma_mean[node] = float(
                    np.asarray(
                        self._trace.posterior[f"sigma_{node}"]
                        .mean(dim=("chain", "draw")).values
                    )
                )
                col = self.column_mapping[node]
                parent_cols = [self.column_mapping[p] for p in parents]
                predicted = X[:, parent_cols] @ self._beta_mean[node]
                self._residual_std[node] = float(np.std(X[:, col] - predicted))
        self._fitted = True

    def forward_pass(
        self,
        rows: np.ndarray,
        nodes: list,
        rng: np.random.Generator,
        add_residual: bool = True,
    ) -> np.ndarray:
        """Forward pass using the posterior mean of beta.

        Computes each node in ``nodes`` (topologically ordered) from its
        parents. Root nodes are skipped. Defect-agnostic: the caller decides
        which nodes to compute.
        """
        if not self._fitted:
            raise RuntimeError("PyMCDGP.forward_pass called before fit")
        n_rows = rows.shape[0]
        for node in nodes:
            col = self.column_mapping[node]
            parents = get_parents(self.graph, node)
            if not parents:
                continue
            parent_cols = [self.column_mapping[p] for p in parents]
            parent_vals = rows[:, parent_cols]
            beta = self._beta_mean[node]
            predicted = parent_vals @ beta
            if add_residual:
                resid_std = self._residual_std.get(node, 0.0)
                rows[:, col] = predicted + rng.normal(0, resid_std, size=n_rows)
            else:
                rows[:, col] = predicted
        return rows

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Sample new rows using the posterior-mean parameters."""
        if not self._fitted:
            raise RuntimeError("PyMCDGP.sample called before fit")
        order = topological_order(self.graph)
        n_features = len(self.graph.nodes)
        X_sampled = np.zeros((n_samples, n_features))
        for node in order:
            col = self.column_mapping[node]
            parents = get_parents(self.graph, node)
            if not parents:
                X_sampled[:, col] = (
                    self._mu_mean[node]
                    + self._sigma_mean[node] * rng.normal(0, 1, size=n_samples)
                )
            else:
                parent_cols = [self.column_mapping[p] for p in parents]
                predicted = X_sampled[:, parent_cols] @ self._beta_mean[node]
                X_sampled[:, col] = (
                    predicted
                    + self._sigma_mean[node] * rng.normal(0, 1, size=n_samples)
                )
        return X_sampled
