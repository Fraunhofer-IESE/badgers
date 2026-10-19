"""PyMC backend: per-node Bayesian linear mechanisms with POSTERIOR draws.

Route B preparation per the structural-modifications resolution (D1–D6):
the backend implements ONLY the per-node protocol core
(``fit_node``/``response``/``draw_noise``) plus ONE hook,
``pick_parameters``; ``fit``/``sample``/``intervene`` are the generic
``DGPMixin`` reference implementations.

Design (PyMC 6.3.2 facts): the old monolithic ``PyMCDGP`` built ONE
whole-graph ``pm.Model`` and sampled it once; the per-node protocol needs
fit_node as the unit, so each node gets its own tiny Bayesian linear
regression model (``beta ~ Normal``, ``sigma ~ HalfNormal``, observed
likelihood) and ``DGPMixin.fit`` walks the graph calling it per node.

Parameter uncertainty IS the backend's selling point (feature #15:
uncertainty-aware deficits). The FULL posterior is kept — NO posterior
mean is ever returned. Separation of concerns (protocol constraint):
``response`` takes NO rng, so stochastic parameter selection cannot live
inside the cascade. Instead the backend overrides the mixin hook
:meth:`~badgers.core.dgp.base.DGPMixin.pick_parameters`, which the
generic ``sample``/``intervene`` call once at the top of every cascade::

    carrier = PyMCDGP(g, m).fit(X)       # keeps ALL posterior draws
    rows    = carrier.sample(rng, n)     # hook pins ONE draw internally
    rows    = carrier.intervene(rows, nodes, values, rng=rng)  # same

The hook returns a ``_ParameterDrawView``: ``response`` evaluates the
pinned draw's ``beta``/``mu`` (deterministic — view-composable), and
``draw_noise`` draws fresh shocks at the pinned draw's ``sigma``. The
carrier itself evaluates draw index 0 (a canonical parameter set).
Exploring the posterior = repeated cascades with different rng streams;
an outlier batch is N points from ONE parameter draw, never a mixture.

Latent nodes are ``pm.Normal`` random variables — ``pt.random.normal``
cannot be embedded in ``pm.Deterministic``.
"""

import numpy as np

from badgers.core.dgp.base import DGPMixin


class _ParameterDrawView(DGPMixin):
    """One pinned posterior draw of a fitted :class:`PyMCDGP` carrier.

    ``response``/``draw_noise`` read the carrier's stored posterior
    samples at the pinned index; all other protocol verbs (fit/sample/
    intervene) are the generic mixin algebra executed THROUGH this
    parameter draw's mechanisms — so edits over the pinned draw compose
    exactly like over any other DGP.
    """

    def __init__(self, carrier, index):
        self._carrier = carrier
        self._index = int(index)
        self.graph = carrier.graph
        self.column_mapping = carrier.column_mapping

    def fit_node(self, node, parents, data):
        self._carrier.fit_node(node, parents, data)

    def is_fitted(self):
        return self._carrier.is_fitted()

    def response(self, node, parent_values):
        carrier = self._carrier
        if node in carrier._beta:
            return parent_values @ carrier._beta[node][self._index]
        if node in carrier._mu:
            return np.full(
                parent_values.shape[0], carrier._mu[node][self._index]
            )
        raise RuntimeError(
            f"posterior draw has no fitted mechanism for node '{node}'"
        )

    def draw_noise(self, node, rng, n):
        sigma = self._carrier._sigma[node][self._index]
        return rng.normal(0.0, float(sigma), size=n)

    def pick_parameters(self, rng=None):
        """Already pinned: idempotent (a second cascade on this view uses
        the SAME parameter draw — re-picking would make results depend on
        how many cascades ran before)."""
        return self


class PyMCDGP(DGPMixin):
    """Per-node Bayesian linear DGP fitted with PyMC MCMC.

    Keeps the FULL posterior (no mean collapse):

    - root node: posterior draws of ``mu`` and ``sigma``;
    - non-root node: posterior draws of ``beta`` (one row per draw, one
      weight per parent in structural predecessor order) and ``sigma``.

    ``response``/``draw_noise`` on the carrier itself evaluate draw index
    0 (a canonical parameter set). To generate from the posterior, the
    generic ``sample``/``intervene`` call the hook
    :meth:`pick_parameters` at the top of each cascade, which pins ONE
    draw from the caller's rng and returns a ``_ParameterDrawView``.

    Requires the optional ``pymc`` extra (``pip install badgers[pymc]``).
    MCMC latency is a different cost class than the deterministic
    adapters — use it for
    posterior-faithful fits, not for the dependency-light default path.
    """

    def __init__(self, graph, column_mapping, n_draws=200):
        self.graph = graph
        self.column_mapping = column_mapping
        self.n_draws = n_draws
        self._beta = {}
        self._mu = {}
        self._sigma = {}

    def is_fitted(self):
        """True iff every graph node has a fitted mechanism (G9)."""
        return all(
            node in self._mu or node in self._beta
            for node in self.graph.nodes
        )

    def fit_node(self, node, parents, data):
        import pymc as pm

        col = self.column_mapping[node]
        # Complete-case fit (G9): NaN-contaminated
        # rows are dropped for the ESTIMATION only; too few rows cannot
        # identify the mechanism.
        mask = np.isfinite(data[:, col])
        for p in parents:
            mask &= np.isfinite(data[:, self.column_mapping[p]])
        d = data[mask]
        if d.shape[0] < len(parents) + 1:
            raise ValueError(
                f"Cannot fit node '{node}': only {d.shape[0]} complete-case "
                f"rows available (need at least {len(parents) + 1}); the "
                f"data contains too many NaNs."
            )
        with pm.Model() as model:
            if not parents:
                mu_i = pm.Normal("mu", 0, 10)
                sigma_i = pm.HalfNormal("sigma", 1)
                pm.Normal("x", mu=mu_i, sigma=sigma_i,
                          observed=d[:, col])
            else:
                pcols = [self.column_mapping[p] for p in parents]
                beta_i = pm.Normal("beta", 0, 10, shape=len(parents))
                sigma_i = pm.HalfNormal("sigma", 1)
                pm.Normal("x", mu=pm.math.dot(d[:, pcols], beta_i),
                          sigma=sigma_i, observed=d[:, col])
            trace = pm.sample(
                draws=self.n_draws,
                tune=max(self.n_draws // 2, 100),
                chains=1,
                cores=1,
                progressbar=False,
                compute_convergence_checks=False,
            )
        if not parents:
            self._mu[node] = np.asarray(
                trace.posterior["mu"].values.ravel()
            )
            self._sigma[node] = np.asarray(
                trace.posterior["sigma"].values.ravel()
            )
            return
        self._beta[node] = np.asarray(
            trace.posterior["beta"].values.reshape(self.n_draws, -1)
        )
        self._sigma[node] = np.asarray(
            trace.posterior["sigma"].values.ravel()
        )

    def _is_root(self, node):
        return node not in self._beta

    def pick_parameters(self, rng=None):
        """Select ONE posterior draw (from ``rng``) as this cascade's
        parameter set; return a ``_ParameterDrawView`` of this carrier.
        """
        idx = int(rng.integers(self.n_draws))
        return _ParameterDrawView(self, idx)

    def response(self, node, parent_values):
        if self._is_root(node):
            if node not in self._mu:
                raise RuntimeError(
                    f"PyMCDGP.response called before fit_node for node "
                    f"'{node}'"
                )
            return np.full(parent_values.shape[0], self._mu[node][0])
        return parent_values @ self._beta[node][0]

    def draw_noise(self, node, rng, n):
        if node not in self._sigma:
            raise RuntimeError(
                f"PyMCDGP.draw_noise called before fit_node for node "
                f"'{node}'"
            )
        return rng.normal(0.0, float(self._sigma[node][0]), size=n)
