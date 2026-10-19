"""Adapter for user-built PyMC structural causal models.

PyMC has no SCM object — its idiom is ONE generative ``pm.Model``. The
convention this adapter accepts (PyMC 6.3.2):

- each STRUCTURAL node is a named OBSERVED random variable whose name
  appears in ``column_mapping`` (the data column it explains);
- the node's structural function lives in its distribution parameters:
  a ``Normal("B", mu=pt.sin(A), sigma=sigma_B, observed=...)`` declares
  ``B = sin(A) + Normal(0, sigma_B)``; free RVs (``sigma_B``, regression
  weights, …) are the Bayesian parameters inferred jointly by
  ``pm.sample`` on the whole model;
- discrete nodes (``Bernoulli``, ``Poisson``, ``Binomial``,
  ``Categorical``) have their parameter expression as function of the
  parent RVs; they are numeric-encoded floats in sampled arrays.

Badgers does NOT replace PyMC: ``fit`` is one ``pm.sample`` (the FULL
posterior is kept — never a mean); sampling/interventions run on the
generic ``DGPMixin`` path with ONE coherent pinned posterior draw per
cascade (see :meth:`~badgers.core.dgp.base.DGPMixin.pick_parameters`).

Why op-by-op graph rebuild instead of ``clone_replace``: pytensor
reconciles static shapes and pins a ``SpecifyShape`` node to the
observed data length, so cloned parameter expressions only evaluate for
the training row count. Rebuilding op-by-op with fresh shape-free
inputs avoids that — and every RV (structural node or free parameter)
MUST be substituted: an unsubstituted RV becomes a freshly DRAWN random
variable when the compiled function is evaluated.
"""

from pytensor.graph.traversal import variable_ancestors
from pytensor.tensor.random.op import RandomVariable

import networkx as nx
import numpy as np
import pytensor
import pytensor.tensor as pt

from badgers.core.dgp.base import DGPMixin

# additive continuous families: params (op signature order, after
# rng/size), which entry is the location, which the scale
_ADDITIVE_FAMILIES = {
    "NormalRV": ("mu", "sigma"),
    "StudentTRV": ("nu", "mu", "sigma"),
    "LaplaceRV": ("mu", "b"),
}
# discrete families: their parameters depend on PARENT VALUES, so they
# are sampled wholesale in draw_noise (pending-capture pattern)
_DISCRETE_FAMILIES = ("BernoulliRV", "PoissonRV", "BinomialRV", "CategoricalRV")


def _rebuild(var, subst, memo):
    """Rebuild a subgraph op-by-op with substituted inputs.

    Bypasses pytensor's static-shape reconciliation (which clones carry
    over from the observed data via ``SpecifyShape``). ``subst`` maps
    RV variables to fresh shape-free inputs; RVs left unsubstituted
    would be freshly DRAWN at evaluation — the caller must substitute
    every RV (validated separately).
    """
    if var in subst:
        return subst[var]
    if var in memo:
        return memo[var]
    if var.owner is None:
        return var
    new_inputs = [_rebuild(i, subst, memo) for i in var.owner.inputs]
    new_apply = var.owner.op.make_node(*new_inputs)
    for orig, new in zip(var.owner.outputs, new_apply.outputs):
        memo[orig] = new
    return memo[var]


class _ParameterDrawView(DGPMixin):
    """One pinned posterior draw of a fitted :class:`PyMCSCMDGP` carrier.

    ``response``/``draw_noise`` evaluate the carrier's compiled node
    functions with THIS draw's parameter values; all other protocol
    verbs (fit/sample/intervene) are the generic mixin algebra executed
    through this parameter draw's mechanisms.
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

    def _pinned(self):
        return self._carrier._pinned_params(self._index)

    def response(self, node, parent_values):
        return self._carrier.response(node, parent_values, pinned=self._pinned())

    def draw_noise(self, node, rng, n):
        return self._carrier.draw_noise(node, rng, n, pinned=self._pinned())

    def pick_parameters(self, rng=None):
        """Already pinned: idempotent."""
        return self


class PyMCSCMDGP(DGPMixin):
    """Plug a user-built PyMC causal model into badgers.

    Args:
        causal_model: a ``pm.Model`` following the module convention:
            structural nodes are named OBSERVED RVs listed in
            ``column_mapping``; their structural functions live in the
            distribution parameters; free RVs are Bayesian parameters.
        column_mapping: dict node name -> column index of the numpy
            data. Required (it NAMES the structural nodes).
        n_draws: posterior draws kept from ``pm.sample``.

    Requires the optional ``pymc`` extra (``pip install badgers[pymc]``).
    Latent free RVs must be continuous (PyMC's default NUTS step
    method); observed discrete nodes are supported.
    """

    def __init__(self, causal_model, column_mapping, n_draws=200):
        self._scm = causal_model
        self.column_mapping = dict(column_mapping)
        self.n_draws = int(n_draws)
        self._idata = None
        self._fns = {}        # node -> compiled param functions
        self._families = {}   # node -> (family name, param exprs)
        self._param_inputs = {}  # node -> fresh inputs per referenced RV
        self._pending = {}    # node -> parent_values (discrete nodes)
        self._free_names = None
        self._sampled = False
        self._derive_structure()

    # -- structure derivation ------------------------------------------------

    def _derive_structure(self):
        """Derive the DAG from the model's own graph BEFORE fitting.

        Uses the official ``pm.model_graph.model_to_networkx`` — its
        predecessors of a structural node include BOTH structural-node
        RVs (true causal parents) and parameter/free RVs (e.g. sigma);
        only the structural names are kept as parents. Structural RVs
        are exactly the ``column_mapping`` names; anything else must be
        a free RV (or a constant).
        """
        import pymc as pm

        structural = {}
        for name in self.column_mapping:
            rv = self._scm.named_vars.get(name)
            if rv is None:
                raise ValueError(
                    f"Structural node '{name}' not found in the PyMC "
                    f"model's named variables."
                )
            structural[name] = rv

        graph = nx.DiGraph()
        graph.add_nodes_from(self.column_mapping)
        self._families = {}
        model_graph = pm.model_graph.model_to_networkx(self._scm)
        for name, rv in structural.items():
            if not rv.owner or rv.owner.inputs is None:
                raise ValueError(f"Node '{name}' is not a random variable.")
            family = type(rv.owner.op).__name__
            if family not in _ADDITIVE_FAMILIES and family not in _DISCRETE_FAMILIES:
                raise NotImplementedError(
                    f"PyMCSCMDGP does not support distribution family "
                    f"{family} for node '{name}' yet. Supported: "
                    f"{sorted(_ADDITIVE_FAMILIES) + list(_DISCRETE_FAMILIES)}."
                )
            # official graph: predecessors minus parameter RVs = parents
            parents = {
                pred for pred in model_graph.predecessors(name)
                if pred in structural and pred != name
            }
            for parent in sorted(parents):
                graph.add_edge(parent, name)
        self.graph = graph

    # -- fitting -------------------------------------------------------------

    def fit(self, X):
        """Validate the model, run ONE joint ``pm.sample``, compile the
        per-node parameter functions. NaN handling is delegated to the
        model the user wrote (complete-case policies are the user's
        PyMC modeling choice)."""
        import pymc as pm

        self._derive_structure()
        # validate observed-ness
        for name in self.column_mapping:
            observed_names = {v.name for v in self._scm.observed_RVs}
            if name not in observed_names:
                raise ValueError(
                    f"Structural node '{name}' must be an observed RV "
                    f"(observed=<column>) — the joint posterior is "
                    f"identified through the observed data."
                )
        # compile + run the unsubstituted-RV safety scan BEFORE sampling:
        # validation errors must surface regardless of whether pm.sample
        # would even run (e.g. a model with no free RVs raises PyMC's
        # SamplingError before the scan could fire)
        self._compile_nodes()
        if self._scm.free_RVs:
            with self._scm:
                self._idata = pm.sample(
                    draws=self.n_draws,
                    tune=max(self.n_draws // 2, 100),
                    chains=1,
                    cores=1,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            self._free_names = [rv.name for rv in self._scm.free_RVs]
        else:
            # degenerate SCM: every structural function is a constant —
            # nothing to sample; the posterior draw index is a no-op
            self._free_names = []
        self._sampled = True
        return self

    def _compile_nodes(self):
        """Compile per-node parameter functions with shape-free inputs."""
        self._fns = {}
        self._param_inputs = {}
        for node in self.graph.nodes:
            rv = self._scm.named_vars[node]
            family = type(rv.owner.op).__name__
            params = list(rv.owner.inputs[2:])
            subst = {}
            # substitute every RV: parents (fresh per-node inputs) and
            # ALL free parameters (fresh inputs, one global input list)
            # The subst dict insertion order IS the compiled function
            # input order: parents (graph-predecessor order) first, then
            # free params in scm.free_RVs order — _eval relies on this.
            for parent in self.graph.predecessors(node):
                subst[self._scm.named_vars[parent]] = pt.vector(
                    f"{parent}__in_{node}", dtype="float64"
                )
            for free_rv in self._scm.free_RVs:
                if free_rv in subst:
                    continue
                subst[free_rv] = (
                    pt.vector(f"{free_rv.name}__in_{node}", dtype="float64")
                    if free_rv.owner.op.ndim_supp > 0
                    else pt.scalar(f"{free_rv.name}__in_{node}", dtype="float64")
                )
            memo = {}
            rebuilt = [_rebuild(e, subst, memo) for e in params]
            # safety scan on the REBUILT graph: a correct rebuild
            # substitutes EVERY RV with a plain vector/scalar input, so
            # no RV op may remain — any leftover is a freshly DRAWN RV
            # at evaluation (silent corruption — verified failure
            # mode). Map rebuilt RV copies back to the originals (via
            # the memo) to name the offender; references to non-parent
            # structural RVs land here.
            original_of = {
                new: orig for orig, new in memo.items()
                if new.owner and isinstance(new.owner.op, RandomVariable)
            }
            for expr in rebuilt:
                for anc in variable_ancestors([expr]):
                    if anc.owner and isinstance(
                        anc.owner.op, RandomVariable
                    ):
                        offender = original_of.get(anc, anc)
                        raise ValueError(
                            f"Node '{node}': its structural function "
                            f"references RV '{offender.name}' which is "
                            f"not one of its parents "
                            f"{[p for p in self.graph.predecessors(node)]} "
                            f"— unsubstituted RVs would be freshly drawn "
                            f"at evaluation (silent corruption)."
                        )
            inputs = list(subst.values())
            fns = [
                pytensor.function(inputs, expr, on_unused_input="ignore")
                for expr in rebuilt
            ]
            self._fns[node] = fns
            self._param_inputs[node] = inputs
            self._families[node] = (family, _ADDITIVE_FAMILIES.get(family))

    def is_fitted(self):
        """True iff the posterior is sampled and every node is compiled."""
        return (
            self._sampled
            and all(node in self._fns for node in self.graph.nodes)
        )

    def _posterior_flat(self, name):
        """(chains, draws, *shape) -> (chains*draws, *shape)."""
        values = np.asarray(self._idata.posterior[name].values)
        return values.reshape((-1,) + values.shape[2:])

    def _pinned_params(self, index):
        return {name: self._posterior_flat(name)[index]
                for name in self._free_names}

    def pick_parameters(self, rng=None):
        """Select ONE coherent posterior draw (from ``rng``) for this
        cascade; return a ``_ParameterDrawView``."""
        if self._free_names:
            total = self._posterior_flat(self._free_names[0]).shape[0]
        else:
            total = 1  # degenerate SCM: all parameters fixed constants
        return _ParameterDrawView(self, int(rng.integers(total)))

    # -- evaluation ----------------------------------------------------------

    def _eval(self, node, parent_values, pinned, which):
        family, roles = self._families[node]
        fns = self._fns[node]
        inputs = self._param_inputs[node]
        args = []
        # inputs order: parents first, then free params — matching the
        # subst dict insertion order in _compile_nodes. parent_values
        # columns are in get_parents (== graph.predecessors) order.
        for col in range(parent_values.shape[1]):
            args.append(parent_values[:, col])
        for name in self._free_names:
            args.append(pinned[name])
        if which == "all":
            return [np.asarray(fn(*args)) for fn in fns]
        return np.asarray(fns[which](*args))

    def response(self, node, parent_values, pinned=None):
        pinned = pinned or self._pinned_params(0)
        family, roles = self._families[node]
        # evaluate ALL parameters once (they may depend on parents);
        # draw_noise reuses them via _pending — it takes no parents.
        params = self._eval(node, parent_values, pinned, "all")
        self._pending[node] = params
        if family in _DISCRETE_FAMILIES:
            # deterministic part contributes nothing: the VALUE comes
            # from draw_noise (whole-distribution draw)
            return np.zeros(parent_values.shape[0])
        loc = np.asarray(params[roles.index("mu")]).ravel()
        if loc.size == 1:
            loc = np.full(parent_values.shape[0], float(loc[0] if loc.shape else loc))
        return loc

    def draw_noise(self, node, rng, n, pinned=None):
        family, roles = self._families[node]
        if node not in self._pending:
            raise RuntimeError(
                f"PyMCSCMDGP.draw_noise for node '{node}' requires a "
                f"preceding response(node, parent_values) call in the "
                f"same cascade"
            )
        params = self._pending.pop(node)  # set by the preceding response
        if family in _DISCRETE_FAMILIES:
            return self._draw_discrete(node, family, params, rng, n)
        # centered noise: loc already returned by response; the shape
        # parameter may be a scalar OR per-row (heteroscedastic)
        scale_idx = next(
            (roles.index(k) for k in ("sigma", "b") if k in roles), None
        )
        if scale_idx is not None:
            scale = np.broadcast_to(
                np.asarray(params[scale_idx], dtype=float).ravel(), (n,)
            )
        else:
            scale = 1.0
        if family == "NormalRV":
            return rng.normal(0.0, scale, size=n)
        if family == "StudentTRV":
            nu = np.asarray(params[roles.index("nu")], dtype=float).ravel()
            if nu.size not in (1, n):
                raise ValueError(
                    f"Node '{node}': StudentT nu must be scalar or "
                    f"per-row (size {n}); got size {nu.size}"
                )
            return rng.standard_t(nu if nu.size == 1 else nu[0], size=n) * scale
        if family == "LaplaceRV":
            return rng.laplace(0.0, scale, size=n)
        raise NotImplementedError(f"No centered-noise rule for {family}")

    def _draw_discrete(self, node, family, params, rng, n):
        # per-row parameters must have shape (n,): a root node's scalar
        # (or (1,)-shaped) parameter would otherwise produce a single
        # draw broadcast to every row
        if family == "BernoulliRV":
            p = np.broadcast_to(np.asarray(params[0], dtype=float).ravel(), (n,))
            return (rng.random(n) < p).astype(float)
        if family == "PoissonRV":
            mu = np.broadcast_to(
                np.asarray(params[0], dtype=float).ravel(), (n,)
            )
            return rng.poisson(np.maximum(mu, 1e-9)).astype(float)
        if family == "BinomialRV":
            n_trials = np.broadcast_to(
                np.asarray(params[0]).ravel(), (n,)
            )
            p = np.broadcast_to(np.asarray(params[1], dtype=float).ravel(), (n,))
            return rng.binomial(n_trials.astype(int), p).astype(float)
        if family == "CategoricalRV":
            probs = np.asarray(params[0])
            probs = np.broadcast_to(probs, (n, probs.shape[-1]))
            u = rng.random(n)
            cum = np.cumsum(probs, axis=1)
            idx = (u[:, None] > cum).sum(axis=1)
            return np.clip(idx, 0, probs.shape[1] - 1).astype(float)
        raise NotImplementedError(f"No discrete draw rule for {family}")
