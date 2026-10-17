"""Builder for docs/dev/Create-New-DGP-Backends.ipynb (run after _build_dev_nbs.py)."""
import json
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"c:\Users\siebert\PycharmProjects\badgers"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in lines]}


def write(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with io.open(full, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", path, "cells:", len(cells))


cells = []

cells.append(md(
    "# Creating DGP backends",
    "",
    "This notebook explains the **DGP protocol** — the abstraction behind",
    "badgers' causal generators — and walks through implementing a",
    "**estimation backend** for it.",
    "",
    "A DGP (*data-generating process*) is a fitted model of how the",
    "columns of a dataset causally depend on each other: given a",
    "graph and data, it can sample fresh data from the estimated joint",
    "distribution, and recompute descendants after an intervention",
    "(Pearl surgery). *Backends* differ only in HOW each node's mechanism",
    "is estimated and evaluated.",
))

cells.append(md(
    "## The architecture: protocol core + generic mixin",
    "",
    "badgers splits the DGP into two layers:",
    "",
    "**1. Backend protocol core** — the three per-node methods YOUR backend",
    "must implement:",
    "",
    "```python",
    "fit_node(node, parents, data)      # estimate node's mechanism from data",
    "response(node, parent_values)      # deterministic part f(parents)",
    "draw_noise(node, rng, n)           # node's exogenous shock for n rows",
    "```",
    "",
    "**2. `DGPMixin` generic machinery** (in `badgers.core.dgp.base`) —",
    "shared by all backends, no need to re-implement:",
    "",
    "* `fit(data)` — topologically walks the graph, calls `fit_node` per node;",
    "* `sample(rng, n)` — ancestors first: `response + draw_noise` per node;",
    "* `intervene(rows, nodes, values=None, rng=...)` — pin intervention",
    "  values IN PLACE, then recompute ONLY descendants through the",
    "  topological order (pass `rows.copy()` to keep the original);",
    "* structure edits (`scale_edge`, ...) built on those primitives;",
    "* `is_fitted()`, NaN handling (complete-case estimation, G9), and",
    "  reproducibility via the caller's rng.",
    "",
    "The core is deliberately tiny: any estimator that can answer",
    "*\"what is this node given its parents, and what is its noise?\"* can",
    "plug in — linear regression, GCM additive-noise models, PyMC",
    "posteriors, ...",
))

cells.append(code(
    "import numpy as np",
    "import networkx as nx",
    "from numpy.random import default_rng",
    "",
    "from badgers.core.dgp import OlsDGP",
    "from badgers.core.dgp.base import DGPMixin",
    "",
    "# reference DGP for the whole notebook",
    "graph = nx.DiGraph([('A', 'B'), ('B', 'C')])",
    "column_mapping = {'A': 0, 'B': 1, 'C': 2}",
    "",
    "rng = default_rng(0)",
    "X = np.zeros((500, 3))",
    "X[:, 0] = rng.normal(size=500)",
    "X[:, 1] = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=500)",
    "X[:, 2] = 0.5 * X[:, 1] + rng.normal(scale=0.1, size=500)",
    "",
    "dgp = OlsDGP(graph, column_mapping).fit(X)",
    "print('fitted:', dgp.is_fitted())",
    "print('sample:', dgp.sample(default_rng(1), 5).shape)",
    "Xi = dgp.intervene(X.copy(), ['B'], [np.full(500, 10.0)], rng=default_rng(2))",
    "print('after do(B=10): B == 10:', np.allclose(Xi[:, 1], 10.0), '| C ~ 5:', np.allclose(Xi[:, 2], 5.0, atol=0.5))",
))

cells.append(md(
    "## The contract in detail",
    "",
    "### `fit_node(node, parents, data)`",
    "",
    "* called once per node in topological order;",
    "* `data` is the **full** 2D array; resolve your column with",
    "  `self.column_mapping[node]` and parent columns via the mapping too;",
    "* **NaN policy (G9)**: estimate on complete-case rows only",
    "  (`np.isfinite` over the node + parent columns); raise `ValueError`",
    "  if too few rows remain to identify the mechanism;",
    "* store whatever parameters you need — nothing is copied for you.",
    "",
    "### `response(node, parent_values)`",
    "",
    "* the deterministic part `f(parents)` for an `(n, n_parents)` array;",
    "* must return a 1D array of length `n` (mind sklearn's `(n, 1)`",
    "  outputs — ravel!);",
    "* root nodes: return the fitted marginal (e.g. mean) broadcast to `n`.",
    "",
    "### `draw_noise(node, rng, n)`",
    "",
    "* the node's exogenous shock for `n` rows, drawn from the **caller's",
    "  rng** — this is what makes the whole library reproducible;",
    "* do NOT use global `np.random` (third-party libraries often do —",
    "  wrap their sampling paths);",
    "* convention: `response(node) + draw_noise(node)` reconstructs the",
    "  node's fitted marginal (e.g. for a linear ANM, noise = residuals;",
    "  for empirical roots, resample the centered observations).",
))

cells.append(md(
    "## Building a custom backend: ridge ANM",
    "",
    "Let's implement a backend that estimates each non-root node with a",
    "**ridge regression** (L2-regularized) and stores Gaussian residual",
    "noise with an estimated std. It subclasses `DGPMixin` so it inherits",
    "`fit`/`sample`/`intervene` and the edits.",
))

cells.append(code(
    "from sklearn.linear_model import Ridge",
    "",
    "",
    "class RidgeAnmDGP(DGPMixin):",
    "    \"\"\"Additive-noise DGP with ridge-regression deterministic parts and",
    "    Gaussian residual noise.\"\"\"",
    "",
    "    def __init__(self, graph, column_mapping, alpha=1.0):",
    "        # DGPMixin is a plain mixin: no __init__ — store attributes directly",
    "        self.graph = graph",
    "        self.column_mapping = column_mapping",
    "        self.alpha = alpha",
    "        self._model = {}     # node -> fitted Ridge (non-root)",
    "        self._mu = {}        # node -> fitted mean (root)",
    "        self._noise_std = {} # node -> residual std (non-root)",
    "",
    "    def is_fitted(self):",
    "        return all(node in self._mu or node in self._model",
    "                   for node in self.graph.nodes)",
    "",
    "    def fit_node(self, node, parents, data):",
    "        col = self.column_mapping[node]",
    "        mask = np.isfinite(data[:, col])",
    "        for p in parents:",
    "            mask &= np.isfinite(data[:, self.column_mapping[p]])",
    "        d = data[mask]",
    "        if d.shape[0] < len(parents) + 1:",
    "            raise ValueError(f\"Cannot fit node '{node}': too many NaN rows\")",
    "        if not parents:",
    "            self._mu[node] = float(np.mean(d[:, col]))",
    "            return",
    "        pcols = [self.column_mapping[p] for p in parents]",
    "        model = Ridge(alpha=self.alpha)",
    "        model.fit(d[:, pcols], d[:, col])",
    "        self._model[node] = model",
    "        residuals = d[:, col] - model.predict(d[:, pcols])",
    "        self._noise_std[node] = float(np.std(residuals, ddof=1))",
    "",
    "    def response(self, node, parent_values):",
    "        if node in self._model:",
    "            return np.asarray(self._model[node].predict(parent_values), dtype=float).ravel()",
    "        return np.full(parent_values.shape[0], self._mu[node])",
    "",
    "    def draw_noise(self, node, rng, n):",
    "        if node in self._model:",
    "            return rng.normal(scale=self._noise_std[node], size=n)",
    "        # root: centered standard noise around the fitted mean",
    "        return rng.normal(size=n)",
    "",
    "",
    "ridge_dgp = RidgeAnmDGP(graph, column_mapping, alpha=1.0).fit(X)",
    "print('fitted:', ridge_dgp.is_fitted())",
    "Xs = ridge_dgp.sample(default_rng(2), 5)",
    "print('sample shape:', Xs.shape)",
))

cells.append(md(
    "## Conformance: your backend through the generic machinery",
    "",
    "A custom backend is trustworthy when it passes the same checks as the",
    "built-ins. Three invariants to verify:",
    "",
    "1. **Sampling** produces finite data with the fitted marginal roughly",
    "   matching the training data;",
    "2. **Interventions** pin the intervened nodes and recompute only",
    "   descendants (ancestors untouched);",
    "3. **Reproducibility**: same rng seed → same output.",
))

cells.append(code(
    "rng = default_rng(0)",
    "",
    "# 1. sampling: finite + rough marginal agreement",
    "Xs = ridge_dgp.sample(default_rng(3), 2000)",
    "assert np.all(np.isfinite(Xs))",
    "print('marginal means (train vs sampled):',",
    "      np.round(X.mean(axis=0), 2), np.round(Xs.mean(axis=0), 2))",
    "",
    "# 2. interventions: B pinned, C recomputed, A untouched (in place!",
    "#    pass a copy to keep the original)",
    "Xi = ridge_dgp.intervene(X.copy(), ['B'], [np.full(500, 10.0)], rng=default_rng(4))",
    "assert np.allclose(Xi[:, 1], 10.0)",
    "assert np.allclose(Xi[:, 0], X[:, 0])            # ancestor unchanged",
    "print('C after do(B=10) ~', np.round(Xi[:, 2].mean(), 2), '(theory ~5)')",
    "",
    "# 3. reproducibility",
    "s1 = ridge_dgp.sample(default_rng(9), 10)",
    "s2 = ridge_dgp.sample(default_rng(9), 10)",
    "assert np.array_equal(s1, s2)",
    "print('reproducible: True')",
))

cells.append(md(
    "## Consuming backends from the causal generators",
    "",
    "Any backend implementing the protocol works with every causal",
    "generator — that's the point of the abstraction:",
    "",
    "```python",
    "from badgers.generators.tabular_data.outliers.causal import CausalOutlierPropagationGenerator",
    "",
    "gen = CausalOutlierPropagationGenerator(random_generator=rng, dgp=ridge_dgp)",
    "Xt, y = gen.generate(X, y=None, perturbation_nodes=['A'], n_outliers=5)",
    "```",
    "",
    "The generator performs the intervention itself (severing incoming edges",
    "of the perturbation nodes, fresh noise for ancestors, recomputing",
    "descendants through `dgp.intervene`) — your backend only needs the",
    "three core methods.",
))

cells.append(code(
    "from badgers.generators.tabular_data.outliers.causal import CausalOutlierPropagationGenerator",
    "",
    "rng = default_rng(0)",
    "gen = CausalOutlierPropagationGenerator(random_generator=rng, dgp=ridge_dgp)",
    "Xt, y_out = gen.generate(X, y=None, perturbation_nodes=['A'], n_outliers=5)",
    "",
    "labels = gen.get_outliers_labels()",
    "print('shape:', X.shape, '->', Xt.shape, '| outliers:', int(labels.sum()))",
    "print('original rows unchanged:', np.allclose(Xt[:500], X))",
))

cells.append(md(
    "## Conformance checklist for a DGP backend",
    "",
    "* [ ] `fit_node` handles root and non-root nodes; complete-case NaN",
    "  policy; clear `ValueError` when under-identified;",
    "* [ ] `response` returns 1D arrays (ravel sklearn outputs);",
    "* [ ] `draw_noise` uses the caller's rng only — no global np.random;",
    "* [ ] `response + draw_noise` reconstructs the fitted marginal;",
    "* [ ] `is_fitted()` correct before/after `fit()`;",
    "* [ ] passes the three conformance invariants (sampling, intervention",
    "  pinning + ancestor preservation, reproducibility);",
    "* [ ] works with at least one causal generator end-to-end;",
    "* [ ] graph / column_mapping stored as constructor attributes with the",
    "  expected names (`graph`, `column_mapping`).",
))

write(r"docs\dev\Create-New-DGP-Backends.ipynb", cells)

print("DONE")
