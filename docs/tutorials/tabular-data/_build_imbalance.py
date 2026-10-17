"""Builder: merge Imbalance-Tabular-Data.ipynb + Causal-Imbalance.ipynb -> Imbalance.ipynb.

Also fixes:
- the misleading `y = -2*X[:,0] + 3*X[:,1] + 5` variable is renamed to `score`
  (it is a demo ranking variable, not a target);
- selector purpose explained explicitly in each section;
- the dead last section is repaired: `sampling_proba_func` is now actually
  used through `RandomSamplingFeaturesGenerator`.
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"c:\Users\siebert\PycharmProjects\badgers"
SRC = os.path.join(ROOT, "docs", "tutorials", "tabular-data")


def load(name):
    with io.open(os.path.join(SRC, name), "r", encoding="utf-8") as f:
        return json.load(f)["cells"]


a = load("Imbalance-Tabular-Data.ipynb")
b = load("Causal-Imbalance.ipynb")


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells = []

# ---- intro ----------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# Generating class imbalance\n",
        "\n",
        "**Imbalance** is a *subsampling* deficit: instead of (or in addition\n",
        "to) changing values, the generator **drops rows** — so `Xt` is\n",
        "*smaller* than `X`. Real datasets are rarely balanced: rare events,\n",
        "censored observations, or biased sampling processes skew the row\n",
        "distribution. This makes pipelines that assume balanced classes or\n",
        "uniform feature coverage fail in instructive ways.\n",
        "\n",
        "This notebook shows the **two levers** badgers gives you:\n",
        "\n",
        "1. **Acting on the observation process** — resample rows directly,\n",
        "   with the affected rows decided by a **selector**\n",
        "   (`RandomSamplingClassesGenerator`, `RandomSamplingTargetsGenerator`,\n",
        "   `RandomSamplingFeaturesGenerator`). These generators *do not use a\n",
        "   DGP*.\n",
        "2. **Acting on the data-generating process (DGP)** — truncate the data\n",
        "   to a quantile range of a *fitted causal model*'s node\n",
        "   (`CausalImbalanceGenerator`).\n",
    ]})

# ---- setup ----------------------------------------------------------------
cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "\n",
    "from numpy.random import default_rng\n",
    "from sklearn.datasets import make_blobs\n",
    "\n",
    "from badgers.generators.tabular_data.imbalance.imbalance import (\n",
    "    RandomSamplingFeaturesGenerator, RandomSamplingTargetsGenerator, RandomSamplingClassesGenerator,\n",
    ")\n",
    "from badgers.core.selectors import ClassSelector, ProportionalSelector\n",
    "from badgers.core.utils import normalize_proba\n",
    "\n",
    "rng = default_rng(seed=0)\n",
))

# ---- data -----------------------------------------------------------------
cells.append(code(
    "X, y = make_blobs(centers=4, random_state=0)\n",
    "# classes as a column of X (y is deprecated as a data carrier)\n",
    "X = np.column_stack([X, y])\n",
    "column_mapping = {f\"f{i}\": i for i in range(X.shape[1])}\n",
    "column_mapping[\"cls\"] = X.shape[1] - 1\n",
    "cls = X[:, column_mapping[\"cls\"]].astype(int)   # explicit class vector for plotting\n",
    "print(X.shape)",
))

# ---- part A ---------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part A — Acting on the observation process (no DGP)\n",
        "\n",
        "### Sample data from different classes\n",
        "\n",
        "**What the selector does here:** `ClassSelector` builds one boolean\n",
        "mask per class value from the shares you give it. The shares are\n",
        "*fractions of the total row count*, clipped to the rows available in\n",
        "each class; classes absent from the mapping are kept as-is.\n",
        "`RandomSamplingClassesGenerator` then drops exactly the fired rows.",
    ]})

cells.append(code(
    "# WHO loses rows and how many: per-class share of the total row count\n",
    "selector = ClassSelector(\"cls\", column_mapping, {\"cls\": {0:0.1, 1:0.05, 2:0.15, 3:0.2}}, rng=rng)\n",
    "trf = RandomSamplingClassesGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X=X.copy(), selector=selector)\n",
    "print('subsampling:', X.shape, '->', Xt.shape)",
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))\n",
    "sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cls, legend='auto', ax=axes[0])\n",
    "sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], hue=Xt[:, column_mapping['cls']].astype(int), legend='auto', ax=axes[1])\n",
    "axes[0].set_title('Original')\n",
    "axes[1].set_title('Transformed')\n",
    "plt.tight_layout()",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Sample data proportionally to a column (weighted resampling)\n",
        "\n",
        "**What the selector does here:** `ProportionalSelector(\"f0\", ...)`\n",
        "returns per-row **float scores in [0, 1]** — the deficit probability\n",
        "grows with the value of `f0`. The consumer,\n",
        "`RandomSamplingTargetsGenerator`, resamples rows with probability\n",
        "proportional to `1 - score`: rows with a HIGH deficit probability are\n",
        "largely dropped, rows with a LOW deficit probability are kept (and can\n",
        "even be replicated). The result concentrates on small-`f0` rows — a\n",
        "form of imbalance driven by a continuous column.\n",
        "\n",
        "For illustration we also compute a demo ranking variable `score` and\n",
        "use it as a color: it is **not** a target and plays no role in the\n",
        "selection.",
    ]})

cells.append(code(
    "# demo ranking variable for coloring only (y is deprecated as a data carrier)\n",
    "score = -2 * X[:, 0] + 3 * X[:, 1] + 5\n",
    "\n",
    "# WHO loses rows: deficit probability proportional to f0\n",
    "selector = ProportionalSelector(\"f0\", {\"f0\": 0, \"f1\": 1, \"cls\": 2})\n",
    "trf = RandomSamplingTargetsGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X=X.copy(), selector=selector)\n",
    "print('subsampling:', X.shape, '->', Xt.shape)",
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))\n",
    "sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=score, legend=False, ax=axes[0])\n",
    "sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], legend=False, ax=axes[1])\n",
    "axes[0].set_title('Original (color = demo score)')\n",
    "axes[1].set_title('Transformed')\n",
    "plt.tight_layout()",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Sample data based upon the features themselves\n",
        "\n",
        "`RandomSamplingFeaturesGenerator` takes a **`sampling_proba_func`**:\n",
        "a function from `X` (the whole feature matrix) to one sampling\n",
        "probability per row. Rows with a high probability are resampled\n",
        "*more often* into the output (the function defines the *keep*\n",
        "weights, the inverse view of the selector scores above). Below we\n",
        "make the sampling probability decrease linearly with `f0`, so the\n",
        "output concentrates on small-`f0` rows again — this time expressed\n",
        "with a custom keep-probability function instead of a selector.",
    ]})

cells.append(code(
    "def sampling_proba_func(X_):\n",
    "    feature = X_[:, 0]\n",
    "    return normalize_proba(\n",
    "        (feature - np.min(feature)) / (np.max(feature) - np.min(feature))\n",
    "    )\n",
    "\n",
    "trf = RandomSamplingFeaturesGenerator(random_generator=rng)\n",
    "Xt_feat, _ = trf.generate(X.copy(), sampling_proba_func=sampling_proba_func)\n",
    "print('subsampling:', X.shape, '->', Xt_feat.shape)",
))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(8,8))\n",
    "\n",
    "sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cls, legend=False, ax=axes[0,0])\n",
    "sns.scatterplot(x=Xt_feat[:, 0], y=Xt_feat[:, 1], ax=axes[0,1])\n",
    "sns.histplot(x=X[:, 0], kde=True, ax=axes[1,0])\n",
    "sns.histplot(x=Xt_feat[:, 0], kde=True, ax=axes[1,1])\n",
    "axes[0,0].set_title('Original')\n",
    "axes[0,1].set_title('Transformed')\n",
    "\n",
    "plt.tight_layout()",
))

# ---- part B ---------------------------------------------------------------
cells.append(code(
    "from badgers.generators.tabular_data.imbalance.causal import CausalImbalanceGenerator\n",
    "from badgers.core.dgp import OlsDGP\n",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part B — Acting on the data-generating process (DGP)\n",
        "\n",
        "`CausalImbalanceGenerator` **truncates the dataset**: it keeps only\n",
        "the rows whose perturbation node falls inside `quantile_range`.\n",
        "Since the kept rows are untouched (no structural equation is\n",
        "re-evaluated), the descendants are untouched too — the result is a\n",
        "*pure subset*: a dataset that is *imbalanced* — concentrated in one\n",
        "part of the perturbation node's range — while remaining structurally\n",
        "consistent by construction.\n",
    ]})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "`CausalImbalanceGenerator` accepts either the built-in\n",
        "`quantile_range` truncation rule, or an explicit **selector** — the\n",
        "same `Selector` objects as Part A decide which rows are dropped\n",
        "(subsampling polarity: a fired row is removed). With an explicit\n",
        "selector, `perturbation_nodes`/`quantile_range` are ignored.\n",
    ]})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Build the graph, generate data, fit the DGP\n",
    ]})
cells.append(b[2])   # imports incl. plot helpers
cells.append(b[3])   # generate_linear_data helper
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import networkx as nx\n",
        "import pandas as pd\n",
        "\n",
        "from badgers.core.dgp import OlsDGP\n",
        "\n",
        "def generate_linear_data(graph, n_samples=100, rng=None, coefficients=None):\n",
        "    \"\"\"Generate data from a linear structural equation model.\"\"\"\n",
        "    nodes = list(nx.topological_sort(graph))\n",
        "    if coefficients is None:\n",
        "        coefficients = {}\n",
        "        for node in nodes:\n",
        "            parents = list(graph.predecessors(node))\n",
        "            coefficients[node] = {p: rng.uniform(0.5, 2.0) * rng.choice([-1, 1]) for p in parents}\n",
        "    column_mapping = {node: i for i, node in enumerate(nodes)}\n",
        "    data = np.zeros((n_samples, len(nodes)))\n",
        "    for node in nodes:\n",
        "        i = column_mapping[node]\n",
        "        parents = list(graph.predecessors(node))\n",
        "        if not parents:\n",
        "            data[:, i] = rng.normal(size=n_samples)\n",
        "        else:\n",
        "            for p, coef in coefficients[node].items():\n",
        "                data[:, i] += coef * data[:, column_mapping[p]]\n",
        "            data[:, i] += rng.normal(scale=0.1, size=n_samples)\n",
        "    return data, column_mapping\n",
    ]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "# --- Build the chain X -> Y -> Z ---\n",
        "chain = nx.DiGraph()\n",
        "chain.add_edges_from([(\"X\", \"Y\"), (\"Y\", \"Z\")])\n",
        "\n",
        "rng = np.random.default_rng(42)\n",
        "\n",
        "X, column_mapping = generate_linear_data(\n",
        "    chain, n_samples=100, rng=rng,\n",
        "    coefficients={'Y': {'X': 2.0}, 'Z': {'Y': 1.0}},\n",
        ")\n",
        "print(f\"X shape: {X.shape}\")\n",
        "print(f\"column_mapping: {column_mapping}\")",
    ]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "sns.pairplot(pd.DataFrame(X, columns=column_mapping.keys()))",
    ]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "dgp = OlsDGP(chain, column_mapping).fit(X.copy())\n",
        "\n",
        "gen = CausalImbalanceGenerator(random_generator=np.random.default_rng(0), dgp=dgp)\n",
        "print('fitted:', dgp.is_fitted_ if hasattr(dgp, 'is_fitted_') else 'ok')",
    ]})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Generate imbalance by truncation\n",
        "\n",
        "`quantile_range=(0.5, 1.0)` keeps the *upper* half of the Y range:\n",
        "rows with Y in the lower half are dropped.",
    ]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "Xt, yt = gen.generate(\n",
        "    X.copy(), y=None,\n",
        "    perturbation_nodes=[\"Y\"],\n",
        "    quantile_range=(0.5, 1.0),\n",
        ")\n",
        "print(f\"output shape: {X.shape} -> {Xt.shape}\")",
    ]})
cells.append(b[14])
cells.append(b[15])
cells.append(b[16])
cells.append(b[17])
cells.append(b[18])

# explicit selector demo
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Generate imbalance with an explicit selector\n",
        "\n",
        "The same selection machinery as Part A also plugs into the causal\n",
        "generator: a fired row is dropped. Here a `TopQuantileSelector`\n",
        "keeps rows where X is *low* (fires where X is high):",
    ]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from badgers.core.selectors import TopQuantileSelector\n",
        "\n",
        "sel = TopQuantileSelector(\n",
        "    variables=\"X\", column_mapping=column_mapping, quantile=0.8,\n",
        ")\n",
        "\n",
        "Xt, yt = gen.generate(X.copy(), y=None, selector=sel)\n",
        "print(f\"output shape: {X.shape} -> {Xt.shape}\")",
    ]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fired = sel(X, None)[\"X\"]\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))\n",
        "axes[0].scatter(X[:, 0], X[:, 1], c=np.where(fired, \"C3\", \"C0\"))\n",
        "axes[1].scatter(Xt[:, 0], Xt[:, 1], c=\"C0\")\n",
        "axes[0].set_title(\"Original (red = dropped by selector)\")\n",
        "axes[1].set_title(\"Transformed (fired rows removed)\")\n",
        "axes[0].set_xlabel(\"X\"); axes[0].set_ylabel(\"Y\")\n",
        "axes[1].set_xlabel(\"X\"); axes[1].set_ylabel(\"Y\")\n",
        "plt.tight_layout()",
    ]})

# ---- summary --------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Summary\n",
        "\n",
        "| Lever | Generator | Uses a DGP? | Effect on shape | Selection knob |",
        "|---|---|---|---|---|\n",
        "| observation process | `RandomSamplingClassesGenerator` | no | drops fired rows | `ClassSelector` (per-class shares) |",
        "| observation process | `RandomSamplingTargetsGenerator` | no | weighted resampling, size = input | `ProportionalSelector` (float scores) |",
        "| observation process | `RandomSamplingFeaturesGenerator` | no | resampling by keep-probability | `sampling_proba_func(X) -> per-row probabilities` |",
        "| DGP | `CausalImbalanceGenerator` | **yes** (fitted) | truncates to a quantile range / drops selector-fired rows | `quantile_range`, or explicit `selector` |",
        "\n",
        "* Imbalance is a *subsampling* deficit: `Xt` is smaller than `X`.\n",
        "* The selector (or `sampling_proba_func`) decides **who** is affected;\n",
        "  the generator realizes the decision.\n",
        "* Truncation keeps rows as-is — descendants are untouched, the subset\n",
        "  stays structurally consistent.\n",
    ]})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(SRC, "Imbalance.ipynb")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote Imbalance.ipynb cells: {len(cells)}")
