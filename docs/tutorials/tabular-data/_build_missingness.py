"""Builder: merge Missingness-Tabular-Data.ipynb + Causal-Missingness.ipynb -> Missingness.ipynb.

Also fixes:
- MCAR/MAR/MNAR defined once in the intro; classic generators = legacy one-knob
  shortcuts; causal generator = the explicit selector-based version of the same
  three mechanisms.
- visualization labels: hue from an explicit `cls` vector, not y.
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


a = load("Missingness-Tabular-Data.ipynb")
b = load("Causal-Missingness.ipynb")


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def src_of(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


cells = []

# ---- intro ----------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# Generating missing data\n",
        "\n",
        "**Missingness** is the defect where some observed values are simply\n",
        "not recorded (`np.nan` in the output). Real pipelines rarely get to\n",
        "see complete data: sensors drop out, forms are left blank, values\n",
        "exceed a recording range. The *mechanism* matters as much as the\n",
        "amount — the same 20% of missing values behaves very differently\n",
        "depending on **why** the values are missing:\n",
        "\n",
        "| Mechanism | Meaning | In badgers (selector) | Legacy one-knob |",
        "|---|---|---|---|\n",
        "| **MCAR** — missing completely at random | a value goes missing independently of everything else | `BernoulliSelector(rate=...)` | `MissingCompletelyAtRandom` |",
        "| **MAR** — missing at random | missingness depends on **another, observed** variable (e.g. values of `X` are high) | `TopQuantileSelector(..., drivers=...)` | `DummyMissingAtRandom` |",
        "| **MNAR** — missing not at random | a value's **own** (unobserved) magnitude drives its missingness | `TopQuantileSelector` on itself | `DummyMissingNotAtRandom` |",
        "\n",
        "The notebook shows both ways of expressing these mechanisms:\n",
        "\n",
        "1. **Acting on the observation process** — the legacy one-knob\n",
        "   generators: `percentage_missing` decides *how much* goes missing,\n",
        "   the generator class decides *why*. They *do not use a DGP* and\n",
        "   nothing is propagated.\n",
        "2. **Explicit selectors** — the same three mechanisms expressed as\n",
        "   selector objects, which decide *which cells* fire. Intensity\n",
        "   (`rate` / `quantile`) lives on the selector, not on `generate()`.\n",
    ]})

# ---- setup ----------------------------------------------------------------
cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from numpy.random import default_rng\n",
    "from sklearn.datasets import make_blobs\n",
    "\n",
    "from badgers.generators.tabular_data.missingness import (\n",
    "    MissingCompletelyAtRandom, DummyMissingAtRandom, DummyMissingNotAtRandom,\n",
    ")\n",
    "from badgers.generators.tabular_data.missingness.causal import CausalMissingnessGenerator\n",
    "from badgers.core.dgp import OlsDGP\n",
    "from badgers.core.selectors import BernoulliSelector, TopQuantileSelector\n",
    "\n",
    "rng = default_rng(seed=0)\n",
))

# plotting helper: class vector, not y
cells.append(code(
    "def plot_missing(X, cls, Xt):\n",
    '    """\n',
    "    Some utility function to generate the plots\n",
    '    """\n',
    "    missing_mask = np.isnan(Xt).any(axis=1)\n",
    "    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))\n",
    "    for label in np.unique(cls):\n",
    "        ix = np.where(cls == label)\n",
    "        axes[0].scatter(X[ix,0],X[ix,1], c = f'C{label}', label = f'{label}')\n",
    "        ix = np.where(cls[~missing_mask] == label )\n",
    "        axes[1].scatter(Xt[~missing_mask][ix,0], Xt[~missing_mask][ix,1], c = f'C{label}', label = f'{label}')\n",
    "    # plot missing values\n",
    "    axes[1].scatter(X[missing_mask][:,0],X[missing_mask][:,1],marker='x', color='black', label = 'missing')\n",
    "    axes[0].set_title('Original')\n",
    "    axes[1].set_title('Transformed')\n",
    "    axes[0].legend()\n",
    "    axes[1].legend()\n",
    "    return fig, axes\n",
))

# ---- data -----------------------------------------------------------------
cells.append(code(
    "X, y = make_blobs(centers=4, random_state=0)\n",
    "# classes are a column of X (y is deprecated as a data carrier)\n",
    "Xc = np.column_stack([X, y])\n",
    "cls = Xc[:, 2].astype(int)   # explicit class vector for plotting\n",
    "print(X.shape)",
))

# ---- part A ---------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part A — Acting on the observation process (legacy one-knob, no DGP)\n",
        "\n",
        "These generators take a single `percentage_missing` value: *how much*\n",
        "goes missing. *Why* it goes missing is hard-coded by the class\n",
        "(MCAR / MAR / MNAR). No causal model is involved and no propagation\n",
        "happens.\n",
        "\n",
        "### Missing completely at random (MCAR)\n",
        "\n",
        "`MissingCompletelyAtRandom` replaces values with `np.nan`\n",
        "independently at random — the fact that a value is missing carries no\n",
        "information.",
    ]})

cells.append(code(
    "trf = MissingCompletelyAtRandom(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, percentage_missing=0.2)   # y=None (deprecated slot)\n",
    "print(f'rows with a NaN: {np.isnan(Xt).any(axis=1).sum()} / {len(Xt)}')",
))
cells.append(code(
    "fig, axes = plot_missing(X, cls, Xt)\n",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Missing at random (MAR)\n",
        "\n",
        "`DummyMissingAtRandom` makes missingness depend on **another,\n",
        "observed** feature: the probability that a value goes missing varies\n",
        "linearly with another column. Looking only at observed data you can\n",
        "partly *explain* the missingness (it correlates with the driver), but\n",
        "you cannot fully predict it.",
    ]})

cells.append(code(
    "trf = DummyMissingAtRandom(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, percentage_missing=0.25)\n",
    "print(f'rows with a NaN: {np.isnan(Xt).any(axis=1).sum()} / {len(Xt)}')",
))
cells.append(code(
    "fig, axes = plot_missing(X, cls, Xt)\n",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Missing not at random (MNAR)\n",
        "\n",
        "`DummyMissingNotAtRandom` makes a value's **own** magnitude drive its\n",
        "missingness: the larger the original value, the more likely it is\n",
        "missing. Here the missingness *biases* the observed data — e.g. peak\n",
        "values systematically vanish.",
    ]})

cells.append(code(
    "trf = DummyMissingNotAtRandom(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, percentage_missing=0.25)\n",
    "print(f'rows with a NaN: {np.isnan(Xt).any(axis=1).sum()} / {len(Xt)}')",
))
cells.append(code(
    "fig, axes = plot_missing(X, cls, Xt)\n",
))

# ---- part B ---------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part B — Explicit selectors on a causal graph\n",
        "\n",
        "`CausalMissingnessGenerator` expresses the **same three mechanisms**\n",
        "with selector objects on a fitted DGP. It writes `NaN` at the fired\n",
        "cells of the perturbation nodes and does **not** propagate: missingness\n",
        "is a property of the *measurement* process, not of the data-generating\n",
        "process (a descendant can still be observed).",
    ]})

cells.append(b[2])   # imports incl. helpers
cells.append(b[3])   # generate_linear_data helper

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Build the graph, generate data, fit the DGP\n",
    ]})
cells.append(b[5])
cells.append(b[7])
cells.append(b[10])  # OlsDGP fit + generator construction

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### MCAR: a value-blind selector\n",
        "\n",
        "`BernoulliSelector(node, rate=...)` — each row fires independently\n",
        "with probability `rate`.\n",
    ]})
cells.append(b[12])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### MNAR: values drive their own missingness\n",
        "\n",
        "With the *self*-mode `TopQuantileSelector`, the **largest values of Y\n",
        "itself** are the ones that go missing (here the top 30%):",
    ]})
cells.append(b[16])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### MAR: a fully-observed driver\n",
        "\n",
        "With *driver*-mode `TopQuantileSelector`, missingness of **Y** is\n",
        "driven by the observed value of **X** (the top half of X values):",
    ]})
cells.append(b[18])

# ---- summary --------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Summary\n",
        "\n",
        "| Lever | Generator | Uses a DGP? | Propagates? | Mechanism knob |",
        "|---|---|---|---|---|\n",
        "| observation process (legacy) | `MissingCompletelyAtRandom` | no | no | class = MCAR, `percentage_missing` = amount |",
        "| observation process (legacy) | `DummyMissingAtRandom` | no | no | class = MAR, `percentage_missing` = amount |",
        "| observation process (legacy) | `DummyMissingNotAtRandom` | no | no | class = MNAR, `percentage_missing` = amount |",
        "| selectors on a causal graph | `CausalMissingnessGenerator` | **yes** (fitted) | **no** (observation layer) | selector object = mechanism **and** intensity |",
        "\n",
        "* With the selector version, intensity (`rate` / `quantile`) is\n",
        "  configured on the selector, not on `generate()`.\n",
        "* Missingness never cascades: it is a property of the measurement,\n",
        "  not of the data-generating process.\n",
        "* Generators mutate `X` in place — pass `X.copy()` if you need the\n",
        "  original.\n",
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
out = os.path.join(SRC, "Missingness.ipynb")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote Missingness.ipynb cells: {len(cells)}")
