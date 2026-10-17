"""Builder: merge Noise-Tabular-Data.ipynb + Causal-Noise.ipynb -> Noise.ipynb.

Also fixes:
- visualization labels: hue from an explicit `cls` vector, not y/yt (labels don't ride in yt)
- ClassSelector explanation: selector decides WHICH rows, noise_std_per_class decides HOW STRONG
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


a = load("Noise-Tabular-Data.ipynb")
b = load("Causal-Noise.ipynb")


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def src_of(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


cells = []

# ---- intro ----------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# Generating noise\n",
        "\n",
        "**Noise** is a *row-wise* perturbation of the observed values: each\n",
        "affected row gets its own random deviation, drawn independently\n",
        "(in contrast to drift, where one systematic shift is applied to all\n",
        "affected rows). Use noise to make a pipeline robust to measurement\n",
        "error, sensor jitter, or rounding — the *signal* is still there, but\n",
        "each observation is a bit off.\n",
        "\n",
        "This notebook shows the **two levers** badgers gives you:\n",
        "\n",
        "1. **Acting on the observation process** — add noise directly to the\n",
        "   observed values (`AdditiveNoiseGenerator`,\n",
        "   `GaussianNoiseClassesGenerator`). These generators *do not use a\n",
        "   DGP* and nothing propagates: the value is changed exactly where\n",
        "   the noise lands.\n",
        "2. **Acting on the data-generating process (DGP)** — inject noise at a\n",
        "   node of a *fitted causal model* and let it **propagate** to the\n",
        "   descendants (`CausalNoiseGenerator`).\n",
        "\n",
        "Both accept a **selector** to decide which rows are affected\n",
        "(see the *Understanding badgers* notebook for the selector contract).\n",
    ]})

# ---- setup ----------------------------------------------------------------
cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from numpy.random import default_rng\n",
    "from scipy.stats import norm\n",
    "from sklearn.datasets import make_blobs\n",
    "\n",
    "from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator, GaussianNoiseClassesGenerator\n",
    "\n",
    "rng = default_rng(seed=0)\n",
))

# plotting helper: hue comes from an explicit class vector, not from yt
cells.append(code(
    "def plot_noise(X, cls, Xt):\n",
    '    """\n',
    "    Some utility function to generate the plots\n",
    '    """\n',
    "    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))\n",
    "    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cls, ax=axes[0])\n",
    "    sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], hue=cls, ax=axes[1])\n",
    "    axes[0].set_title('Original')\n",
    "    axes[1].set_title('Transformed')\n",
    "    return fig, axes\n",
))

# ---- data -----------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Load and prepare data\n",
        "\n",
        "We first load an existing dataset from `sklearn.datasets`. The class\n",
        "values are stored as a **column of `X`** (the `y` argument is\n",
        "deprecated as a data carrier), so plotting uses the explicit vector\n",
        "`cls`.\n",
    ]})

cells.append(code(
    "X, y = make_blobs(centers=4, random_state=0, cluster_std=0.25)\n",
    "# classes are a column of X (y is deprecated as a data carrier)\n",
    "Xc = np.column_stack([X, y])\n",
    "column_mapping = {'f0': 0, 'f1': 1, 'cls': 2}\n",
    "cls = Xc[:, column_mapping['cls']]   # explicit class vector for plotting\n",
    "print(X.shape, '->', Xc.shape)",
))

# ---- part A ---------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part A — Acting on the observation process (no DGP)\n",
        "\n",
        "### Add noise to all values\n",
        "\n",
        "`AdditiveNoiseGenerator` adds a draw from `distribution` to **every\n",
        "cell** of `X` — each row gets its own independent deviation.\n",
    ]})

cells.append(code(
    "trf = AdditiveNoiseGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, distribution=norm(scale=0.25))   # y=None (deprecated slot)\n",
    "print('shapes equal (gating):', Xt.shape == X.shape)",
))
cells.append(code(
    "fig, axes = plot_noise(X, cls, Xt)\n",
))

# ---- per-class noise + ClassSelector explanation --------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Add noise per class\n",
        "\n",
        "`GaussianNoiseClassesGenerator` combines **two independent knobs**\n",
        "that are easy to confuse:\n",
        "\n",
        "* the **selector** decides *which rows* receive noise. It returns one\n",
        "  boolean mask (or firing probability) per class value. In the demo\n",
        "  below, `{'cls': {0: 1., 1: 1., 2: 1., 3: 1.}}` means: *every row of\n",
        "  every class fires with probability 1* — so all rows are noisy. Set\n",
        "  a value to `0.5` and half of that class's rows would be noisy\n",
        "  (Bernoulli-wise); set it to `0.` and that class stays clean.\n",
        "* `noise_std_per_class` decides *how strong* the noise is **on the\n",
        "  fired rows**, per class: `{0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}` makes\n",
        "  class 3 rows four times noisier than class 0 rows. It has **no**\n",
        "  effect on *which* rows are noisy.\n",
        "\n",
        "In short: **selector = who is affected, `noise_std_per_class` = how\n",
        "badly.**",
    ]})

cells.append(code(
    "from badgers.core.selectors import ClassSelector\n",
    "\n",
    "# WHO is affected: every row of every class fires (probability 1.0)\n",
    "selector = ClassSelector('cls', column_mapping, {'cls': {0: 1., 1: 1., 2: 1., 3: 1.}}, rng=rng)\n",
    "\n",
    "# HOW BADLY: noise std per class value (only applies to fired rows)\n",
    "trf = GaussianNoiseClassesGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(Xc.copy(), None, selector=selector,\n",
    "                     noise_std_per_class={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})\n",
    "print('shapes equal (gating):', Xt.shape == Xc.shape)",
))
cells.append(code(
    "fig, axes = plot_noise(Xc, cls, Xt)\n",
))

# ---- part B ---------------------------------------------------------------
cells.append(code(
    "from badgers.generators.tabular_data.noise.causal import CausalNoiseGenerator\n",
    "from badgers.core.dgp import OlsDGP\n",
))

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part B — Acting on the data-generating process (DGP)\n",
        "\n",
        "`CausalNoiseGenerator` injects noise at a node of a *fitted causal\n",
        "model*. The noise perturbs the node's value, and the change\n",
        "**cascades to the descendants** through the structural equations:\n",
        "descendant = f(new parent values) + fresh noise. Only downstream nodes\n",
        "are affected; upstream nodes and untouched rows stay as they were.\n",
    ]})

cells.append(b[2])   # imports incl. helpers
cells.append(b[3])   # generate_linear_data helper

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 1. Build the causal graph\n",
        "\n",
        "We use a three-node chain `X -> Y -> Z`. The node names are symbolic;\n",
        "a `column_mapping` maps them to column positions of `X`.\n",
    ]})
cells.append(b[5])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 2. Generate data from the DAG\n",
    ]})
cells.append(b[7])
cells.append(b[8])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 3. Fit the DGP\n",
        "\n",
        "The default backend `OlsDGP` is a deterministic per-node least-squares\n",
        "backend (pure NumPy): for every non-root node it learns the linear\n",
        "coefficients from its parents and the residual noise, fitted on the\n",
        "*original* data.",
    ]})
cells.append(b[9])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 4. Inject causally-propagated noise\n",
        "\n",
        "We perturb node `Y` and watch the noise propagate to `Z` (but not to\n",
        "`X`).",
    ]})
cells.append(b[10])
cells.append(b[12])
cells.append(b[14])

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Expanding rows (`n_copies`)\n",
        "\n",
        "Set `n_copies > 1` to *expand* the data: each original row produces\n",
        "`n_copies` noisy copies. Shape grows (appending mode).",
    ]})
cells.append(b[16])

# ---- summary --------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Summary\n",
        "\n",
        "| Lever | Generator | Uses a DGP? | Propagates? | Selector support |",
        "|---|---|---|---|---|\n",
        "| observation process | `AdditiveNoiseGenerator` | no | no (values only) | — |",
        "| observation process, per class | `GaussianNoiseClassesGenerator` | no | no (values only) | `ClassSelector` |",
        "| DGP | `CausalNoiseGenerator` | **yes** (fitted) | **yes** — descendants re-evaluated | yes |",
        "\n",
        "* Noise is *row-wise*: each affected row draws its own deviation —\n",
        "  in contrast to drift, where one systematic shift hits all rows.\n",
        "* **selector = who is affected** (masks/probabilities per class),\n",
        "  **`noise_std_per_class` = how strongly** (magnitude per class).\n",
        "* Causal noise mutates `X` in place — pass `X.copy()` if you need the\n",
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
out = os.path.join(SRC, "Noise.ipynb")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote Noise.ipynb cells: {len(cells)}")
