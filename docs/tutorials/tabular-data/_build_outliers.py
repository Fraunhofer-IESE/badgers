"""Builder: merge Outliers-Tabular-Data.ipynb + Causal-Outlier-Propagation.ipynb -> Outliers.ipynb.

Also fixes:
- broken double-assignment generate cells (`Xt, yt = Xt, _ = Xt, _ = ...`);
- visualization labels: explicit `cls` class vector + explicit `labels` outlier
  boolean vector (from get_outliers_labels()) — not y/yt placeholders;
- causal part shortened: drops the concept/contract sections that repeat the
  Understanding-badgers notebook, keeps chain + fork topologies and the
  sampling-strategy comparison.
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


a = load("Outliers-Tabular-Data.ipynb")
b = load("Causal-Outlier-Propagation.ipynb")


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def src_of(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


cells = []

# ---- intro ----------------------------------------------------------------
cells.append(md(
    "# Generating outliers\n",
    "\n",
    "**Outliers** are *appended* data points: the generator samples new\n",
    "rows that look extreme relative to your data, and stacks them **after**\n",
    "the original rows — so `Xt` is *larger* than `X`. Use outliers to test\n",
    "how pipelines behave when rare, extreme observations appear: robust\n",
    "statistics, anomaly detection, scaling, and model fitting all react\n",
    "very differently to them.\n",
    "\n",
    "This notebook shows the **two levers** badgers gives you:\n",
    "\n",
    "1. **Acting on the observation process** — sample outliers directly\n",
    "   from distributions fitted to your data (`ZScoreSamplingGenerator`,\n",
    "   `HypersphereSamplingGenerator`, histogram and low-density samplers).\n",
    "   These generators *do not use a DGP*.\n",
    "2. **Acting on the data-generating process (DGP)** — perturb a node of\n",
    "   a *fitted causal model*; the perturbation **propagates** to the\n",
    "   descendants, so the outliers are not just extreme — they are\n",
    "   *structurally consistent* extreme events\n",
    "   (`CausalOutlierPropagationGenerator`).\n",
    "\n",
    "Labels: the generated rows are the *last* `n_outliers` rows of `Xt`;\n",
    "the boolean array is exposed via `get_outliers_labels()` (labels do\n",
    "**not** ride in the deprecated `yt` return slot).\n",
))

# ---- setup ----------------------------------------------------------------
cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import networkx as nx\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "\n",
    "from numpy.random import default_rng\n",
    "from sklearn.datasets import make_blobs\n",
    "\n",
    "from badgers.generators.tabular_data.outliers.distribution_sampling import (\n",
    "    ZScoreSamplingGenerator, HypersphereSamplingGenerator,\n",
    ")\n",
    "from badgers.generators.tabular_data.outliers.low_density_sampling import (\n",
    "    IndependentHistogramsGenerator, HistogramSamplingGenerator, LowDensitySamplingGenerator,\n",
    ")\n",
    "from badgers.generators.tabular_data.outliers import StructuredOutlierGenerator\n",
    "\n",
    "rng = default_rng(seed=0)\n",
))

# plotting helper: explicit cls + labels vectors (not y/yt)
cells.append(code(
    "def plot_outliers(X, Xt, labels, cls=None):\n",
    '    """\n',
    "    Plot original data and the transformed data (outliers marked).\n",
    "    labels = boolean outlier labels (get_outliers_labels());\n",
    "    cls = optional class vector for coloring the original data.\n",
    '    """\n',
    "    outliers = Xt[labels]\n",
    "\n",
    "    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))\n",
    "    if cls is not None:\n",
    "        for i in range(len(np.unique(cls))):\n",
    "            ix = np.where(cls == i)\n",
    "            axes[0].scatter(X[ix,0],X[ix,1], c = f'C{i}', label = f'{i}')\n",
    "            axes[1].scatter(X[ix,0],X[ix,1], c = f'C{i}', label = f'{i}')\n",
    "    else:\n",
    "        axes[0].scatter(X[:,0], X[:,1], c='C0', label='original')\n",
    "        axes[1].scatter(X[:,0], X[:,1], c='C0', label='original')\n",
    "    axes[1].scatter(outliers[:,0],outliers[:,1], marker = 'x', c = f'black', label = 'outliers')\n",
    "    axes[0].set_title('Original')\n",
    "    axes[1].set_title('Transformed')\n",
    "    axes[1].legend(ncol=1, bbox_to_anchor=(1, 1))\n",
    "    plt.tight_layout()\n",
    "    return fig, axes\n",
))

# ---- data -----------------------------------------------------------------
cells.append(md(
    "## Load and prepare data\n",
    "\n",
    "We first load an existing dataset from `sklearn.datasets`.\n",
))
cells.append(code(
    "X, y = make_blobs(centers=4, random_state=0, cluster_std=0.5)\n",
    "cls = y.astype(int)   # explicit class vector for plotting\n",
    "print(X.shape)",
))

# ---- part A ---------------------------------------------------------------
cells.append(md(
    "## Part A — Acting on the observation process (no DGP)\n",
    "\n",
    "These generators sample outlier rows from distributions or densities\n",
    "derived directly from your data. No causal model is involved; the\n",
    "outliers are *marginal* extremes, unrelated to any structure between\n",
    "the columns.\n",
    "\n",
    "### Generate outliers using z-score sampling\n",
    "\n",
    "`ZScoreSamplingGenerator` generates points whose z-score (distance to\n",
    "the mean, in units of the standard deviation) exceeds 3 — far out on\n",
    "the tails of the empirical distribution.",
))

cells.append(code(
    "trf = ZScoreSamplingGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, n_outliers=25)   # y=None (deprecated slot)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X, Xt, labels, cls)",
))

cells.append(md(
    "### Generate outliers using hypersphere sampling\n",
    "\n",
    "`HypersphereSamplingGenerator` generates points uniformly on a\n",
    "hypersphere (a shell around your data) — extreme in every direction,\n",
    "not only along single axes.",
))
cells.append(code(
    "trf = HypersphereSamplingGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X, Xt, labels, cls)",
))

cells.append(md(
    "### Generate outliers using independent histogram sampling\n",
    "\n",
    "`IndependentHistogramsGenerator` fits a histogram to each column\n",
    "independently, then samples outliers from the empty (or rare) bins —\n",
    "combinations that never occur in the data.",
))
cells.append(code(
    "trf = IndependentHistogramsGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X, Xt, labels, cls)",
))

cells.append(md(
    "### Generate outliers using histogram sampling\n",
    "\n",
    "`HistogramSamplingGenerator` works on the *joint* histogram of the\n",
    "data — but only for low-dimensional datasets (the bin count explodes\n",
    "with dimensionality).",
))
cells.append(code(
    "trf = HistogramSamplingGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X, Xt, labels, cls)",
))

cells.append(md(
    "### Generate outliers using low density sampling\n",
    "\n",
    "`LowDensitySamplingGenerator` samples points in the *low-density*\n",
    "regions of the empirical distribution — between and beyond the\n",
    "clusters, rather than at the extremes of single columns.",
))
cells.append(code(
    "trf = LowDensitySamplingGenerator(random_generator=rng)\n",
    "Xt, _ = trf.generate(X.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X, Xt, labels, cls)",
))

# ---- structured outliers --------------------------------------------------
cells.append(md(
    "### Generate outliers in a reduced space (decomposition + sampling)\n",
    "\n",
    "For higher-dimensional data you can first **reduce the dimensionality**\n",
    "(any sklearn transformer with `transform`/`inverse_transform`, e.g.\n",
    "`PCA`, `KernelPCA`, `FastICA`) and sample outliers in the reduced\n",
    "space: `StructuredOutlierGenerator` chains a decomposition and an\n",
    "outlier sampler — outliers are sampled where the data *really* has\n",
    "little support, in the informative subspace rather than the noisy one.",
))
cells.append(code(
    "from sklearn.decomposition import PCA, KernelPCA\n",
    "\n",
    "# generate some data with higher dimensionality\n",
    "X_hi, y_hi = make_blobs(n_features=10, centers=4, cluster_std=0.5, random_state=0)\n",
    "cls_hi = y_hi.astype(int)\n",
    "\n",
    "trf = StructuredOutlierGenerator(\n",
    "    dimension_reduction_transformer=PCA(n_components=3),\n",
    "    outlier_generator=ZScoreSamplingGenerator(),\n",
    ")\n",
    "Xt, _ = trf.generate(X_hi.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X_hi.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X_hi, Xt, labels, cls_hi)",
))

cells.append(md(
    "Any decomposition with `inverse_transform` works, e.g. a kernel PCA —\n",
    "and the outlier sampler can be any sampler from this section:",
))
cells.append(code(
    "trf = StructuredOutlierGenerator(\n",
    "    dimension_reduction_transformer=KernelPCA(n_components=3, kernel=\"rbf\", fit_inverse_transform=True),\n",
    "    outlier_generator=HypersphereSamplingGenerator(),\n",
    ")\n",
    "Xt, _ = trf.generate(X_hi.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X_hi.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))",
))
cells.append(code(
    "fig, axes = plot_outliers(X_hi, Xt, labels, cls_hi)",
))

# custom preprocessor demo
cells.append(md(
    "Preprocessing stays out of the generator's responsibility: instead of\n",
    "the built-in standardization + decomposition you can pass a custom\n",
    "`preprocessor` (anything with `transform`/`inverse_transform`, e.g. a\n",
    "fitted sklearn pipeline):\n",
))
cells.append(code(
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.preprocessing import RobustScaler\n",
    "from sklearn.decomposition import FastICA\n",
    "\n",
    "trf = StructuredOutlierGenerator(\n",
    "    outlier_generator=ZScoreSamplingGenerator(),\n",
    "    preprocessor=make_pipeline(RobustScaler(), FastICA(n_components=3, whiten=\"unit-variance\")),\n",
    ")\n",
    "Xt, _ = trf.generate(X_hi.copy(), None, n_outliers=25)\n",
    "labels = trf.get_outliers_labels()\n",
    "print('appending:', X_hi.shape, '->', Xt.shape, '| outlier rows:', int(labels.sum()))\n",
))
cells.append(code(
    "fig, axes = plot_outliers(X_hi, Xt, labels, cls_hi)",
))

# ---- part B ---------------------------------------------------------------
cells.append(md(
    "## Part B — Acting on the data-generating process (DGP)\n",
    "\n",
    "`CausalOutlierPropagationGenerator` perturbs a node of a *fitted causal\n",
    "model* with `do()`-style intervention: the node is set to an extreme\n",
    "value and the **descendants are recomputed** through the structural\n",
    "equations. The result: outliers that are extreme *and structurally\n",
    "consistent* — unlike Part A's marginal extremes, these respect the\n",
    "relationships between your columns.\n",
    "\n",
    "**Which node you perturb matters**, and the topology decides how far\n",
    "the effect reaches — that is what the following examples show.\n",
))

cells.append(b[8])    # imports incl. plot helpers of the causal nb
cells.append(b[9])    # topology helpers
cells.append(b[10])   # plot_data helper
cells.append(b[11])   # draw_graph helper
cells.append(b[12])   # shared RNG

cells.append(md(
    "### Chain: $X \\rightarrow Y \\rightarrow Z$\n",
    "\n",
    "**Perturbing X** affects Y (direct child) and Z (indirect child).\n",
    "**Perturbing Y** affects Z but *not* X. The propagation direction is\n",
    "always downstream.",
))
cells.append(b[14])   # build chain + data
cells.append(md(
    "#### Visualize graph and data",
))
cells.append(b[16])
cells.append(md(
    "#### Generate outliers (perturb X, then Y)",
))
cells.append(b[18])  # imports: OlsDGP + CausalOutlierPropagationGenerator
cells.append(b[19])   # do(X) + do(Y) generation
cells.append(md(
    "#### Visualize generated data",
))
cells.append(b[21])
cells.append(b[22])

cells.append(md(
    "### Fork (common cause): $C \\rightarrow X, C \\rightarrow Y$\n",
    "\n",
    "**Perturbing the common cause C** moves *both* children X and Y — the\n",
    "outliers appear correlated even though the perturbation hit a single\n",
    "node.",
))
cells.append(b[24])   # build fork + data
cells.append(md(
    "#### Visualize graph and data",
))
cells.append(b[26])
cells.append(md(
    "#### Generate outliers",
))
cells.append(b[28])
cells.append(md(
    "#### Visualize generated data",
))
cells.append(b[30])

cells.append(md(
    "### Sampling strategy comparison\n",
    "\n",
    "The `within_distribution_sampler` and `out_of_distribution_sampler`\n",
    "knobs control *how* the outlier values are drawn at the perturbation\n",
    "node:",
))
cells.append(b[48])
cells.append(b[49])

# ---- summary --------------------------------------------------------------
cells.append(md(
    "## Summary\n",
    "\n",
    "| Lever | Generator | Uses a DGP? | Propagates? | Effect on shape |",
    "|---|---|---|---|---|\n",
    "| observation process | `ZScoreSamplingGenerator` | no | no | appends `n_outliers` rows |",
    "| observation process | `HypersphereSamplingGenerator` | no | no | appends `n_outliers` rows |",
    "| observation process | `IndependentHistogramsGenerator` / `HistogramSamplingGenerator` / `LowDensitySamplingGenerator` | no | no | appends `n_outliers` rows |",
    "| observation process + decomposition | `StructuredOutlierGenerator` | no | no | appends (sampling happens in the reduced space) |",
    "| DGP | `CausalOutlierPropagationGenerator` | **yes** (fitted) | **yes** — descendants re-evaluated | appends perturbed + propagated rows |",
    "\n",
    "* Outliers are an *appending* deficit: `Xt` is larger than `X`, and the\n",
    "  generated rows are the **last** `n_outliers` rows.\n",
    "* Outlier labels come from `get_outliers_labels()` — they do not ride\n",
    "  in the deprecated `yt` slot.\n",
    "* Part A's outliers are marginal extremes; Part B's outliers are\n",
    "  structurally consistent extreme events whose footprint depends on\n",
    "  the causal topology (chain / fork / collider / diamond).\n",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(SRC, "Outliers.ipynb")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote Outliers.ipynb cells: {len(cells)}")
