"""Builder: merge Drift-Tabular-Data.ipynb + Causal-Drift.ipynb -> Drift.ipynb.

Pilot for the per-deficit merge. Structure:
  - intro: what drift is, the two levers (observation process vs DGP)
  - Part A: acting on the observation process (no DGP)
  - Part B: acting on the data-generating process (causal propagation)
  - drift_std semantics clarified per the normalize -> shift -> un-normalize formula
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


def join_md(cell):
    """Return markdown source as a list of lines without trailing newlines."""
    return [l.rstrip("\n") for l in (cell["source"] if isinstance(cell["source"], list) else cell["source"].split("\n"))]


a = load("Drift-Tabular-Data.ipynb")   # non-causal
b = load("Causal-Drift.ipynb")         # causal

cells = []

# ---- intro ----------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# Generating drift\n",
        "\n",
        "**Drift** is a systematic shift of the data distribution: the same\n",
        "data-generating process starts producing values that are translated\n",
        "(or scaled) compared to what you saw before. It is one of the most\n",
        "common defects in production ML systems — a sensor slowly degrades, a\n",
        "supplier changes units, an upstream model update shifts an input.\n",
        "\n",
        "This notebook shows the **two levers** badgers gives you to inject\n",
        "drift:\n",
        "\n",
        "1. **Acting on the observation process** — directly translate the\n",
        "   observed values of one or several columns. No causal model is\n",
        "   involved; these generators (`RandomShiftGenerator`,\n",
        "   `RandomShiftClassesGenerator`) *do not use a DGP* and do not\n",
        "   propagate anything: the change is exactly the shift you ask for.\n",
        "2. **Acting on the data-generating process (DGP)** — perturb a node of\n",
        "   a *fitted causal model* and let the shift **propagate** to the\n",
        "   descendants through the structural equations\n",
        "   (`CausalDriftGenerator`).\n",
        "\n",
        "Both accept a **selector** to decide which rows are affected\n",
        "(see the *Understanding badgers* notebook for the selector contract).\n",
    ]})

# ---- setup ----------------------------------------------------------------
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "from numpy.random import default_rng\n",
        "from sklearn.datasets import make_blobs\n",
        "\n",
        "from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator\n",
        "\n",
        "rng = default_rng(seed=0)\n",
    ]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "X, y = make_blobs(centers=4, random_state=0)\n",
        "# classes are a column of X (y is deprecated as a data carrier)\n",
        "Xc = np.column_stack([X, y])\n",
        "column_mapping = {'f0': 0, 'f1': 1, 'cls': 2}\n",
        "cls = Xc[:, column_mapping['cls']]   # explicit class vector for plotting\n",
        "print(X.shape, '->', Xc.shape)",
    ]})

# ---- part A ---------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part A — Acting on the observation process (no DGP)\n",
        "\n",
        "**What the generators do:** they standardize `X` (each column to mean\n",
        "0, variance 1), add a random shift, and un-standardize. The result is\n",
        "a translation of the observed values only — no causal structure is\n",
        "used, and nothing propagates beyond the shifted columns.\n",
        "\n",
        "### Translate all rows (every value of the selected columns)\n",
    ]})

cells.append(a[8])   # RandomShiftGenerator.generate(...)
cells.append(a[9])   # scatter before/after

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### Translate per class\n",
        "\n",
        "The class values are read from a column of `X` (here the last column,\n",
        "`cls`). A `ClassSelector` builds one boolean mask per class value:\n",
        "each class gets its own independent shift.\n",
    ]})

# fix: y -> cls in the class demo cells
for idx in (11, 13):
    cell = a[idx]
    src = "".join(cell["source"])
    src = src.replace("hue=y", "hue=cls").replace("hue=Xc[:, 2]", "hue=cls")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})
for idx in (12, 14):
    cell = a[idx]
    src = "".join(cell["source"])
    src = src.replace("hue=y", "hue=cls").replace("hue=Xc[:, 2]", "hue=cls")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

# ---- drift_std semantics --------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### What `shift_std` / `drift_std` actually mean\n",
        "\n",
        "Both parameters express the shift **in units of the column's standard\n",
        "deviation** (`sigma`). Internally every drift generator does:\n",
        "\n",
        "```\n",
        "z = (X - mean) / std          # normalize: each column has sigma = 1\n",
        "z += shift                    # shift drawn from N(0, shift_std * sigma)\n",
        "X' = z * std + mean           # un-normalize\n",
        "```\n",
        "\n",
        "So `shift_std = 2` on column `f0` means exactly: *\"move column `f0` by\n",
        "2 of its own standard deviations\"* — a column with a wide range gets a\n",
        "proportionally larger absolute shift than a narrow one. The drawn\n",
        "shift is a single value per column, applied to **all** affected rows\n",
        "(that is what makes it *drift* and not *noise* — noise redraws per\n",
        "row).\n",
    ]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "# verify the semantics: the shift DRAWN for f0, expressed in sigma units,\n",
        "# equals a draw from N(0, shift_std) -- i.e. shift_std sets the scale of the\n",
        "# shift in standard deviations, it is not the shift itself.\n",
        "X_test = X[:, :2].astype(float).copy()\n",
        "gen = RandomShiftGenerator(random_generator=default_rng(42))\n",
        "Xt_test, _ = gen.generate(X_test, None, shift_std=np.array([2.0, 0.0]))\n",
        "\n",
        "sigma0 = X_test[:, 0].std()                 # sigma of the ORIGINAL column\n",
        "observed_z = (Xt_test[:, 0] - X_test[:, 0]).mean() / sigma0\n",
        "expected_z = default_rng(42).normal(0., 2.0, size=2)[0]  # the same draw\n",
        "print(f'shift of f0 in sigma units: {observed_z:.4f}')\n",
        "print(f'draw from N(0, shift_std=2):  {expected_z:.4f}')\n",
        "print('match:', np.isclose(observed_z, expected_z))\n",
        "print('f1 untouched:', np.allclose(Xt_test[:, 1], X_test[:, 1]))",
    ]})

# ---- part B ---------------------------------------------------------------
cells.append({    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from badgers.generators.tabular_data.drift.causal import CausalDriftGenerator\n",
        "from badgers.core.dgp import OlsDGP\n",
    ]})

cells.append({    "cell_type": "markdown", "metadata": {}, "source": [
        "## Part B — Acting on the data-generating process (DGP)\n",
        "\n",
        "**What the generator does:** `CausalDriftGenerator` takes a *fitted\n",
        "DGP* (a causal model of your data). It shifts a node of your choice,\n",
        "then **re-evaluates the structural equations of the descendants** on\n",
        "the shifted rows (Pearl surgery): the change propagates exactly as it\n",
        "would in the real system. Only downstream nodes are affected; upstream\n",
        "nodes and untouched rows stay as they were.\n",
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
cells.append(b[5])   # build + draw chain

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 2. Generate data from the DAG\n",
        "\n",
        "Each node is a weighted sum of its parents plus noise, so the data\n",
        "already carries the causal structure the generator will exploit.\n",
    ]})
cells.append(b[7])   # generate linear data
cells.append(b[8])   # pairplot

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 3. Fit the DGP\n",
        "\n",
        "The default backend `OlsDGP` is a deterministic per-node least-squares\n",
        "backend (pure NumPy): for every non-root node it learns the linear\n",
        "coefficients from its parents and the residual noise, fitted on the\n",
        "*original* data.",
    ]})
cells.append(b[10])  # OlsDGP fit + generator construction

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "### 4. Generate causally-propagated drift\n",
        "\n",
        "We shift node `Y` and watch it propagate to `Z` (but not to `X`).\n",
    ]})
cells.append(b[12])  # generate
cells.append(b[13])  # plot X-Y panel
cells.append(b[14])  # plot Y-Z panel

cells.append(b[15])  # propagation explanation markdown

# ---- summary --------------------------------------------------------------
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Summary\n",
        "\n",
        "| Lever | Generator | Uses a DGP? | Propagates? | Selector support |",
        "|---|---|---|---|---|\n",
        "| observation process | `RandomShiftGenerator` | no | no (values only) | — |",
        "| observation process, per class | `RandomShiftClassesGenerator` | no | no (values only) | `ClassSelector` |",
        "| DGP | `CausalDriftGenerator` | **yes** (fitted) | **yes** — descendants re-evaluated | yes (boolean masks) |",
        "\n",
        "* Drift is a *single* systematic shift per column, drawn once and\n",
        "  applied to all affected rows — in contrast to noise, which redraws\n",
        "  per row.\n",
        "* `shift_std` / `drift_std` are in units of the column's standard\n",
        "  deviation (normalize, shift, un-normalize).\n",
        "* Causal drift mutates `X` in place — pass `X.copy()` if you need the\n",
        "  original.\n",
    ]})

# ---- write ----------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(SRC, "Drift.ipynb")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote Drift.ipynb cells: {len(cells)}")
