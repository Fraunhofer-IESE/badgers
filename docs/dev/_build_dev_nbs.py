"""One-shot builder for the two dev notebooks (selectors + DGP backends)."""
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


# =====================================================================
# dev/Create-New-Selectors.ipynb
# =====================================================================
cells = []

cells.append(md(
    "# Creating selectors",
    "",
    "This notebook explains the **selector contract** in full detail and",
    "shows how to implement, validate and consume a custom selector.",
    "",
    "Selectors are the objects that decide **which rows of the data receive",
    "a deficit** (outlier / missing value / noise / drift / drop). A",
    "generator that accepts a `selector` delegates the *who* to the selector",
    "and keeps the *what* to itself.",
))

cells.append(md(
    "## The contract",
    "",
    "A selector is any callable with the signature:",
    "",
    "```python",
    "selector(X, y) -> dict[str, array]",
    "```",
    "",
    "with, per **variable** (usually a column of `X`, resolved via",
    "`column_mapping`):",
    "",
    "* a **boolean mask** (`True` = deficit applied to that row), or",
    "* a **float score in [0, 1]** — the probability that the deficit is",
    "  applied (1.0 = definitely, 0.0 = never). Boolean masks are simply the",
    "  degenerate deterministic case of that contract.",
    "",
    "### Rules every selector must respect",
    "",
    "1. **No randomness for rates.** The selector must never draw from its",
    "   rng to decide *whether* a row fires; if it needs randomness at all",
    "   (e.g. a Bernoulli *preset* that pre-realizes), that must be",
    "   explicit and documented. Scores are proposals, not realizations.",
    "2. **Pure function of the inputs** (plus constructor-time intensity):",
    "   calling the selector twice on the same `X` must return the same",
    "   values.",
    "3. **y is a first-class argument** but deprecated as a data carrier:",
    "   prefer reading class/trigger columns from `X` via `column_mapping`.",
    "4. **Validate early**: raise a clear error in the constructor for",
    "   unknown variables/columns.",
    "5. **Optional `stats` support**: some consumers pass",
    "   `stats=...` (row-level summary statistics) so selectors can fire",
    "   based on statistics rather than raw values. Accept a `stats=None`",
    "   keyword only if you can use it; otherwise don't declare it.",
    "",
    "### Who realizes the scores?",
    "",
    "Realization belongs to the **consuming generator**:",
    "",
    "* *gating* consumers (noise, drift, missingness) preserve shape and",
    "  fire per-row Bernoulli-wise on fractional scores;",
    "* *subsampling* consumers (imbalance) drop fired rows — fractional",
    "  scores become selection weights (`1 - score` = keep probability).",
))

cells.append(code(
    "import numpy as np",
    "from numpy.random import default_rng",
    "",
    "from badgers.core.selectors import AlwaysSelector, BernoulliSelector, ProportionalSelector",
    "",
    "rng = default_rng(0)",
    "X = rng.normal(size=(100, 2))",
    "",
    "# the three contract shapes in action",
    "print('AlwaysSelector  :', AlwaysSelector(['f0'])(X, None))",
    "print('BernoulliSelector mask (pre-realized, first 5):', BernoulliSelector(['f0'], rate=0.3, rng=rng)(X, None)['f0'][:5])",
    "print('ProportionalSelector:', ProportionalSelector(['f0'], column_mapping={'f0': 0})(X, None)['f0'][:5])",
))

cells.append(md(
    "## Building a custom selector",
    "",
    "The minimal recipe — a class with `__call__(self, X, y, **kw)` returning",
    "a `{variable: values}` dict. We build a *threshold* selector: fire rows",
    "whose driver column exceeds a constructor-time threshold.",
))

cells.append(code(
    "class ThresholdSelector:",
    "    \"\"\"Fire rows whose driver column exceeds a fixed threshold.",
    "",
    "    Returns a BOOLEAN mask (deterministic case of the contract).",
    "    \"\"\"",
    "",
    "    def __init__(self, variables, column_mapping, threshold=0.0):",
    "        self.variables = list(variables)",
    "        self.column_mapping = column_mapping",
    "        self.threshold = threshold",
    "        unknown = [v for v in self.variables if v not in column_mapping]",
    "        if unknown:",
    "            raise ValueError(f\"variables not in column_mapping: {unknown}\")",
    "",
    "    def __call__(self, X, y=None, stats=None):",
    "        out = {}",
    "        for v in self.variables:",
    "            col = X[:, self.column_mapping[v]]",
    "            out[v] = col > self.threshold",
    "        return out",
    "",
    "    def __repr__(self):",
    "        return f\"ThresholdSelector({self.variables!r}, threshold={self.threshold})\"",
    "",
    "",
    "sel = ThresholdSelector(['f0'], column_mapping={'f0': 0}, threshold=1.0)",
    "X = default_rng(0).normal(size=(100, 2))",
    "values = sel(X, None)",
    "print(sel)",
    "print('dtype:', values['f0'].dtype, '| fired:', int(values['f0'].sum()))",
))

cells.append(md(
    "## A score-returning variant",
    "",
    "Now the probabilistic flavor: instead of a mask we return the",
    "probability that each row is fired. We make it a smooth function of the",
    "driver column — the further above the threshold, the closer to 1.0.",
    "Notice the selector draws **no randomness**.",
))

cells.append(code(
    "def sigmoid(z):",
    "    return 1.0 / (1.0 + np.exp(-z))",
    "",
    "",
    "class SmoothThresholdSelector:",
    "    \"\"\"Scores: P(deficit) = sigmoid((col - threshold) / slope).\"\"\"",
    "",
    "    def __init__(self, variables, column_mapping, threshold=0.0, slope=1.0):",
    "        self.variables = list(variables)",
    "        self.column_mapping = column_mapping",
    "        self.threshold = threshold",
    "        self.slope = slope",
    "",
    "    def __call__(self, X, y=None, stats=None):",
    "        out = {}",
    "        for v in self.variables:",
    "            col = X[:, self.column_mapping[v]]",
    "            scores = sigmoid((col - self.threshold) / self.slope)",
    "            assert np.all((scores >= 0) & (scores <= 1))",
    "            out[v] = scores",
    "        return out",
    "",
    "",
    "sel = SmoothThresholdSelector(['f0'], column_mapping={'f0': 0}, threshold=0.0, slope=0.5)",
    "values = sel(X, None)['f0']",
    "print('range:', values.min(), '...', values.max())",
    "print('monotone in driver:', np.all(np.diff(values[np.argsort(X[:, 0])]) >= 0))",
))

cells.append(md(
    "## Consuming your selector with a real generator",
    "",
    "Selectors are only useful through consumers. The two realization",
    "duties live on opposite sides of the contract:",
    "",
    "* **boolean masks** are realized directly by *gating* consumers, e.g.",
    "  the causal drift generator (fires the mask row-wise, in place);",
    "* **fractional scores** are realized by *subsampling* consumers, e.g.",
    "  `RandomSamplingTargetsGenerator` (resamples rows with keep-weight",
    "  `1 - score`; a boolean mask therefore yields an EXACT thinning —",
    "  see its tutorial for the full semantics).",
    "",
    "Below we consume our `SmoothThresholdSelector` (scores) with the",
    "subsampling consumer and verify that the resampled rows skew toward",
    "low-score rows.",
))

cells.append(code(
    "from badgers.generators.tabular_data.imbalance import RandomSamplingTargetsGenerator",
    "",
    "# A SUBSAMPLING consumer: realizes fractional scores as keep-weights.",
    "# It resamples `size` rows with replacement, weighted by (1 - score):",
    "# rows with a LOW deficit probability (score ~ 0) are likely kept/replicated,",
    "# rows with a HIGH score (deficit) are under-sampled.",
    "rng = default_rng(0)",
    "sel = SmoothThresholdSelector(['f0'], column_mapping={'f0': 0}, threshold=0.0, slope=0.5)",
    "Xw = default_rng(1).normal(size=(200, 2))",
    "gen = RandomSamplingTargetsGenerator(random_generator=rng)",
    "Xt, _ = gen.generate(Xw, y=None, selector=sel, size=100)",
    "",
    "scores = sel(Xw, None)['f0']",
    "print('input rows:', len(Xw), '-> resampled rows:', len(Xt))",
    "print('population mean f0:', Xw[:, 0].mean().round(3))",
    "print('resampled mean f0: ', Xt[:, 0].mean().round(3), '(skewed toward low-score rows)')",
    "assert Xt[:, 0].mean() < Xw[:, 0].mean()",
))

cells.append(md(
    "## Conformance checklist",
    "",
    "Before contributing a selector, check:",
    "",
    "* [ ] constructor validates variables/column_mapping and raises clearly;",
    "* [ ] `__call__` accepts `(X, y)` and, if used, `stats=None`;",
    "* [ ] returned dict covers exactly the declared variables;",
    "* [ ] boolean outputs are `np.bool_` arrays; score outputs stay in [0, 1]",
    "  (assert internally);",
    "* [ ] no rng draws inside `__call__` (unless documented as a realizing",
    "  preset);",
    "* [ ] deterministic given the same inputs;",
    "* [ ] consumed correctly by a gating generator AND, if scores are",
    "  meaningful as weights, by a subsampling generator;",
    "* [ ] `repr` shows the intensity parameters.",
))

write(r"docs\dev\Create-New-Selectors.ipynb", cells)

print("DONE")
