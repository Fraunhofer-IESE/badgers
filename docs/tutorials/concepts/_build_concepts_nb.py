"""Builder for the tier-1 concepts notebook — step-by-step, dummy-proof ordering.

Each new concept gets: (a) a markdown cell saying what it is about, and
(b) a code cell exemplifying exactly that concept. Concepts are introduced
strictly one at a time; no forward references except explicit pointers
("covered in section N").
"""
import io
import json
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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    full = os.path.join(ROOT, path)
    with io.open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {path} cells: {len(cells)}")


cells = []

# ---------------------------------------------------------------- title
cells.append(md(
    "# Understanding badgers: concepts and contracts",
    "",
    "Badgers generates **bad data**: it takes known-good datasets and injects",
    "well-defined *deficits* — outliers, noise, drift, missingness, imbalance,",
    "and causal perturbations — for testing and benchmarking data-science",
    "pipelines.",
    "",
    "This notebook walks through the concepts of badgers **one step at a",
    "time**. Every step has a code cell that exemplifies exactly the concept",
    "it introduces — nothing is used before it has been explained. By the",
    "end you will know:",
    "",
    "1. where generators live (module structure),",
    "2. how to create one (randomness),",
    "3. how to call it (`generate`),",
    "4. what you get back (shapes),",
    "5. what it does to your input (in-place mutation),",
    "6. the `y` deprecation,",
    "7. selectors, data-generating processes (DGP), and pipelines.",
))

# ---------------------------------------------------------------- step 1
cells.append(md(
    "## 1. Where generators live: the module structure",
    "",
    "**This step is about:** *how the badgers package is organized.*",
    "",
    "Generators are grouped first by **modality** (`tabular_data`,",
    "`time_series`, `text`, `image`, `graph`, `geolocated_data`), then by",
    "**deficit family** (`outliers`, `noise`, `drift`, `missingness`,",
    "`imbalance`, `relationship`). Every family is a Python package of",
    "modules, and every generator is a **class** whose name ends in",
    "`Generator`.",
    "",
    "The cell below lists all tabular-data generator modules and the",
    "generator classes they define. It only inspects the package — nothing",
    "is instantiated yet.",
))

cells.append(code(
    "import pkgutil",
    "import importlib",
    "import inspect",
    "import badgers.generators.tabular_data as tabular",
    "",
    "def walk(pkg):",
    "    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):",
    "        mod = importlib.import_module(f'{pkg.__name__}.{m.name}')",
    "        if m.ispkg:",
    "            walk(mod)",
    "        else:",
    "            classes = [n for n, c in inspect.getmembers(mod, inspect.isclass)",
    "                       if n.endswith('Generator') and c.__module__ == mod.__name__]",
    "            if classes:",
    "                rel = mod.__name__.replace('badgers.generators.tabular_data.', '')",
    "                print(f'{rel:40s} {classes}')",
    "",
    "walk(tabular)",
))

# ---------------------------------------------------------------- step 2
cells.append(md(
    "## 2. Create a generator (and own the randomness)",
    "",
    "**This step is about:** *turning a generator class into an object — and",
    "the one constructor argument every generator takes.*",
    "",
    "A generator is a plain Python class; you create it with `()` like any",
    "other. Each generator takes a `numpy.random.Generator` at construction:",
    "",
    "```python",
    "generator = SomeGenerator(random_generator=default_rng(seed))",
    "```",
    "",
    "Badgers generators **never touch global random state**. The sequence of",
    "random numbers they use is the one you hand them, so:",
    "",
    "* same seed => identical outputs, bit for bit;",
    "* you own the stream — create the generator once and pass it to",
    "  several generators to keep a single reproducible stream.",
))

cells.append(code(
    "from numpy.random import default_rng",
    "from badgers.generators.tabular_data.outliers.distribution_sampling import ZScoreSamplingGenerator",
    "",
    "outlier_gen = ZScoreSamplingGenerator(random_generator=default_rng(0))",
    "print(outlier_gen)",
))

cells.append(md(
    "**Same seed, same output.** Two generators built from the same seed",
    "produce bit-identical results — this is the reproducibility contract",
    "in action:",
))

cells.append(code(
    "import numpy as np",
    "from sklearn.datasets import make_blobs",
    "",
    "X, y = make_blobs(n_samples=100, centers=2, random_state=2)   # known-good data",
    "",
    "g1 = ZScoreSamplingGenerator(random_generator=default_rng(42))",
    "g2 = ZScoreSamplingGenerator(random_generator=default_rng(42))",
    "Xt1, _ = g1.generate(X, n_outliers=10)",
    "Xt2, _ = g2.generate(X, n_outliers=10)",
    "print('same seed -> identical output:', np.array_equal(Xt1, Xt2))",
    "",
    "g3 = ZScoreSamplingGenerator(random_generator=default_rng(7))",
    "Xt3, _ = g3.generate(X, n_outliers=10)",
    "print('different seed -> different output:', not np.array_equal(Xt1, Xt3))",
))

# ---------------------------------------------------------------- step 3
cells.append(md(
    "## 3. Call a generator: `generate()`",
    "",
    "**This step is about:** *the one method you call — and what it returns.*",
    "",
    "Every generator exposes a single method:",
    "",
    "```python",
    "Xt, yt = generator.generate(X, y=None, **params)",
    "```",
    "",
    "* `X` — your known-good data (a 2D NumPy array or pandas DataFrame);",
    "* `y` — *deprecated*, see section 6 (pass nothing for now);",
    "* `**params` — the knobs of the deficit, different per generator;",
    "* the return is a **tuple** `(Xt, yt)` — the transformed features and",
    "  whatever `y` you passed in (here `None`).",
    "",
    "This example adds 10 outliers to `X`. Note what happens to the",
    "shape: the outliers are **appended** as new rows, so `Xt` is bigger",
    "than `X`.",
))

cells.append(code(
    "outlier_gen = ZScoreSamplingGenerator(random_generator=default_rng(0))",
    "",
    "Xt, yt = outlier_gen.generate(X, n_outliers=10)   # y omitted -> None",
    "print('outliers:', X.shape, '->', Xt.shape)",
    "print('yt is:', yt)",
))

cells.append(md(
    "**Where does the bookkeeping go?** Not into `yt`. Generators that need",
    "to tell you which rows carry the deficit store that *inside the",
    "generator* and expose it through a dedicated getter — for outliers,",
    "`get_outliers_labels()`:",
))

cells.append(code(
    "labels = outlier_gen.get_outliers_labels()      # bool array: True = outlier row",
    "print('labels:', labels.dtype, 'length', len(labels), '| outlier rows:', int(labels.sum()))",
    "print('first rows are original:', not labels[:len(X)].any())",
    "print('last rows are outliers:', labels[len(X):].all())",
))

# ---------------------------------------------------------------- step 4
cells.append(md(
    "## 4. What you get back: the shape rules",
    "",
    "**This step is about:** *the size of `Xt` compared to `X`.*",
    "",
    "The shape of `Xt` depends on the deficit family — there are exactly",
    "three modes:",
    "",
    "| Mode | Families | Effect on shape |",
    "|---|---|---|",
    "| **Appending** | outliers | `Xt` is *larger* than `X` (deficit rows appended) — you saw this above |",
    "| **Gating** | noise, missingness, drift | `Xt` has the *same shape* as `X` (existing values are changed) |",
    "| **Subsampling** | imbalance | `Xt` is *smaller* than `X` (affected rows are dropped) |",
    "",
    "**Gating, exemplified:** additive noise replaces the existing values",
    "in place, so 200 rows stay 200 rows:",
))

cells.append(code(
    "import scipy.stats",
    "from badgers.generators.tabular_data.noise import AdditiveNoiseGenerator",
    "",
    "noise_gen = AdditiveNoiseGenerator(random_generator=default_rng(0))",
    "Xt, _ = noise_gen.generate(X, None, distribution=scipy.stats.norm(scale=0.5))  # y=None (deprecated slot)",
    "print('noise (gating):', X.shape, '->', Xt.shape)",
))

cells.append(md(
    "**Subsampling, exemplified:** an imbalance generator *removes* the",
    "rows affected by the deficit. The only thing it needs to know is",
    "*which rows* — that decision is packaged in a `selector` (a boolean",
    "mask per column). Selectors are their own concept and get a full",
    "section below (section 7); here we just use one preset so the shape",
    "rule is visible: about half of the 100 rows are removed.",
))

cells.append(code(
    "from badgers.generators.tabular_data.imbalance.imbalance import RandomSamplingTargetsGenerator",
    "from badgers.core.selectors import BernoulliSelector",
    "",
    "imbalance_gen = RandomSamplingTargetsGenerator(random_generator=default_rng(1))",
    "selector = BernoulliSelector(['f0'], rate=0.5, rng=default_rng(2))",
    "",
    "Xt, yt = imbalance_gen.generate(X, selector=selector)",
    "print('imbalance (subsampling):', X.shape, '->', Xt.shape)",
))

# ---------------------------------------------------------------- step 5
cells.append(md(
    "## 5. Mind your originals: no copies are made for you",
    "",
    "**This step is about:** *what `generate()` does to the `X` you passed.*",
    "",
    "Badgers does **not** defensively copy your inputs. Some generators",
    "mutate `X` **in place** (for example, drift generators add the shift",
    "directly to the columns you passed in). If you need the pristine",
    "dataset afterwards, copy it yourself:",
    "",
    "```python",
    "Xt, yt = generator.generate(X.copy())",
    "```",
))

cells.append(code(
    "from badgers.generators.tabular_data.drift.causal import CausalDriftGenerator",
    "from badgers.core.dgp import OlsDGP",
    "import networkx as nx",
    "",
    "# a tiny causal DGP for the drift generator to operate on",
    "# (DGP mechanics are explained in section 8 — here it is just plumbing)",
    "X_work = X.astype(float)",
    "graph = nx.DiGraph([('f0', 'f1')])",
    "dgp = OlsDGP(graph, {'f0': 0, 'f1': 1}).fit(X_work)",
    "",
    "drift_gen = CausalDriftGenerator(random_generator=default_rng(1), dgp=dgp)",
    "Xt = drift_gen.generate(X_work, None, perturbation_nodes=['f0'], drift_std=3.0)[0]",
    "print('same object (mutated in place):', Xt is X_work)",
))

# ---------------------------------------------------------------- step 6
cells.append(md(
    "## 6. The `y` deprecation: from `(X, y) -> (Xt, yt)` to `X -> Xt`",
    "",
    "**This step is about:** *the API transition you must know about.*",
    "",
    "The original badgers API treated the target as a first-class citizen:",
    "`generate(X, y) -> (Xt, yt)`. It is **deprecated** for the simpler",
    "`generate(X) -> Xt` contract:",
    "",
    "* the `y` argument is no longer *used* by any generator — selection",
    "  and targeting work on columns of `X` (via selectors, section 7);",
    "* `yt` is never modified and never invented — it is a pure",
    "  pass-through, so it carries no information;",
    "* the `y` argument and the `yt` return slot will be **removed in the",
    "  next release (0.1.x)**.",
    "",
    "What to do today:",
    "",
    "* **do not pass a real `y`** — where the signature still requires one",
    "  positionally, pass `None`;",
    "* but `generate` still returns a *tuple* `(Xt, yt)`, so unpack it",
    "  defensively: `Xt, _ = generator.generate(X, ...)`. When `y` is",
    "  removed in 0.1.x, `generate` will return `Xt` alone.",
    "",
    "The cells above already follow this pattern.",
))

# ---------------------------------------------------------------- step 7
cells.append(md(
    "## 7. Selectors: which rows get the deficit",
    "",
    "**This step is about:** *the object that decides who is affected.*",
    "",
    "Several generators (drift, missingness, noise, imbalance, ...) accept a",
    "**selector** — the object that decides *which rows receive the",
    "deficit*. The contract:",
    "",
    "```python",
    "values = selector(X, y)   # -> {variable: array}",
    "```",
    "",
    "Per variable (column) the returned array is either:",
    "",
    "* a **boolean mask** — `True` = the deficit is applied to that row",
    "  (deterministic), or",
    "* a **float score in [0, 1]** — the *probability* that the deficit is",
    "  applied to that row (1.0 = definitely, 0.0 = never).",
    "",
    "A selector **never draws randomness for the rates** — it only proposes",
    "them. Realizing the probabilities belongs to the *consumer* generator:",
    "gating generators fire row-wise Bernoulli-wise; resampling generators",
    "use the scores as selection weights.",
    "",
    "badgers ships presets you can compose:",
    "",
    "| Preset | Behaviour |",
    "|---|---|",
    "| `AlwaysSelector(variables)` | every row fired |",
    "| `BernoulliSelector(variables, rate, rng)` | pre-realized boolean mask, each row fired with probability `rate` |",
    "| `ExactCountSelector(variables, rate, weights, rng)` | exactly `rate * n` rows |",
    "| `TopQuantileSelector(...)` | rows in the top quantile of a driver column |",
    "| `ClassSelector(...)` | per-class proportions |",
    "| `ProportionalSelector(...)` | deficit probability proportional to a column |",
))

cells.append(code(
    "from badgers.core.selectors import BernoulliSelector, TopQuantileSelector",
    "",
    "X_raw, _ = make_blobs(n_samples=200, centers=2, random_state=3)",
    "",
    "# BernoulliSelector is a PRE-REALIZING preset: it draws its rng at",
    "# call time and returns a boolean mask (True = affected row).",
    "selector = BernoulliSelector(['f0'], rate=0.2, rng=default_rng(0))",
    "mask = selector(X_raw, y=None)['f0']",
    "print('BernoulliSelector mask dtype:', mask.dtype, '| fired rows:', int(mask.sum()))",
    "",
    "# A boolean-mask selector: rows in the top 10% of f0.",
    "selector = TopQuantileSelector(['f0'], column_mapping={'f0': 0}, quantile=0.9)",
    "mask = selector(X_raw, y=None)['f0']",
    "print('TopQuantileSelector mask dtype:', mask.dtype, '| fired rows:', int(mask.sum()))",
))

# ---------------------------------------------------------------- step 8
cells.append(md(
    "## 8. Data-generating process (DGP)",
    "",
    "**This step is about:** *the fitted causal model behind the `Causal...`",
    "generators.*",
    "",
    "The causal generators (drift / missingness / noise / outliers *via",
    "causal graphs*, edge perturbation) need a **DGP** — a fitted",
    "data-generating process. A DGP bundles:",
    "",
    "* a `networkx.DiGraph` describing the causal structure,",
    "* a `column_mapping` from node names to column positions in `X`, and",
    "* per-node **mechanisms** estimated from your data.",
    "",
    "Three verbs:",
    "",
    "```python",
    "dgp = OlsDGP(graph, column_mapping).fit(X)   # fit mechanisms",
    "Xs = dgp.sample(rng, n)                      # sample fresh data",
    "Xi = dgp.intervene(X.copy(), ['B'], [values], rng=rng)  # do(B = ...)",
    "```",
    "",
    "The default backend `OlsDGP` is a deterministic per-node least-squares",
    "estimator (pure NumPy, no extras). Optional estimation backends live in",
    "`badgers.core.dgp.backends` (`DowhyGcmDGP`, `PymcDGP` — installed via",
    "the `badgers[dowhy]` / `badgers[pymc]` extras). See the dev notebook",
    "*Creating DGP Backends* for the full protocol.",
))

cells.append(code(
    "import networkx as nx",
    "from badgers.core.dgp import OlsDGP",
    "",
    "graph = nx.DiGraph([('A', 'B'), ('B', 'C')])",
    "column_mapping = {'A': 0, 'B': 1, 'C': 2}",
    "",
    "rng = default_rng(0)",
    "X_raw = np.zeros((500, 3))",
    "X_raw[:, 0] = rng.normal(size=500)",
    "X_raw[:, 1] = 2.0 * X_raw[:, 0] + rng.normal(scale=0.1, size=500)",
    "X_raw[:, 2] = 0.5 * X_raw[:, 1] + rng.normal(scale=0.1, size=500)",
    "",
    "dgp = OlsDGP(graph, column_mapping).fit(X_raw)",
    "",
    "# sample fresh data from the fitted DGP (signature: sample(rng, n))",
    "Xs = dgp.sample(default_rng(1), 10)",
    "print('sampled shape:', Xs.shape)",
    "",
    "# intervene in place: pass a copy, pin B, recompute C downstream",
    "Xi = dgp.intervene(X_raw.copy(), ['B'], [np.full(500, 10.0)], rng=default_rng(2))",
    "print('B pinned:', np.allclose(Xi[:, 1], 10.0))",
    "print('C propagated (B=10 -> C~5):', np.allclose(Xi[:, 2], 5.0, atol=0.5))",
))

# ---------------------------------------------------------------- step 9
cells.append(md(
    "## 9. Pipelines: chaining deficits",
    "",
    "**This step is about:** *applying several deficits in one call.*",
    "",
    "Deficits compose. A `Pipeline` takes a dict of generators (the keys",
    "are step names) and applies them in order, passing each output to the",
    "next step. Per-step parameters are passed at `generate()` time keyed",
    "by the step name:",
    "",
    "```python",
    "pipeline = Pipeline({",
    "    'noise': AdditiveNoiseGenerator(rng),",
    "    'outliers': ZScoreSamplingGenerator(rng),",
    "})",
    "Xt, _ = pipeline.generate(X, params={",
    "    'noise': {'distribution': norm(scale=0.3)},",
    "    'outliers': {'n_outliers': 10},",
    "})",
    "```",
))

cells.append(code(
    "from badgers.core.pipeline import Pipeline",
    "from badgers.generators.tabular_data.outliers.distribution_sampling import HypersphereSamplingGenerator",
    "",
    "pipeline = Pipeline({",
    "    'noise': AdditiveNoiseGenerator(random_generator=default_rng(0)),",
    "    'outliers': HypersphereSamplingGenerator(random_generator=default_rng(0)),",
    "})",
    "Xt, _ = pipeline.generate(",
    "    X,",
    "    None,   # y=None (deprecated slot)",
    "    params={",
    "        'noise': {'distribution': scipy.stats.norm(scale=0.3)},",
    "        'outliers': {'n_outliers': 10},",
    "    },",
    ")",
    "print('pipeline output:', X.shape, '->', Xt.shape)",
))

# ---------------------------------------------------------------- next
cells.append(md(
    "## 10. Where to go next",
    "",
    "| You want to ... | Tutorial |",
    "|---|---|",
    "| Add outliers to tabular data | *Generating Outliers* (Part A) |",
    "| Add noise | *Generating Noise* (Part A) |",
    "| Introduce missing values | *Generating Missing Data* (Part A) |",
    "| Generate class imbalance | *Generating Imbalanced Data* (Part A) |",
    "| Shift data over time (drift) | *Generating Drift* (Part A) |",
    "| Change relationships between columns | *Perturbing Edge Strength* |",
    "| Make deficits structurally consistent (with a causal graph) | each tabular tutorial, **Part B** — Acting on the DGP |",
    "| Chain several deficits | *Chaining Generators* |",
    "| Build your own generator / selector / DGP backend | the *Implementing Novel Generators* section (three dev notebooks) |",
))

write(r"docs\tutorials\concepts\Badgers-Concepts.ipynb", cells)

print("DONE")
