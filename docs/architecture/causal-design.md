# Causal defect family

The causal generators answer a different question from the rest of the library: not "replace
these cells with bad values" but "if this cell is bad, what does that imply for the rest of the
record?" That requires a model of how the columns depend on each other, and that model is the
`DGP`.

The design has four layers. Each layer knows only about the layer below it.

## 1. Structure — static, user-provided

- the structural `nx.DiGraph` (latent variables),
- the `column_mapping` (node → integer column),
- an optional observation `nx.DiGraph` (measurement nodes, `R_<name>` indicator nodes,
  derivation edges).

Validation lives in `badgers.core.causal_graph` as plain functions.

## 2. Fitted knowledge — the `DGP` protocol

A backend implements exactly **one** operation:

```python
fit_node(node, parents, data)
```

and exposes two views:

| View | Meaning |
|---|---|
| `response(node, parent_values)` | the fitted deterministic part |
| `draw_noise(node, rng, n)` | the node's own shock — parametric or empirical |

`fit(X)` is a provided topological loop, not a backend responsibility.

The DGP is **immutable and rngless**: every stochastic method takes the caller's `rng`, and
nothing retains training data.

## 3. Cascade composition — the `structural` hook

`structural(node, parent_values, rng, n)` composes *one* node's values for a cascade as

```
response(parent_values) + draw_noise(node, rng, n)
```

This is the default in `DGPMixin`. Backends whose mechanisms are not additive override it — the
gcm backend delegates to the mechanism's own `evaluate(parents, noise)` (rounded discrete
predictions, class CDFs) while badgers still draws the noise.

`sample` and `intervene` never compose the two views themselves; they always go through
`structural`.

## 4. Deficit algebra — written once, backend-blind

The generic `DGPMixin` cascade provides:

- `intervene(rows, nodes, values=None, *, rng=None)` — Pearl surgery on given rows: fix the
  nodes, recompute **all** descendants in topological order with fresh noise; intervened nodes
  stay pinned.
- `sample(rng, n)` — draw a fresh dataset from the fitted model.

Both start by calling the **`pick_parameters(rng)` hook** (default: `self`; the PyMC backend
overrides it to pin one coherent posterior draw per cascade).

Structure edits are **free functions** in `badgers.core.dgp.edits` — the DGP itself has *no*
edit methods:

| Function | Effect |
|---|---|
| `scale_edge` / `remove_edge` | edge reweighting with variance-budget calibration — no coefficient access needed |
| `exogenize` | make a node a root; **intervention-only** semantics |
| `set_form` | replace a node's functional form |
| `add_edge_contribution` | ANM-scoped cheap edge addition |

Edits return read-only views and never mutate the base DGP. Re-estimation after a structural
modification is out of scope: build the modified SCM with the wrapped library's own API, fit
it, and wrap it again with `GcmSCMDGP` / `PyMCSCMDGP`.

Indicator derivation from the observation graph lives in the missingness generator
(structure-layer helpers + generator-local selection/spread logic).

## Backend selection

**Object injection only** — `dgp=...` in the constructor, or `use_dgp(dgp)`.

- Deficit semantics live in the generators; they never see which backend ran.
- The generator carries **no structure of its own**: `CausalGenerator.__init__` accepts only
  `random_generator=` and `dgp=`, and derives its `graph` / `column_mapping` views from the DGP.
- There is no generator-side `fit` and no implicit default backend — a missing `dgp=` is a hard
  error.
- Unfitted DGPs are fitted lazily on the first `generate(X)`.

```python
from badgers.core.dgp.backends import GcmSCMDGP
from badgers.generators.tabular_data.missingness import CausalMissingnessGenerator

dgp = GcmSCMDGP(make_gcm_scm(graph, column_mapping)).fit(X)   # fit ONCE, on clean X
missingness = CausalMissingnessGenerator(random_generator=rng, dgp=dgp)
drift = CausalDriftGenerator(random_generator=rng, dgp=dgp)   # share the same world
```

Sharing one fitted DGP across a pipeline guarantees a single coherent world for the whole
pipeline. Fitting lazily inside each generator would fit the second one on already-contaminated
data.

## Why this shape

| Decision | Reason |
|---|---|
| One backend operation (`fit_node`) | Any SCM library can be adapted by implementing one method. |
| `response` / `draw_noise` split | Lets badgers own the noise draw, so cascades stay reproducible and backend-independent. |
| Immutable, rngless DGP | No hidden state; a cascade is a pure function of `(dgp, rng, rows)`. |
| Edits as free functions | Keeps the protocol minimal; edits are a *vocabulary*, not a responsibility of the model. |
| No implicit default backend | Silent fallbacks make results irreproducible; an explicit `dgp=` is one line. |
