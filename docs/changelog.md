# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## dev_0_0_15 (unreleased)

### Breaking changes

- **`PyMCDGP` and the old `badgers.core.dgp` module are removed.** The causal
  defect generators now fit a **per-node data generation process** behind the
  new `DGP` protocol (`badgers.core.dgp` is a package). The default backend is
  a deterministic pure-NumPy least-squares DGP (`OlsDGP`) — no extra
  dependency needed. Custom backends are injected as objects (`dgp=...` /
  `use_dgp`); PyMC-specific knobs (`n_samples_mcmc`, `fit_dgp`) no longer
  appear in `fit(...)`. PyMC- and DoWhy-GCM-based backends are planned as
  follow-ups (the `causal` extra still installs PyMC + PyTensor for writing
  them).
- **`CausalGenerator.fit` removed AND the `graph=`/`column_mapping=`
  generator-side structure is gone (breaking).** The DGP is the ONLY structure
  a causal generator accepts: the constructor is `CausalGenerator(random_generator=..., dgp=None)`
  and the only structure param anywhere on the generator is `dgp=` (or
  `use_dgp(dgp)`). The DGP carries `graph` and `column_mapping` itself — pass
  `dgp = OlsDGP(graph, column_mapping).fit(X)` and pass it via the constructor;
  there is no implicit OLS default backend inside the generator. Unfitted DGPs
  are fitted lazily on first `generate(X)` (`DGP.is_fitted` is respected).
- **`fit` signature change** for the causal defect family:
  `fit(X, graph, column_mapping, dgp=None)` — `n_samples_mcmc` and
  `fit_dgp` are gone.
- **`AdditiveNoiseGenerator`** (new, non-causal): injects noise drawn from
  any *frozen* scipy distribution object (`distribution=...
  `.rvs(size, random_state=rng)`) instead of hard-coded Gaussian white
  noise (e.g. `scipy.stats.laplace(scale=1)`, `scipy.stats.t(df=3)`).
  `noise_std` is a global magnitude multiplier (1.0 = raw draw); what you
  pass is what is added — no rescaling. Optional `perturbation_nodes`
  restricts the noise to a column subset.
- **`CausalAdditiveNoiseGenerator`** (new, causal): same swappable
  distribution at the perturbation nodes, with descendant recomputation
  via the DGP (`dgp.intervene` — Pearl surgery). Row gating is delegated
  to a `selector` (default: every row fires); strictly in-place (no
  `n_copies`).
- **`GaussianNoiseGenerator` removed (breaking).** It is a strict special
  case of the new `AdditiveNoiseGenerator`: replace
  `GaussianNoiseGenerator().generate(X, y, noise_std=0.5)` with
  `AdditiveNoiseGenerator().generate(X, y, distribution=scipy.stats.norm(scale=0.5))`.
  (Note the semantic shift: the old class standardized first, so its
  `noise_std` was column-relative; the new API adds the frozen
  distribution's raw draws — pass the per-column scale yourself.)
  `GaussianNoiseClassesGenerator` (per-class noise) remains; the
  time-series `Local/GlobalGaussianNoiseGenerator` are unrelated and
  unchanged.
- **`CausalEdgePerturbationGenerator.generate` drops the dead
  `mechanism`/`mechanism_cause`/`mechanism_rate` parameters** — they were
  accepted but ignored (the edge generator's gate is always "all rows").
  The signature is now
  `generate(X, y, perturbation_nodes=None, edge=None, weight_factor=2.0)`.
- **`CausalImbalanceGenerator` no longer requires a DGP** (breaking): the
  generator is a pure row-subset (quantile-range truncation); the kept
  rows' parents are unchanged, so no structural equation is re-evaluated.
  The dead `mechanism`/`mechanism_cause`/`mechanism_rate` parameters, the
  on-demand OLS fit, and the no-op propagate pass are removed — the
  signature is now
  `generate(X, y, perturbation_nodes=None, quantile_range=(0.5, 1.0))`.
  This is the first generator in the causal family whose `generate` needs
  neither a DGP nor any random numbers (selection-side only).
- **`CausalMissingnessGenerator` is chooser-only (breaking)**: missingness is
  an observation-layer defect — it only hides cells (writes NaN) and does
  NOT cascade to descendants. The `missingness_rule` mechanism parameter,
  the on-demand DGP fit, and the `any`/`all` NaN-propagation passes are
  removed; the fire mask comes from a `selector` object (see the selector
  bullet below). Signature:
  `generate(X, y, perturbation_nodes=None, selector=None)`.
- **String `mechanism` flags replaced by `selector` objects (breaking).**
  The row-gating knobs of the causal defect generators
  (`CausalDriftGenerator`, `CausalMissingnessGenerator`,
  `CausalNoiseGenerator`, `CausalAdditiveNoiseGenerator`) are now a single
  `selector=` parameter holding a callable from
  `badgers.core.selectors`; `None` defaults to "every row fires"
  (`AlwaysSelector` semantics). The `mechanism`/`mechanism_cause`/
  `mechanism_rate` strings and the `_fire_masks` string dispatcher are
  removed. Migration:
  - `mechanism="always"` → omit `selector` (or `AlwaysSelector(nodes)`).
  - `mechanism="random", mechanism_rate=r` →
    `selector=BernoulliSelector(nodes, rate=r, rng=gen.random_generator)`.
  - `mechanism="self", mechanism_rate=r` →
    `selector=TopQuantileSelector(nodes, column_mapping, quantile=1-r)`.
  - `mechanism="observed", mechanism_cause=c, mechanism_rate=r` →
    `selector=TopQuantileSelector(nodes, column_mapping, quantile=1-r,
    drivers=c)`.
  Intensity lives on the selector object: `rate` and `quantile` are now
  CONSTRUCTOR arguments of `BernoulliSelector`/`ExactCountSelector`/
  `TopQuantileSelector` (previously `rate` was a `__call__` kwarg and
  `quantile` a call kwarg) — so an intensity sweep means constructing a
  new selector. `AlwaysSelector` was added (the all-rows preset).
- **`RandomSamplingClassesGenerator` takes a `selector` (breaking).** The
  `proportion_classes` call kwarg is removed; pass instead
  `selector=ClassSelector("cls", column_mapping, proportions, rng=...)` from
  `badgers.core.selectors`, where `proportions` maps each class column to
  `{class_value: proportion}`. Class columns are columns of `X` (the
  target `y` is being deprecated as a data carrier: selection no longer
  reads it). The selector's fire mask is the standard
  contract (True = deficit); the generator's deficit is DROPPING the fired
  rows (subsampling), so the output is smaller than the input. Classes
  absent from the mapping are kept as-is; proportions are shares of the
  total row count, clipped to the rows available in each class. The class
  variable must be a column of `X` resolved via
  `column_mapping` (the target `y` is deprecated as a data carrier and is
  never read for selection). `selector=None` is a
  no-op. Migration: `generate(X, y, proportion_classes={0: 0.5, ...})` →
  `generate(np.column_stack([X, y]), y, selector=ClassSelector("cls",
  {"cls": n_features}, {"cls": {0: 0.5, ...}}, rng=rng))`. Note the new
  semantics sample WITHOUT replacement (the old
  implementation duplicated rows when a proportion exceeded the class
  size).
- **Outlier labels are internal generator state, no longer returned in
  ``yt`` (breaking).** All outlier generators
  (`HyperCubeSampling`, `ZScoreSamplingGenerator`,
  `HypersphereSamplingGenerator`, `UniformInstanceAttributeSampling`,
  `IndependentHistogramsGenerator`, `HistogramSamplingGenerator`,
  `LowDensitySamplingGenerator`, `CausalOutlierPropagationGenerator`,
  and `DecompositionAndOutlierGenerator`) now expose
  ``get_outliers_labels()``: a boolean vector aligned with the rows of
  the LAST generated output, ``True`` where the row is an outlier,
  ``False`` where it is an original row. ``generate()`` returns
  ``(Xt, y)`` with ``y`` a pure pass-through (and ``y=None`` signatures
  throughout — the previous ``"original"``/``"outliers"`` string arrays
  are gone). Calling ``get_outliers_labels()`` before ``generate()``
  raises ``RuntimeError``. Migration: ``Xt, yt = gen.generate(X)``
  followed by ``yt == "outliers"`` → ``Xt, y = gen.generate(X)``
  followed by ``gen.get_outliers_labels()``.
- **Selector scores may now be fractional (additive contract extension).**
  A selector's per-row value may be a float in ``[0, 1]`` — the
  PROBABILITY that the deficit is applied — where boolean masks remain
  the degenerate (deterministic) case. A selector never draws randomness
  for these rates; realization belongs to the consuming generator (a
  gating consumer fires Bernoulli-wise, a resampling consumer uses the
  scores as selection weights). All existing presets return boolean
  masks, so nothing breaks; the new ``ProportionalSelector(variables,
  column_mapping)`` preset returns ``normalize_proba(X[:, col])`` (the
  deficit probability proportional to a column's value).
- **`RandomSamplingTargetsGenerator` takes a `selector` (breaking).** The
  `sampling_proba_func(y)` kwarg is removed; the generator now resamples
  rows weighted by ``1 - deficit_probability`` from a selector (noisy-OR
  combined over its variables), i.e. rows with a high deficit probability
  are dropped/under-sampled, rows with a low deficit probability are kept
  or replicated. Signature:
  `generate(X, y=None, *, selector=None, replace=True, size=None)`;
  `replace=True` (default) may duplicate rows (importance resampling),
  `replace=False` gives a weighted thinning, `size` controls the output
  row count (default: input row count). The target `y` is no longer read
  for selection (columns of `X` instead). Migration: the old default
  (sample ∝ target values) becomes
  `ProportionalSelector("col", {"col": idx})` over whichever `X` column
  holds the signal (e.g. the stacked target column), reading
  `X` rather than `y`.
- **`RandomShiftClassesGenerator` takes a `selector` (breaking).** The
  class variable is read from a column of `X` (resolved via the
  selector's `column_mapping`), not from `y`; `y` is passed through
  unchanged. Signature:
  `generate(X, y=None, *, selector=None, shift_std=0.1)`. The selector is
  REQUIRED (`selector=None` raises) — a per-class drift has no class-free
  default; fire masks are OR-ed. `shift_std` is now a float (one
  independent draw magnitude per class value) or a dict
  `{class_value: std}`; the per-dimension 2D-array form is removed
  (per-class shifts are isotropic — scale the data first for anisotropic
  drift). Class values absent from a dict are skipped. Migration:
  `generate(X, y, shift_std=np.array([2., 2., 2., 2.]))` →
  `generate(np.column_stack([X, y]), y,
  selector=ClassSelector("cls", {"cls": n_features},
  {"cls": {c: 1.0 for c in classes}}, rng=rng), shift_std=2.0)`.
- **`GaussianNoiseClassesGenerator` takes a `selector` (breaking).** Same
  contract as above: classes are a column of `X` resolved via the
  selector's `column_mapping` (`selector=None` raises; a per-class lookup
  requires `selector.column_mapping` to resolve the class column). `y` is
  now passed through UNCHANGED — the old implementation rebuilt `yt` as
  the class labels, silently discarding the original target (bug fixed).
  Signature: `generate(X, y=None, *, selector=None,
  noise_std_per_class=dict())`; fired rows whose class value is absent
  from the dict receive no noise.
- **`do()` and `ScaledEdgeDGP` are removed from the public API.** They are
  replaced by the intervention/edition surface on the DGP protocol itself
  (see Added): a single data-level `DGP.intervene` and the structure edits
  as free functions in the new `badgers.core.dgp.edits` module (see below).
  `ordered_descendants` moved to `badgers.core.causal_graph` (still
  re-exported from `badgers.core.dgp`). Consequence for the causal defect
  generators: descendants of an intervention are recomputed with fresh
  noise (Pearl surgery) instead of propagating the delta with each node's
  exogenous noise preserved — the produced deficits are slightly "wider"
  (noise re-rolled downstream).
- **DGP structure-edit methods removed (breaking):**
  `scale_edge`/`remove_edge`/`exogenize`/`set_form` (and the
  `add_edge`/`reverse_edge` stubs) are no longer methods on the DGP. The
  DGP is an immutable, fitted model with ONLY evaluation verbs
  (`fit`/`response`/`draw_noise`/`sample`/`intervene`); all structure
  modifications live as FREE FUNCTIONS in the new `badgers.core.dgp.edits`
  module (`from badgers.core.dgp import edits`):
  `edits.scale_edge(dgp, child, parent, k)`,
  `edits.remove_edge(dgp, child, parent)`,
  `edits.add_edge_contribution(dgp, child, parent, contribution=... or
  coefficient=..., draw_noise=...)` (the A5 cheap edit, ANM-scoped; cycle
  and existing-edge checks; an estimated new parent routes to
  `edits.rebuild`), `edits.exogenize(dgp, node)` (INTERVENTION-ONLY
  semantics — see below), `edits.set_form(dgp, node, response_fn,
  draw_noise=None)` (Q2: an optional noise-law closure),
  `edits.reversed_graph(graph, child, parent)` + `edits.rebuild(dgp, X,
  graph=None)` (the A3 estimation route: one refit of the fully modified
  graph, same backend class). Edits return read-only views and never
  mutate the base DGP; they compose by delegation.

### Added

- **`ClassSelector` preset** (`badgers.core.selectors`): per-class
  fixed-count sampling on a discrete column — `ClassSelector(variables,
  column_mapping, proportions, rng=...)` with `proportions` mapping each
  class column to `{class_value: proportion}`. Discovers the distinct
  values from the data (no manual class enumeration), raises
  `ValueError` on class values absent from the column (typo protection),
  and fires exactly `min(int(prop * n_rows), n_rows_in_class)` rows per
  listed class (partial mappings are legitimate: unlisted classes are
  kept). Class columns are columns of `X` resolved via `column_mapping`
  (selection does not read `y` -- the target is being deprecated as a
  data carrier). First
  subsampling consumer: `RandomSamplingClassesGenerator`.

- **Per-node DGP protocol** (`badgers.core.dgp` package): `DGP` protocol and
  `DGPMixin` (topological `fit`/`sample` walk) — backends implement exactly
  one operation, `fit_node(node, parents, data)`, and expose
  `response(node, parent_values)` and `draw_noise(node, rng, n)`;
  `sample(rng, n)` is tier 2 and no generator depends on it.
- **`DowhyGcmDGP` backend** (`badgers.core.dgp.backends.dowhy_gcm`, lazy
  import; new `dowhy` optional extra — `pip install badgers[dowhy]`):
  per-node `dowhy.gcm.AdditiveNoiseModel` mechanisms with GCM's linear
  regressor as the prediction model and EMPIRICAL noise (resampled
  residuals/shocks through the caller's rng — GCM's own sampling paths use
  global `np.random` with no seed parameter). Same complete-case fit
  contract (G9) as `OlsDGP`.
- **`PyMCDGP` backend** (`badgers.core.dgp.backends.pymc`, lazy import;
  `causal` extra): per-node Bayesian linear regression (one small
  `pm.Model` + MCMC per node, `n_draws=200` knob) — port of the removed
  monolithic PyMC backend to the per-node protocol. The FULL posterior is
  kept; NO posterior-mean collapse.
- **`pick_parameters` hook** (`DGPMixin`): called once at the top of the
  generic `sample`/`intervene` cascades; returns the DGP instance whose
  mechanisms drive the batch (ONE coherent parameter set per call).
  Default: `self` (deterministic backends fit exactly one parameter set).
  `PyMCDGP` overrides it to pin ONE posterior draw from the caller's rng
  (a `_ParameterDrawView`): each cascade evaluates one coherent
  parameterization; the carrier itself evaluates draw 0. Posterior draws
  compose through the structure edits (an edited view rebinds the pinned
  base). The generator API is unchanged — backends are swappable
  transparently.
- **Value-level intervention `DGP.intervene(rows, nodes, values=None, *,
  rng=None)`** (Pearl surgery on given rows): fix the nodes at `values`
  (or at their current column values when `values=None`), recompute ALL
  their descendants in topological order with FRESH noise; upstream and
  non-descendant columns are untouched; intervened nodes stay pinned (an
  intervened descendant is not recomputed over its pinned value).
- **Structure edits as methods on the DGP** (read-only views; the base DGP
  is never mutated; edits compose by delegation):
  `scale_edge(child, parent, k)` (edge reweighting without coefficient
  access; `calibrate(data)` evaluates the variance budget and raises
  `ValueError` when infeasible), `remove_edge(child, parent)` (=
  `scale_edge(..., 0.0)` — SCM-surgery view), `exogenize(node)` (make the
  node a root: incoming edges removed from the view's graph), `set_form(node,
  response_fn)` (replace one node's functional form with a user closure;
  noise law inherited — pending Q2). `add_edge`/`reverse_edge` raise
  `NotImplementedError` pending the training-data-retention decision (A3).
- **`exogenize` is INTERVENTION-ONLY (documented contract)**: designed for
  the pin-then-intervene flow (the generator writes the node's column; the
  cascade never recomputes a root). `sample()` on an exogenized DGP draws
  the node as `0 + N(0, residual_std)` — NOT the real marginal; the
  empirical-resampling fix is deferred (it would require training-data
  retention).
- **`CausalOutlierPropagationGenerator` now propagates through the shared
  DGP machinery** (internal change): the hand-rolled per-call lstsq
  forward pass is replaced by `OlsDGP(...).fit(X)` +
  `dgp.intervene(outliers, perturbation_nodes, rng=...)` — uniform Pearl
  surgery semantics with the other causal generators. Behavior note: a
  degenerate zero-residual fitted mechanism now contributes no noise (the
  old `0.1*col_std` fallback is gone).
- **Causal defect family** (five generators sharing `CausalGenerator`, an
  explicit `fit(X, graph, column_mapping)` phase followed by `generate(...)`
  with `selector=` row gating):
  `CausalDriftGenerator` (single systematic shift, recomputed downstream),
  `CausalNoiseGenerator` (per-edge noise injection, `noise_std`),
  `CausalImbalanceGenerator` (quantile-range truncation — pure subset,
  descendants untouched), `CausalMissingnessGenerator` (m-graph missingness
  with `any`/`all` propagation, optional observation-graph derivation),
  `CausalEdgePerturbationGenerator` (marginal-preserving edge-weight
  perturbation; child recomputed with the new coefficient, descendants
  recomputed from the perturbed values).
- **`pymc` optional extra** (renamed from `causal`; `pip install
  "badgers[pymc]"`) — only needed for writing custom PyMC-based backends;
  the default OLS path is pure NumPy.

- **`TopQuantileSelector`** preset (`badgers.core.selectors`): fires rows at
  or above the ``quantile`` of a column — self mode (own column) or driver
  mode (a driver column, optionally broadcast to all gated variables).
  Deterministic (no rng); thresholds computed on the called rows or on a
  separate ``stats`` row set (for generators gating expanded rows against
  the original columns' statistics). All intensity knobs are now
  CONSTRUCTOR arguments of the presets (``rate=`` for
  ``BernoulliSelector``/``ExactCountSelector``, ``quantile=`` for
  ``TopQuantileSelector``); ``__call__(X, y)`` takes no intensity kwarg,
  so an intensity sweep means constructing a new selector. The
  ``quantile`` is the direct quantile of the column (``quantile=0.75`` =
  everything at or above Q3). `AlwaysSelector` was added (the all-rows
  preset, the ``selector=None`` default of the causal generators).

## dev_0_0_14 (unreleased)

### Added

- **`CausalOutlierPropagationGenerator`**: New tabular generator that propagates outliers along causal-graph edges. First generator to rely on the causal-graph API.
- **Causal-graph utilities**: `validate_dag`, `topological_order`, and related helpers for working with causal graphs.
- **Complete tabular generator benchmark coverage**: All 16 tabular generators are now registered in the benchmark framework (up from 7). Added registrations for `HistogramSamplingGenerator`, `ZScoreSamplingGenerator`, `LowDensitySamplingGenerator`, `HypersphereSamplingGenerator`, `GaussianNoiseClassesGenerator`, `RandomShiftClassesGenerator`, `RandomSamplingClassesGenerator`, and `RandomSamplingTargetsGenerator`.

### Changed

- **Generators now output NumPy arrays** (breaking): Tabular and time-series generators return NumPy arrays instead of pandas DataFrames/Series. Input handling still accepts pandas DataFrames/Series.
- **Vectorized generators**: Replaced per-row/per-column `.iloc` loops with batched NumPy operations for improved performance.
- **Two-tier sampler API**: The `sampler` parameter is split into `within_distribution_sampler` and `out_of_distribution_sampler`.
- **Explicit `column_mapping` parameter**: The causal-graph API now requires an explicit `column_mapping` parameter.
- **Benchmark outputs moved to `.benchmarks/`**: Results and baselines are now stored in `.benchmarks/results/` and `.benchmarks/baselines/` (gitignored) instead of `benchmarks/results/` and `benchmarks/baselines/`.

### Removed

- `node_to_index` parameter from the causal-graph API (replaced by `column_mapping`).

## dev_0_0_13 (released)

### Changed

- **Separated tests from benchmarks**: Generator correctness tests now live in `tests/generators/` as standalone pytest functions with fixtures, independent of the benchmark framework. The `benchmarks/checks/` directory and `FunctionalCheck` infrastructure have been removed. Benchmarks now focus exclusively on performance measurement (time/memory). (9 commits: `943e230`..`56fee5d`)

### Added

- Pytest fixtures for each data type: `tabular_small`/`tabular_large`, `time_series_sine`/`time_series_walk`, `graph_erdos_renyi`, `text_word_list`
- Test conftest files in `tests/generators/{tabular_data,time_series,graph,text}/`

### Removed

- `benchmarks/checks/` directory and all `FunctionalCheck` infrastructure
- `--type functional` option from benchmark CLI
- `functional_checks` parameter from `GeneratorBenchmark` model