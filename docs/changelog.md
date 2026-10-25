# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## dev_0_0_15 (unreleased)

### Breaking changes

- **Selectors work on column positions only (breaking).**
  Every selector preset now takes integer column positions at construction
  and returns a ``dict[int, spec]`` keyed by position: ``AlwaysSelector``,
  ``BernoulliSelector``, ``ExactCountSelector``, ``TopQuantileSelector``,
  ``ClassSelector``, ``ProportionalSelector``. Variable names and the
  selector-side ``column_mapping`` are gone; read columns (``class_column``,
  ``column``, ``driver``) are independent int positions. Value-free presets
  return firing decisions (bool masks; ``AlwaysSelector``'s constant
  scalar ``1.0`` realizes to all-true); value-driven presets return ``[0, 1]``
  firing specs (float scalar or ``(n,)`` array) that generators realize with
    the single rule ``rng.random(n) < spec``. Direct generators no longer take
    ``column_mapping`` — ``AdditiveNoiseGenerator.generate(X, distribution,
    noise_std=1.0, *, selector=None)``. Causal generators' call-time targets
    are owned by the selector itself: its ``columns`` (column positions) are
    the intervention targets, so ``perturbation_nodes`` is gone from their
    signatures.
  - **Wave 2: the ``fired()`` field unification and selector-owned causal targets.**
    The three realization helpers collapse into one —
    ``badgers.core.selectors.fired(X, selector, rng)`` returns the narrow bool
    field ``(n_rows, n_targets)`` (column ``i`` ↔ the selector's ``i``-th
    target); row-level union is ``field.any(axis=1)`` (``fired_mask`` and
    ``fired_rows`` are deleted). All causal generators now require a selector
    at call time — ``perturbation_nodes`` is removed from their signatures
    (``CausalImbalanceGenerator``'s legacy ``perturbation_nodes`` +
    ``quantile_range`` fallback is the one exception). Drift's ``drift_std``
    is replaced by ``shift_amount`` (deterministic signed shift, scalar or
    dict keyed by column position), and ``CausalNoiseGenerator.noise_std`` is
    scalar-or-dict-by-position as well. ``CausalOutlierPropagationGenerator``
    now requires a selector (per-cell pinning: firing masks a single cell per
    selected row, pure-bootstrap rows are labeled ``False``, and rows sharing
    a firing pattern are perturbed as one group).
- **Wave 3: the outliers generators consume selectors; outlier rows are seeded from real rows.**
  All six tabular outliers generators take a selector and realize it through the
  shared firing field. ``CausalOutlierPropagationGenerator.generate(X, selector,
  *, out_of_distribution_sampler=None)``: the selector is required and fires on
  ``X`` — the fired rows are appended as COPIES (an unfired cell keeps the seed
  row's value; descendants of fired targets recompute through the DGP; labels are
  all ``True``); ``n_outliers`` and ``within_distribution_sampler`` are removed
  (``CountSelector([x], count=10)`` replaces ``n_outliers=10``). The direct
  generators (``HyperCubeSampling``, ``ZScoreSamplingGenerator``,
  ``HypersphereSamplingGenerator``) make selectors the only firing mechanism:
  ``n_outliers`` is only the DEFAULT count (``selector=None`` builds an internal
  ``CountSelector``, default 10), passing both raises, and appended rows are
  copies of fired rows with fired cells overwritten by the same samplers as
  before (same-seed values shift: the selector machinery consumes draws first).
  The low-density trio (``IndependentHistogramsGenerator``,
  ``HistogramSamplingGenerator``, ``LowDensitySamplingGenerator``) uses selector
  KEYS as the synthesized columns (mask discarded) and fills unselected columns
  with per-cell bootstrap copies. ``UniformInstanceAttributeSampling`` accepts
  ``selector`` for API uniformity but ignores it (warning; ``n_outliers`` stays
  its count). Everywhere in the family a requested count is capped at the
  dataset size (``count > len(X)`` raises), 0 is a no-op and a negative count
  raises.
- **The `(X, y)` contract is removed; generators and selectors take `X` only (breaking).**
  Every generator's signature is `generate(X, **params)` and it returns exactly one
  generated object `Xt` — the old `y` parameter and the two-tuple return are gone
  (a hard break, no deprecation shims). Every selector is called as `selector(X)` and
  returns the per-column ``dict[int, spec]`` mapping (positions keyed; bool masks
  from value-free presets, ``[0, 1]`` firing specs from value-driven presets).
  `Pipeline.generate(X, params=None)`
  threads each single-output step into the next. The `preprocess_inputs` decorators
  wrap `generate(self, X, **kwargs)`. Benchmark scenario factories return one `X`.
  Target-like data lives inside `X`: dedicated columns for tabular and time-series data
  (class columns read via the selector's `class_column` int position), node/edge attributes for
  graphs, and modality-native values for text. Deficit labels remain available through
  the generators' own getters (`get_drift_labels()`, `get_missing_labels()`,
  `get_imbalance_labels()`, `get_noise_labels()`, `get_outliers_labels()`).
  Supported Python versions are unchanged (3.11–3.14).
- **Causal generators: labels are internal generator state (breaking).**
  The causal defect generators
  (``CausalDriftGenerator``, ``CausalMissingnessGenerator``,
  ``CausalImbalanceGenerator``, ``CausalNoiseGenerator``,
  ``CausalAdditiveNoiseGenerator``) no
  longer return deficit labels: ``generate(X, **params)`` returns
  exactly one generated object ``Xt`` and each generator exposes its own boolean label getter
  aligned with the rows of the last generated output —
  (``get_drift_labels()``, ``get_missing_labels()``,
  ``get_imbalance_labels()``, ``get_noise_labels()`` (both noise
  generators), ``get_outliers_labels()``. Calling a getter before ``generate()``
  raises ``RuntimeError``. The base class owns the storage slot
  (``deficit_labels_``); each generator owns its getter.
- **`CausalGenerator._validate` replaced by two raise-only helpers.**
  ``_validate(X, perturbation_nodes)`` (which mixed validation with
  graph-derivative computation and returned a 5-tuple) is split into
  ``_validate_structure(X)`` (graph/DAG, column count, column_mapping
  coverage and indices) and ``_validate_perturbation_nodes(nodes)``
  (non-empty list of known nodes). Neither returns anything — the
  mapping is ``self.column_mapping``. The dead ``order`` member and the
  graph algebra (descendants/exogenous/perturb_cols) moved into the only
  generator that used them (`CausalOutlierPropagationGenerator`).
- **`PyMCDGP` and the old `badgers.core.dgp` module are removed.** The causal
  defect generators now fit a **per-node data generation process** behind the
  new `DGP` protocol (`badgers.core.dgp` is a package). Backends are
  injected as objects (`dgp=...` / `use_dgp`); PyMC-specific parameters
  (`n_samples_mcmc`, `fit_dgp`) no longer appear in `fit(...)`.
- **`CausalGenerator.fit` removed AND the `graph=`/`column_mapping=`
  generator-side structure is gone (breaking).** The DGP is the ONLY structure
  a causal generator accepts: the constructor is `CausalGenerator(random_generator=..., dgp=None)`
  and the only structure param anywhere on the generator is `dgp=` (or
  `use_dgp(dgp)`). The DGP carries `graph` and `column_mapping` itself —
  build one with e.g. `dgp = GcmSCMDGP(scm).fit(X)` (see
  `docs/tutorials/dev/Using-DGPs.ipynb`). Unfitted DGPs are fitted lazily on first
  `generate(X)` (`DGP.is_fitted` is respected).
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
  via the DGP (`dgp.intervene` — Pearl surgery). Row firing is delegated
  to a `selector` (default: every row fires); strictly in-place (no
  `n_copies`).
- **`GaussianNoiseGenerator` removed (breaking).** It is a strict special
  case of the new `AdditiveNoiseGenerator`: replace
  `GaussianNoiseGenerator().generate(X, noise_std=0.5)` with
  `AdditiveNoiseGenerator().generate(X, distribution=scipy.stats.norm(scale=0.5))`.
  (Note the semantic shift: the old class standardized first, so its
  `noise_std` was column-relative; the new API adds the frozen
  distribution's raw draws — pass the per-column scale yourself.)
  `GaussianNoiseClassesGenerator` (per-class noise) remains; the
  time-series `Local/GlobalGaussianNoiseGenerator` are unrelated and
  unchanged.
- **`CausalEdgePerturbationGenerator` is removed for this release
  (deferred to future work).** The relationship-level (edge weight)
  defect — scaling one edge coefficient and recomputing the child
  marginal-preserving — was developed behind the family API but is not
  part of the 0.0.15 surface: the generator, its tests and the
  "Perturbing Edge Strength" tutorial are removed. The underlying
  `edits.scale_edge` DGP edit remains in `badgers.core.dgp.edits` (and
  is tested), so re-introducing the generator is a small follow-up.
- **`CausalImbalanceGenerator` no longer requires a DGP** (breaking): the
  generator is a pure row-subset (quantile-range truncation); the kept
  rows' parents are unchanged, so no structural equation is re-evaluated.
  The dead `mechanism`/`mechanism_cause`/`mechanism_rate` parameters, the
  on-demand OLS fit, and the no-op propagate pass are removed — the
  signature is now
  `generate(X, perturbation_nodes=None, quantile_range=(0.5, 1.0))`.
  This is the first generator in the causal family whose `generate` needs
  neither a DGP nor any random numbers (selection-side only).
- **`CausalMissingnessGenerator` is chooser-only (breaking)**: missingness is
  an observation-layer defect — it only hides cells (writes NaN) and does
  NOT cascade to descendants. The `missingness_rule` mechanism parameter,
  the on-demand DGP fit, and the `any`/`all` NaN-propagation passes are
  removed; the fire mask comes from a `selector` object (see the selector
  bullet below). Signature:
  `generate(X, perturbation_nodes=None, selector=None)`.
- **String `mechanism` flags replaced by `selector` objects (breaking).**
  The row-firing parameters of the causal defect generators
  (`CausalDriftGenerator`, `CausalMissingnessGenerator`,
  `CausalNoiseGenerator`, `CausalAdditiveNoiseGenerator`) are now a single
  `selector=` parameter holding a callable from
  `badgers.core.selectors`; `None` defaults to "every row fires"
  (`AlwaysSelector` semantics). The `mechanism`/`mechanism_cause`/
  `mechanism_rate` strings and the `_fire_masks` string dispatcher are
  removed. Migration (`pos...` = the perturbation nodes' column
  positions from the DGP's `column_mapping`):
  - `mechanism="always"` → omit `selector` (or `AlwaysSelector([pos...])`).
  - `mechanism="random", mechanism_rate=r` →
    `selector=BernoulliSelector([pos...], rate=r,
    rng=gen.random_generator)`.
  - `mechanism="self", mechanism_rate=r` →
    `selector=TopQuantileSelector([pos...], column=pos, quantile=1-r)`
    (value-driven presets read the one `pos` and broadcast it).
  - `mechanism="observed", mechanism_cause=c, mechanism_rate=r` →
    `selector=TopQuantileSelector([pos...], column=pos, quantile=1-r,
    driver=pos_c)` with `pos_c` the cause column's position from the
    DGP's `column_mapping`.
  Intensity lives on the selector object: `rate` and `quantile` are now
  CONSTRUCTOR arguments of `BernoulliSelector`/`ExactCountSelector`/
  `TopQuantileSelector` (previously `rate` was a `__call__` kwarg and
  `quantile` a call kwarg) — so an intensity sweep means constructing a
  new selector. `AlwaysSelector` was added (the all-rows preset).
- **`RandomSamplingClassesGenerator` takes a `selector` (breaking).** The
  `proportion_classes` call kwarg is removed; pass instead
  `selector=ClassSelector([pos], class_column=pos, proportions={...},
  rng=...)` from `badgers.core.selectors`, with `pos` the class column's
  int position in `X` and `proportions` = `{class_value: proportion}`.
  Class columns are columns of `X`:
  selection reads only `X` (the class column is the selector's
  `class_column` int position). The selector's fire mask is the standard
  contract (True = deficit); the generator's deficit is DROPPING the fired
  rows (subsampling), so the output is smaller than the input. Classes
  absent from `proportions` are kept as-is; proportions are shares of the
  total row count, clipped to the rows available in each class. The class
  variable must be a column of `X`, given as the selector's
  `class_column` int position. `selector=None` is a
  no-op. Migration: `generate(X, proportion_classes={0: 0.5, ...})` →
  `generate(np.column_stack([X, y]), selector=ClassSelector([pos],
  class_column=pos, proportions={0: 0.5, ...}, rng=rng))` with `pos`
  the position of the stacked target column. Note the new
  semantics sample WITHOUT replacement (the old
  implementation duplicated rows when a proportion exceeded the class
  size).
- **Outlier labels are internal generator state, no longer returned by
  ``generate`` (breaking).** All outlier generators
  (`HyperCubeSampling`, `ZScoreSamplingGenerator`,
  `HypersphereSamplingGenerator`, `UniformInstanceAttributeSampling`,
  `IndependentHistogramsGenerator`, `HistogramSamplingGenerator`,
  `LowDensitySamplingGenerator`, `CausalOutlierPropagationGenerator`,
  and `StructuredOutlierGenerator`) now expose
  ``get_outliers_labels()``: a boolean vector aligned with the rows of
  the LAST generated output, ``True`` where the row is an outlier,
  ``False`` where it is an original row. ``generate(X, **params)``
  returns exactly one generated object ``Xt`` (the previous
  ``"original"``/``"outliers"`` string arrays are gone). Calling
  ``get_outliers_labels()`` before ``generate()`` raises
  ``RuntimeError``. Migration: the old two-tuple unpacking from
  ``gen.generate(X)`` followed by string-comparing the second element
  → ``Xt = gen.generate(X)`` followed by ``gen.get_outliers_labels()``.
- **Selector scores may now be fractional (additive contract extension).**
  A selector's per-row value may be a float in ``[0, 1]`` — the
  PROBABILITY that the deficit is applied — where boolean masks remain
  the degenerate (deterministic) case. Realization belongs to the
  consuming generator (a modifying consumer fires Bernoulli-wise, a
  resampling consumer uses the scores as selection weights); a preset's
  own rng only draws its spec (see the index-contract bullet above). The
  ``ProportionalSelector([pos], column=pos)`` preset returns
  ``normalize_proba(X[:, pos])`` (the deficit probability proportional to
  a column's value).
- **`RandomSamplingTargetsGenerator` takes a `selector` (breaking).** The
  `sampling_proba_func(y)` kwarg is removed; the generator now resamples
  rows weighted by ``1 - deficit_probability`` from a selector (noisy-OR
  combined over its variables), i.e. rows with a high deficit probability
  are dropped/under-sampled, rows with a low deficit probability are kept
  or replicated. Signature:
  `generate(X, *, selector=None, replace=True, size=None)`;
  `replace=True` (default) may duplicate rows (importance resampling),
  `replace=False` gives a weighted thinning, `size` controls the output
  row count (default: input row count). Selection reads only `X`
  (per-column specs keyed by the columns' int positions).
  Migration: the old default
  (sample ∝ target values) becomes
  `ProportionalSelector(idx, column=idx)` over whichever `X` column
  holds the signal (e.g. the stacked target column).
- **`RandomSamplingFeaturesGenerator` removed (breaking).** It was the
  free-probability twin of the resampler; its value-weighted resample is
  now `RandomSamplingTargetsGenerator(selector=ProportionalSelector(
  idx, column=idx))`, and its per-class behavior is the same
  generator with a boolean selector (`ClassSelector` over the class
  column).
- **`RandomShiftClassesGenerator` takes a `selector` (breaking).** The
  class variable is read from a column of `X` (the selector's
  `class_column` int position). Signature:
  `generate(X, *, selector=None, shift_std=0.1)`. The selector is
  REQUIRED (`selector=None` raises) — a per-class drift has no class-free
  default; fire masks are OR-ed. `shift_std` is now a float (one
  independent draw magnitude per class value) or a dict
  `{class_value: std}`; entries are SIGNED (`|std|` = draw scale,
  sign = direction). The per-dimension 2D-array form is removed
  (per-class shifts are isotropic — scale the data first for anisotropic
  drift). Class values absent from a dict are skipped. Migration:
  `generate(X, shift_std=np.array([2., 2., 2., 2.]))` →
  `generate(np.column_stack([X, y]),
  selector=ClassSelector([pos], class_column=pos,
  proportions={c: 1.0 for c in classes}, rng=rng), shift_std=2.0)`
  with `pos` the position of the stacked target column.
- **Drift generators: signed per-column ``shift_std`` and class-column
  protection (breaking, ``RandomShiftGenerator``/
  ``RandomShiftClassesGenerator``).** ``shift_std`` entries are now
  signed: the magnitude (``|shift_std|``) is the scale of the normal draw
  and the sign is the shift direction — a negative entry drifts the
  column downward. For ``RandomShiftClassesGenerator`` the class column
  is identification metadata and is NEVER shifted by default (shifting
  it would move rows away from their class values); new parameters
  ``variables`` (columns to shift) and ``class_column`` (explicit class
  column position) allow shifting an explicit subset and/or a
  different class column. A dict ``shift_std={class_value: std}`` now
  requires a resolvable class column (raises otherwise).
- **Deterministic drift generators (new):** ``FixedShiftGenerator`` and
  ``FixedShiftClassesGenerator`` — the non-random counterparts of the
  random shift generators. Each column is translated by exactly
  ``shift_std`` (in standard-deviation units, data standardized first;
  sign = direction); no random number is drawn, so repeated calls with
  the same input return the same output. Both are exported from
  ``badgers.generators.tabular_data.drift``.
- **`GaussianNoiseClassesGenerator` takes a `selector` (breaking).** Same
  contract as above: classes are a column of `X`, given as the
  selector's `class_column` int position (`selector=None` raises; a
  per-class lookup requires a selector exposing `class_column` to
  resolve the class column). The
  class column itself is never noised — the old implementation rebuilt
  the output's last column as class labels, silently corrupting it (bug
  fixed). Signature: `generate(X, *, selector=None,
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
  and existing-edge checks), `edits.exogenize(dgp, node)`
  (INTERVENTION-ONLY semantics — see below), `edits.set_form(dgp, node,
  response_fn, draw_noise=None)` (Q2: an optional noise-law closure).
  Edits return read-only views and never
  mutate the base DGP; they compose by delegation.

### Added

- **Exact-count selector presets**: ``CountSelector(columns, count, rng)`` —
  exactly ``count`` distinct rows with one shared mask under every target
  (``count > len(X)`` raises); ``ClassCountSelector(columns, class_column,
  counts, rng)`` — exact row counts per class value (unknown class values and
  per-class overflow raise).
- **New within-distribution sampler ``InstanceSampler``** — per-cell bootstrap
  (each cell copies the value of a random real row); extracted from
  ``UniformInstanceAttributeSampling``, which now delegates to it
  (bit-identical values).

- **New DGP backend `GcmSCMDGP` (dowhy extra)**: plug an externally built
  `dowhy.gcm.StructuralCausalModel` into badgers — mechanism fitting is
  delegated to `gcm.fit`, sampling and interventions run on the generic
  protocol with caller-provided rngs. Supports `AdditiveNoiseModel`
  (non-root), `DiscreteAdditiveNoiseModel` (non-root count nodes —
  discrete predictions are rounded so the composition stays integral),
  `EmpiricalDistribution` (root) and `ClassifierFCM` (non-root discrete
  nodes with numeric-encoded classes). Cascade composition is delegated
  to the mechanism's own `evaluate(parents, noise)` via the DGP's
  `structural` hook — badgers draws the noise, dowhy owns the math.
- **New DGP backend `PyMCSCMDGP` (pymc extra)**: plug a user-built PyMC
  causal model into badgers — structural nodes are observed RVs whose
  distribution parameters carry the structural functions; mechanism
  fitting is one joint `pm.sample` (full posterior kept), sampling and
  interventions run on the generic protocol with caller-provided rngs
  and ONE coherent posterior draw per cascade. Supported families:
  `Normal`, `StudentT`, `Laplace` (additive) and `Bernoulli`, `Poisson`,
  `Binomial`, `Categorical` (numeric-encoded discrete).
- **`ClassSelector` preset** (`badgers.core.selectors`): per-class
  fixed-count sampling on a discrete column — `ClassSelector([pos],
  class_column=pos, proportions={...}, rng=...)` with `pos` the class
  column's int position and `proportions` = `{class_value: proportion}`.
  Discovers the distinct
  values from the data (no manual class enumeration), raises
  `ValueError` on class values absent from the column (typo protection),
  and fires exactly `min(int(prop * n_rows), n_rows_in_class)` rows per
  listed class (partial mappings are legitimate: unlisted classes are
  kept). The class column is a column of `X`, given as the
  `class_column` int position
  (selection reads only `X`). First
  subsampling consumer: `RandomSamplingClassesGenerator`.

- **Per-node DGP protocol** (`badgers.core.dgp` package): `DGP` protocol and
  `DGPMixin` (topological `fit`/`sample` walk) — backends implement exactly
  one operation, `fit_node(node, parents, data)`, and expose
  `response(node, parent_values)` and `draw_noise(node, rng, n)`;
  `sample(rng, n)` is tier 2 and no generator depends on it.
  Cascades compose each node through the `structural(node,
  parent_values, rng, n)` hook (default: the additive composition
  `response + draw_noise`; backends with non-additive mechanisms
  override it — `GcmSCMDGP` delegates to dowhy's `evaluate`).
- **`pick_parameters` hook** (`DGPMixin`): called once at the top of the
  generic `sample`/`intervene` cascades; returns the DGP instance whose
  mechanisms drive the batch (ONE coherent parameter set per call).
  Default: `self` (deterministic backends fit exactly one parameter set).
  `PyMCSCMDGP` overrides it to pin ONE coherent posterior draw from the
  caller's rng
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
  forward pass is replaced by a fitted DGP (`GcmSCMDGP` from a dowhy-gcm
  SCM) +
  `dgp.intervene(outliers, perturbation_nodes, rng=...)` — uniform Pearl
  surgery semantics with the other causal generators. Behavior note: a
  degenerate zero-residual fitted mechanism now contributes no noise (the
  old `0.1*col_std` fallback is gone).
- **Causal defect family** (four generators sharing `CausalGenerator`, an
  explicit `fit(X, graph, column_mapping)` phase followed by `generate(...)`
  with `selector=` row firing):
  `CausalDriftGenerator` (single systematic shift, recomputed downstream),
  `CausalNoiseGenerator` (per-edge noise injection, `noise_std`),
  `CausalImbalanceGenerator` (quantile-range truncation (pure subset,
  descendants untouched) or explicit `selector=` — fired rows dropped), `CausalMissingnessGenerator` (m-graph missingness
  with `any`/`all` propagation, optional observation-graph derivation).
- **`pymc` optional extra** (renamed from `causal`; `pip install
  "badgers[pymc]"`) — only needed for building the PyMC SCM passed to
  `PyMCSCMDGP`.

- **`TopQuantileSelector`** preset (`badgers.core.selectors`): fires rows at
  or above the ``quantile`` of a column — self mode (own column) or driver
  mode (a driver column, optionally broadcast to all selected variables).
  Deterministic (no rng); thresholds computed on the called rows or on a
  separate ``stats`` row set (for generators firing expanded rows against
  the original columns' statistics). All intensity parameters are now
  CONSTRUCTOR arguments of the presets (``rate=`` for
  ``BernoulliSelector``/``ExactCountSelector``, ``quantile=`` for
  ``TopQuantileSelector``); ``__call__(X)`` takes no intensity kwarg,
  so an intensity sweep means constructing a new selector. The
  ``quantile`` is the direct quantile of the column (``quantile=0.75`` =
  everything at or above Q3). `AlwaysSelector` was added (the all-rows
  preset, the ``selector=None`` default of the causal generators).

- **`CausalImbalanceGenerator` now accepts an explicit `selector=`**
  (additive; the old `quantile_range` rule remains the `selector=None`
  default): the rows FIRED by the selector are dropped (subsampling
  polarity, union over the selector's variables);
  `perturbation_nodes`/`quantile_range` are ignored when a selector is
  given, and `perturbation_nodes` is no longer required in selector
  mode.
- **`DecompositionAndOutlierGenerator` renamed to
  `StructuredOutlierGenerator`** (old name kept as a backwards-compatible
  alias), with a new optional **`preprocessor=`** constructor argument:
  any sklearn pipeline-like object with `transform`/`inverse_transform`
  can replace the built-in standardization + decomposition —
  preprocessing is no longer hardcoded inside the generator.

### Removed

- **Seaborn is no longer a dependency of the ``dev`` extra.** All
  tutorial/dev notebooks use plain matplotlib for visualizations.
- **`OlsDGP`, `DowhyGcmDGP` and `PyMCDGP` DGP backends are removed.**
  The only DGP backends are now the two SCM adapters
  (`badgers.core.dgp.backends.GcmSCMDGP` and `PyMCSCMDGP`): build the
  structural causal model with the wrapped library (dowhy-gcm or PyMC),
  then wrap it — `GcmSCMDGP(scm)` / `PyMCSCMDGP(pymc_model)`. See
  `docs/tutorials/dev/Using-DGPs.ipynb`.
- **`edits.rebuild` and `edits.reversed_graph` are removed** (and with
  them `CausalGenerator.rebuild_with_graph`). Re-estimation after a
  structural modification is out of scope: build the modified SCM with
  the wrapped library's own API, fit it, and wrap it again with the
  appropriate adapter.

### Changed

- **Direct outliers generators delegate to the OOD samplers
  (internal).** ``ZScoreSamplingGenerator``,
  ``HypersphereSamplingGenerator`` and ``HyperCubeSampling`` no longer
  inline their value production; they delegate to ``ZScoreSampler``,
  ``HypersphereSampler`` and ``UniformOutOfDistributionSampler`` from
  ``badgers.core.sampling``, making it the single source of
  out-of-distribution sampling math. ``UniformOutOfDistributionSampler``
  now accepts ``expansion=0`` (the exact data bounding box; previously
  it required ``expansion > 0``). One semantic behavior change: for degenerate
  (constant) columns, outliers are now pinned at exactly the constant
  value (bounding-box semantics of a zero range), where the old
  ``MinMaxScaler`` round-trip spread them around/above the constant
  (e.g. constant 5.0: ``expansion=0.1`` → [4.949, 5.936];
  ``expansion=0`` → [5.041, 5.863], above the data max). Same seeds
  produce the same outliers up to floating-point ordering (z-score and
  hypersphere identical within ±1 ulp of the old scaler-based path; the
  hypercube's uniform draw is unchanged, values may shift by ±1 ulp on
  the new arithmetic path).
  Public `generate` signatures are unchanged.
- **Causal generators require an explicit DGP (hard error when
  missing).** There is no implicit default backend: every causal defect
  generator raises when no `dgp=` is supplied to the
  `CausalGenerator` constructor. Build a DGP from the library of choice
  (dowhy-gcm or PyMC) and wrap it with `GcmSCMDGP`/`PyMCSCMDGP`.
- Drift tutorials and dev notebooks: extensive fixes — collapsed summary
  tables rebuilt, garbled prose fixed, one seeded rng per notebook
  (``gen = Generator()`` naming, ``seed = 0``, no global ``np.random``),
  verification cells asserting the documented behavior (noise variance,
  constant drift shift, edge correlation/std preservation, class counts,
  NaN counts, repeat/drop lengths), and causal-outlier plots using
  ``get_outliers_labels()``.

### Documentation

- **New "Understanding badgers" concepts notebook**
  (`tutorials/concepts/Badgers-Concepts.ipynb`): a step-by-step
  introduction — module structure, generator instantiation with an rng,
  `generate()`, shape rules (appending / firing / subsampling), the
  in-place mutation contract ("pass `X.copy()`"), the X-only contract
  (target-like data as columns of `X`), selectors, DGP backends and
  pipelines.
- **Tabular tutorials merged per deficit** (one notebook per deficit
  instead of causal/non-causal pairs): `Outliers.ipynb`, `Noise.ipynb`,
  `Missingness.ipynb`, `Imbalance.ipynb`, `Drift.ipynb`.
  Each has **Part A** (acting on the
  observation process — no DGP) and **Part B** (acting on the DGP).
  Visualizations use explicit `cls` class vectors and `labels` deficit
  vectors (no `y`/`yt` placeholders); the outlier tutorial was
  substantially shortened (the causal part no longer repeats the
  concepts notebook).

## dev_0_0_14 (released)

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