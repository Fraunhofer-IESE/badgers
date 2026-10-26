# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## Unreleased

### Added

- **`UndersamplingGenerator`** (in `badgers.generators.tabular_data.imbalance`)
  — a row-removing imbalance generator: `generate(X, selector=None)` removes
  the rows flagged by a deficit `Selector` (`True`/`1.0` = row removed;
  fractional specs are removed stochastically; multiple columns = union;
  `selector=None` is a no-op). `get_imbalance_labels()` returns a boolean
  array aligned with the input rows (`True` = row removed).
- **`OversamplingGenerator`** (in `badgers.generators.tabular_data.imbalance`)
  — a row-duplicating imbalance generator: `generate(X, selector=None)`
  duplicates rows according to per-row integer multiplicities
  (`ConstantMultiplicity(k)`, `MaskedMultiplicity(mask, k)`, or a bare int /
  `(n,)` int array; all values must be `>= 1`). The output is a superset of
  the input — no row is ever dropped and the original order is preserved.
  `get_imbalance_labels()` returns a boolean array aligned with the input
  rows (`True` = row duplicated, i.e. multiplicity > 1).
- **`ConstantMultiplicity(k)`** — a `MultiplicitiesSelector` that assigns
  every row multiplicity `k`.
- **`MaskedMultiplicity(mask, k)`** — a `MultiplicitiesSelector` that assigns
  multiplicity `k` to the rows where `mask` is `True` (a `(n,)` bool array or
  a callable on `X`) and multiplicity 1 to the rest.
- **`OutlierSamplingGenerator`** (in
  `badgers.generators.tabular_data.outliers.sampling`) — the generic
  non-causal outlier generator: `generate(X, selector, *,
  out_of_distribution_sampler="zscore")` appends rows that are copies of the
  selector's seed rows with the selector-selected cells replaced by
  out-of-distribution values (all other cells keep the seed row's values).
  The `selector` is REQUIRED — it owns the outlier count and the perturbed
  targets (e.g. `CountSelector([0, 1], count=20, rng=...)`).
  `get_outliers_labels()` returns a boolean array aligned with the OUTPUT
  rows (`True` = appended outlier row).
- **`KernelDensitySampler`** (in `badgers.core.sampling`, registry name
  `"low_density"`) — a KDE-based out-of-distribution sampler preserved from
  the removed `LowDensitySamplingGenerator`. Rewritten as a grid-CDF
  construction (no rejection): it scores a bounded regular grid of at most
  ~`grid_budget` cells (hard cap of 65536 cells; `NotImplementedError` if
  fewer than 2 cells per dimension is feasible) with an sklearn
  `KernelDensity` (Scott bandwidth, `ball_tree`), then draws exactly
  `n_samples` points from a categorical over the grid cells at or below the
  `threshold_low_density` percentile of the data log-densities, each point
  uniform inside its cell (fallback: the least-dense cells). The new
  `grid_budget` parameter replaces the earlier `max_samples`; the count is
  always exactly `n_samples` and deterministic for a fixed `rng` seed.
- **`TreeCellSampler`** (in `badgers.core.sampling`, registry name `"tree"`)
  — an adaptive-histogram out-of-distribution sampler (no KDE): recursive
  median splits partition the standardized columns into kd cells holding at
  most `leaf_size` points (fine resolution in dense regions, coarse in
  sparse ones); every cell is extended axis-aligned to the midplanes with
  the neighbouring points (an axis-aligned Voronoi approximation), and
  exactly `n_samples` points are drawn from a categorical over the cells
  whose density (count/volume) is at most `threshold_low_density` times the
  global mean density (fallback: the least-dense 10 cells). Deterministic
  for a fixed `rng` seed.

### Changed

- **The imbalance API is split into under- and oversampling (breaking).**
  Imbalance is no longer a single score-driven resampler. Deficit-based row
  removal lives in `UndersamplingGenerator` (a deficit `Selector` decides
  which rows are dropped); row duplication lives in
  `OversamplingGenerator` (a per-row multiplicity selector decides how many
  copies of each row survive). Oversampling input validation rejects
  fractional values in `(0, 1)` ("looks like a deficit Selector"), zeros
  ("row dropped"), negative values ("must be >= 1"), and non-integer
  multiplicities ("not supported").
- **The non-causal outlier generators collapsed into a single generic
  `OutlierSamplingGenerator` (breaking).** It mirrors the causal
  `CausalOutlierPropagationGenerator`'s contract: a REQUIRED `selector`
  (owns the outlier count and the perturbed targets) plus a pluggable
  `out_of_distribution_sampler` (an `OutOfDistributionSampler` instance from
  `badgers.core.sampling`, or its registry name as a string, default
  `"zscore"`). The former strategy-specific generators are now samplers:
  `ZScoreSampler` (`"zscore"`), `HypersphereSampler` (`"hypersphere"`),
  `UniformOutOfDistributionSampler` (`"uniform"`),
  `IndependentHistogramSampler` (`"independent_histogram"`),
  `JointHistogramSampler` (`"histogram"`), and `KernelDensitySampler`
  (`"low_density"`).

### Removed

- **`RandomSamplingClassesGenerator`** — replaced by
  `UndersamplingGenerator` (with a `ClassSelector`) and
  `OversamplingGenerator` (with a `MaskedMultiplicity`).
- **`RandomSamplingTargetsGenerator`** — replaced by
  `UndersamplingGenerator` (score-driven deficit selectors such as
  `ProportionalSelector`) and `OversamplingGenerator`.
- **`CausalImbalanceGenerator`** — removed from
  `badgers.generators.tabular_data.causal`; causal no longer ships an
  imbalance generator (row removal/duplication is an observation-process
  operation in `badgers.generators.tabular_data.imbalance`).
- **`HyperCubeSampling`** — replaced by
  `OutlierSamplingGenerator` with `UniformOutOfDistributionSampler(expansion)`
  (registry name `"uniform"`).
- **`ZScoreSamplingGenerator`** — replaced by
  `OutlierSamplingGenerator` with `ZScoreSampler(scale)`
  (registry name `"zscore"`).
- **`HypersphereSamplingGenerator`** — replaced by
  `OutlierSamplingGenerator` with `HypersphereSampler(scale)`
  (registry name `"hypersphere"`).
- **`IndependentHistogramsGenerator`** — replaced by
  `OutlierSamplingGenerator` with `IndependentHistogramSampler(bins)`
  (registry name `"independent_histogram"`).
- **`HistogramSamplingGenerator`** — replaced by
  `OutlierSamplingGenerator` with
  `JointHistogramSampler(bins, threshold_low_density)`
  (registry name `"histogram"`).
- **`LowDensitySamplingGenerator`** — replaced by
  `OutlierSamplingGenerator` with
  `KernelDensitySampler(threshold_low_density, grid_budget)`
  (registry name `"low_density"`).

  **No backward compatibility** is provided for any of these classes.
  See the new [Imbalance tutorial](tutorials/tabular-data/Imbalance.md)
  for the replacement patterns, and the
  [Outliers tutorial](tutorials/tabular-data/Outliers/) for the
  `OutlierSamplingGenerator` + sampler patterns.

## dev_0_0_15 (unreleased)

### Breaking changes

- **Selectors work on column positions only (breaking).**
  Every selector preset now takes integer column positions at construction
  and returns a ``dict[int, spec]`` keyed by position: ``AlwaysSelector``,
  ``BernoulliSelector``, ``ExactCountSelector``, ``TopQuantileSelector``,
  ``ClassSelector``, ``ProportionalSelector``. Variable names and the
  selector-side ``column_mapping`` are gone; read columns (``class_column``,
  ``column``, ``driver``) are independent int positions. Value-free presets
  return selection decisions (bool masks; ``AlwaysSelector``'s constant
  scalar ``1.0`` realizes to all-true); value-driven presets return ``[0, 1]``
  selection specs (float scalar or ``(n,)`` array) that generators realize with
    the single rule ``rng.random(n) < spec``. Direct generators no longer take
    ``column_mapping`` — ``AdditiveNoiseGenerator.generate(X, distribution,
    noise_std=1.0, *, selector=None)``. Causal generators' call-time targets
    are owned by the selector itself: its ``columns`` (column positions) are
    the intervention targets, so ``perturbation_nodes`` is gone from their
    signatures.
  - **Wave 2: the ``get_selected_fields()`` field unification and selector-owned causal targets.**
    The three realization helpers collapse into one —
    ``badgers.core.selectors.get_selected_fields(X, selector, rng)`` returns the narrow bool
    field ``(n_rows, n_targets)`` (column ``i`` ↔ the selector's ``i``-th
    target); row-level union is ``field.any(axis=1)`` (``fired_mask`` and
    ``fired_rows`` are deleted). All causal generators now require a selector
    at call time — ``perturbation_nodes`` is removed from their signatures
    (``CausalImbalanceGenerator``'s legacy ``perturbation_nodes`` +
    ``quantile_range`` fallback is the one exception). Drift's ``drift_std``
    is replaced by ``shift_amount`` (deterministic signed shift, scalar or
    dict keyed by column position), and ``CausalNoiseGenerator.noise_std`` is
    scalar-or-dict-by-position as well. ``CausalOutlierPropagationGenerator``
    now requires a selector (per-cell pinning: the selection pins a single cell per
    selected row, pure-bootstrap rows are labeled ``False``, and rows sharing
    a selection pattern are perturbed as one group).
- **Wave 3: the outliers generators consume selectors; outlier rows are seeded from real rows.**
  All six tabular outliers generators take a selector and realize it through the
  shared selection field. ``CausalOutlierPropagationGenerator.generate(X, selector,
  *, out_of_distribution_sampler=None)``: the selector is required and realizes on
  ``X`` — the selected rows are appended as COPIES (an unselected cell keeps the seed
  row's value; descendants of selected targets recompute through the DGP; labels are
  all ``True``); ``n_outliers`` and ``within_distribution_sampler`` are removed
  (``CountSelector([x], count=10)`` replaces ``n_outliers=10``). The direct
  generators (``HyperCubeSampling``, ``ZScoreSamplingGenerator``,
  ``HypersphereSamplingGenerator``) make selectors the only selection mechanism:
  ``n_outliers`` is only the DEFAULT count (``selector=None`` builds an internal
  ``CountSelector``, default 10), passing both raises, and appended rows are
  copies of selected rows with selected cells overwritten by the same samplers as
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
  from value-free presets, ``[0, 1]`` selection specs from value-driven presets).
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
  via the DGP (`dgp.intervene` — Pearl surgery). Row selection is delegated
  to a `selector` (default: every row is selected); strictly in-place (no
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
  removed; the selection comes from a `selector` object (see the selector
  bullet below). Signature:
  `generate(X, perturbation_nodes=None, selector=None)`.
- **String `mechanism` flags replaced by `selector` objects (breaking).**
  The row-selection parameters of the causal defect generators
  (`CausalDriftGenerator`, `CausalMissingnessGenerator`,
  `CausalNoiseGenerator`, `CausalAdditiveNoiseGenerator`) are now a single
  `selector=` parameter holding a callable from
  `badgers.core.selectors`; `None` defaults to "every row is selected"
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
  `class_column` int position). The selector's selection is the standard
  contract (True = deficit); the generator's deficit is DROPPING the selected
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
  consuming generator (a modifying consumer is realized Bernoulli-wise, a
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
  default; selection masks are OR-ed. `shift_std` is now a float (one
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
  noise_std_per_class=dict())`; selected rows whose class value is absent
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
- **Every generator requires an explicit `random_generator` (breaking).**
  No generator constructor has a random-generator default anymore —
  `AdditiveNoiseGenerator()` now fails with ``TypeError``; pass one:
  `AdditiveNoiseGenerator(random_generator=numpy.random.default_rng(0))`.
  The old `default_rng(seed=0)` default was evaluated ONCE at import
  time, so all instances constructed without an explicit rng shared ONE
  Generator object (two generators created without an rng advanced each
  other's streams). Each generator now owns exactly the stream its user
  gave it. The two deterministic drift generators
  (`FixedShiftGenerator`, `FixedShiftClassesGenerator`) take an rng too
  (realizing the selector's specs consumes one; the shift itself still
  draws nothing) — repeated calls with the same input return the same
  output.
- **Causal generators require the selector as a positional parameter
  (breaking).** `CausalNoiseGenerator.generate(X, selector, ...)` and
  `CausalAdditiveNoiseGenerator.generate(X, selector, ...)` no longer
  default `selector` to ``None``: omitting it is a ``TypeError``, passing
  ``None`` explicitly raises ``ValueError`` (the selector owns the
  intervention targets). Same contract as the drift, missingness and
  outlier causal generators (which already declared it positionally);
  the one exception is `CausalImbalanceGenerator`'s legacy
  `selector=None` quantile-range path.

### Added

- **New OOD samplers `IndependentHistogramSampler` and `JointHistogramSampler`**
  (`badgers.core.sampling`, factory names ``"independent_histogram"`` and
  ``"histogram"``) — per-column and joint low-density histogram sampling,
  extracted from the outlier generators `IndependentHistogramsGenerator` and
  `HistogramSamplingGenerator`, which now delegate to them with bit-for-bit
  identical seeded synthesis and plug into the causal outlier generator as-is
  (`55502c4`, `caeb0e7`, `9054f30`).
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
  and selects exactly `min(int(prop * n_rows), n_rows_in_class)` rows per
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
  with `selector=` row selection):
  `CausalDriftGenerator` (single systematic shift, recomputed downstream),
  `CausalNoiseGenerator` (per-edge noise injection, `noise_std`),
  `CausalImbalanceGenerator` (quantile-range truncation (pure subset,
  descendants untouched) or explicit `selector=` — selected rows dropped), `CausalMissingnessGenerator` (m-graph missingness
  with `any`/`all` propagation, optional observation-graph derivation).
- **`pymc` optional extra** (renamed from `causal`; `pip install
  "badgers[pymc]"`) — only needed for building the PyMC SCM passed to
  `PyMCSCMDGP`.

- **`TopQuantileSelector`** preset (`badgers.core.selectors`): selects rows at
  or above the ``quantile`` of a column — self mode (own column) or driver
  mode (a driver column, optionally broadcast to all selected variables).
  Deterministic (no rng); thresholds computed on the called rows or on a
  separate ``stats`` row set (for generators realizing expanded rows against
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
  default): the rows SELECTED by the selector are dropped (subsampling
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
- **``fired()`` renamed to ``get_selected_fields()`` (breaking rename, no
  shim).** Same contract and behavior — narrow bool field
  ``(n_rows, len(targets))`` (column ``i`` ↔ the selector's ``i``-th
  target), ``selector=None`` → all-true WITHOUT consuming rng draws, single
  realization rule ``rng.random(n) < spec``. The selector layer migrates to
  the select/selected vocabulary in code and docstrings; the outliers
  plumbing module ``_firing.py`` becomes ``_selection.py`` with helpers
  ``select_and_seed`` (was ``fire_and_seed``) and ``iter_selection_groups``
  (was ``iter_firing_groups``). (commit `2621df9`)

### Documentation

- **New "Understanding badgers" concepts notebook**
  (`tutorials/concepts/Badgers-Concepts.ipynb`): a step-by-step
  introduction — module structure, generator instantiation with an rng,
  `generate()`, shape rules (appending / selection / subsampling), the
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

### Fixed

- **`AdditiveLinearTrendGenerator` crashes when `start=None` (the default).**
  The randomly drawn start was a float (`uniform(0, int(0.5 * len(X)))`) and hit
  numpy slicing → ``TypeError: 'float' object cannot be interpreted as an
  integer``. The draw is now cast to ``int``; the signature documents
  ``start=None`` (random start in the first half of the signal) and ``end=None``
  (trend lasts until the end).
- **`generate_random_patterns_indices` capacity guard off by one.**
  The pattern-position helper (``RandomlySpacedLinearTrends``, ``LocalNoiseGenerator``,
  ``RandomlySpacedLocalNoiseGenerator``, value-drop generators) passed a signal of size 11
  with 2 patterns of max width 6 — then crashed inside ``rng.integers`` with
  ``high <= 0``. The guard is now ``max_width_patterns * n_patterns <= signal_size``
  (each equal-length segment must hold one full-width pattern).
- **`RandomChangeInMeanGenerator` could draw duplicate changepoint indices.**
  All indices were drawn independently, so one position could appear twice in the reported
  ``changepoints`` list (verified: 14 duplicates in 40 draws) and the realized level jump at a
  repeated position was the sum of two draws. Indices are now drawn distinct (sampling without
  replacement) and reported sorted by index; each reported change is the realized level jump
  at that position (documented in the docstring). Same-seed outputs change (different draw
  mechanism).
- **`GaussianNoiseClassesGenerator` no longer silently skips unconfigured
  classes (contract change).** A selected class value absent from
  ``noise_std_per_class`` now raises ``ValueError`` (previously it silently
  received no noise — documented-and-tested partial mappings are gone; pass a
  zero std explicitly to keep a class noise-free). The dict keys are
  normalized against the class column's dtype: plain-int keys work for a
  float-encoded class column, and keys of a wrong kind (e.g. strings for
  numeric classes) raise instead of silently disabling the noise.
- **The per-class drift generators take the class column from the selector
  (contract change).** ``RandomShiftClassesGenerator`` and
  ``FixedShiftClassesGenerator`` no longer accept a ``class_column=`` generate
  argument — the selector owns the class information. The class column is
  resolved from the selector's ``class_column`` (``ClassSelector``, typical);
  a selector that exposes no class column and an out-of-range position both
  raise ``ValueError`` (the previous "all selected rows treated as one class"
  fallback is gone). The dict ``shift_std``/``shift`` knobs follow the same
  contract as ``noise_std_per_class``: a selected class value absent from the
  dict raises (no more silent skip), keys are dtype-normalized (plain-int keys
  work for a float-encoded class column), wrong-kind keys raise.
- **``selector=None`` now means "the relevant default selector" where one
  exists.** The three all-rows generators — ``AdditiveNoiseGenerator``,
  ``RandomShiftGenerator`` and ``FixedShiftGenerator`` — substitute
  ``AlwaysSelector(columns=range(X.shape[1]))`` (noise/shift on every row,
  the previous implicit behavior, now an explicit documented default).
  Where no relevant default exists, ``None`` still raises with a rationale:
  the per-class drift/noise generators need a selector that owns the class
  column, the causal generators need the user's intervention targets, and
  ``RandomSamplingTargetsGenerator`` needs the user's deficit scores;
  ``RandomSamplingClassesGenerator`` keeps its documented no-op ("selects
  nowhere"). Realization change in ``get_selected_fields()``: constant scalar specs
  (exactly ``0.0``/``1.0``) are realized deterministically WITHOUT a
  ``rng`` draw (fractional scalars and bool arrays still draw). Results
  per selector are identical, but same-seed downstream draws shift for
  callers whose selectors produce constant scalar specs (e.g. the new
  ``AlwaysSelector`` defaults).
- **``FixedShiftGenerator.generate`` renames ``shift_std`` → ``shift``
  (breaking).** The deterministic generator's parameter now matches the
  paired per-class generator (``FixedShiftClassesGenerator.generate(X,
  shift=...)``); the random counterparts keep ``shift_std`` correctly.
  No deprecation shim — passing ``shift_std=`` raises a ``TypeError``
  like any unknown keyword.
- **Parameter validation switched from bare ``assert`` to ``raise
  ValueError`` (contract change).** All 30 bare `assert` validations across
  the outliers, missingness, text, graph, time-series generators and core
  utils (e.g. ``assert expansion >= 0``, ``assert 0 <= percentage_missing <=
  1``, ``assert bins > 0``, ``assert size > 0``) are now explicit
  ``ValueError``s with descriptive messages — bare asserts silently vanish
  under ``python -O``. Code catching ``AssertionError`` at these sites must
  catch ``ValueError`` instead (8 tests updated).
- **Stale "capped at len(X)" docstrings corrected.** The outlier generators'
  docstrings claimed a requested outlier count is capped at the dataset size
  where the code actually raises (also true of the internal ``CountSelector``
  fallback). Wording is now "The count must not exceed ``len(X)`` (a larger
  count raises)" — ``distribution_sampling`` (3 sites), the low-density trio
  (3 sites + ``_resolve_count_and_targets``) and ``instance_sampling``.
- **Scalar ``slope`` in the time-series trend generators is broadcast to
  every column (bug fix).** ``AdditiveLinearTrendGenerator`` and
  ``GlobalAdditiveLinearTrendGenerator`` previously crashed with a scalar
  slope on non-square X (``ValueError: could not broadcast ... (n,) into
  (n,2)``) and silently applied the ramp along COLUMNS on square X
  (n == n_cols) — a transposed trend. A scalar slope now applies the same
  per-row ramp to every column (the documented behavior, consistent with the
  array-slope path, whose outputs are unchanged).
- **The five hand-copied causal ``get_*_labels()`` getters are unified**
  (internal refactor, no behavior change): drift, missingness, imbalance and
  both noise generators now delegate to one shared helper on the causal base
  (``_get_deficit_labels``); the error message is derived from each
  generator's ``deficit_name``.
- **``SwapLettersGenerator`` no longer mutates the input list.** The other
  text generators return a new list, so did this one — the returned list is
  a copy and the input words are left unchanged (same-seed outputs are
  identical to before).
- **Graph missingness generators raise ``TypeError`` for non-`nx.Graph`
  input** (was ``NotImplementedError`` — nothing is "not implemented yet",
  the input type is wrong).
- **Time-series ``MissingAtRandomGenerator`` renamed to
  ``MissingCompletelyAtRandomGenerator``** (breaking rename, no shim): the
  generator is MCAR — values are set to NaN uniformly at random, with no
  dependence on any observed column (the missingness mechanism of the
  tabular generators is already called ``MissingCompletelyAtRandom``).
  The benchmark registration is renamed accordingly.
- **Docstring hygiene sweep.** The drift base-class docstring no longer
  claims to "add noise" (it shifts data); the time-series
  ``MissingValuesGenerator`` docstring no longer claims to generate "point
  outliers" (it generates missing values); the "thant" typo in the
  two-dimensional-values error message of [utils.py](../..//badgers/generators/time_series/utils.py)
  is fixed; the obscure ``for...else`` in
  ``generate_random_patterns_indices`` (its else-body always ran after loop
  completion and read the leaked loop variable) is restructured to plain
  code with the same draw order and outputs; the ``import warnings`` inside
  ``_column_stds`` moved to module top.

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