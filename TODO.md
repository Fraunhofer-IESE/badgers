# Getting started:

- section 6 shall introduce "SCM" and make it more explicit what hte code is doing (intervenene and sample)

- all links at the end are broken (example: https://fraunhofer-iese.github.io/badgers/tutorials/concepts/tabular-data/Outliers.ipynb shall be https://fraunhofer-iese.github.io/badgers/tutorials/tabular-data/Outliers.ipynb). Maybe move all tutorials (also the ones in docs/dev) to the same folder docs/tutorials and update the links accordingly.

# Refactoring requirements: selector-unification (tabular_data)

Context: a **selector** (`badgers.core.selectors`: `X -> {variable: bool-mask | float score}`,
`True`/`1.0` = deficit applied) is the shared observation-side contract that decides WHICH rows/cells
receive a data quality defect. Today only part of the `tabular_data` generators use it (the
`*Classes` variants, the imbalance selectors, the causal generators). These items track the
unification; each states current behavior, target behavior, acceptance criteria and open decisions
so that specifications/requirements can be derived directly from them.

## 1. `AdditiveNoiseGenerator` must take a selector ✅ DONE

- **Current** ([noise/noise.py](badgers/generators/tabular_data/noise/noise.py)): adds noise drawn
  from an arbitrary frozen scipy distribution (global `noise_std` multiplier) to EVERY row;
  restriction is only possible via `perturbation_nodes` COLUMN indices — there is no per-row
  selection.
- **Target**: add a `selector=None` parameter (the `badgers.core.selectors` contract); only FIRED
  rows receive noise, all other rows are returned unchanged. `selector=None` keeps the current
  behavior (every row fires) for backward compatibility.
- **Acceptance criteria**:
  - `generate(X, distribution, ..., selector=None)`: masks computed on the UN-mutated input
    (compute-all-masks-first invariant) and OR-combined across the selector's variables.
  - Modifying consumer semantics: shape unchanged, noise added in place on fired rows only.
  - Any selector preset (Always, Bernoulli, ExactCount, TopQuantile, Class, Proportional) and any
    callable with the same signature is accepted.
  - Tests: `selector=None` reproduces the current whole-data behavior; a class selector restricts
    noise to exactly that class's rows; unfired rows are bit-identical to the input.
- **Implementation status**: done — `AdditiveNoiseGenerator.generate` accepts keyword-only
  `selector=None`; the drawn noise is zeroed on unfired rows (OR of the selector's masks, computed
  on the un-mutated input). The shared realization helper is
  `badgers.core.selectors.fired(X, selector, rng)`: it returns the narrow bool field
  `(n_rows, n_targets)` (column `i` ↔ the selector's `i`-th target); consumers take the
  row-level union themselves with `field.any(axis=1)` (the drift and imbalance consumers do
  the same). Covered by 6 new tests in
  [test_additive.py](tests/generators/tabular_data/noise/test_additive.py); full suite green.
- **Consequence / open decisions**:
  - `GaussianNoiseClassesGenerator` (a.k.a. GaussianWhiteNoise) then degenerates to
    `AdditiveNoiseGenerator` with `distribution=norm(...)` — a special case. DECIDE after this
    refactor: remove it, or keep/re-implement it as a thin configuration over the unified
    generator. Note: its per-class `noise_std_per_class` lookup is NOT expressible through a
    selector (a selector decides which rows fire, not the noise magnitude), so a true per-class
    std needs an explicit mechanism (e.g. per-class distribution configs or per-column std).
  - ~~Decide whether noise generators should expose `get_noise_labels()` like the causal ones.~~
    DECIDED (2026-10-25, review): the direct generators deliberately report nothing — TODO item 6
    tracks the uniform getter for every family; adding `get_noise_labels()` now would only add a
    third convention.

## 2. MCAR / MAR / MNAR are selection mechanisms — refactor onto selectors

- **Current**
  ([missingness/missingness.py](badgers/generators/tabular_data/missingness/missingness.py)):
  `MissingCompletelyAtRandom`, `DummyMissingAtRandom`, `DummyMissingNotAtRandom` each hard-code its
  mechanism (internally built `ExactCountSelector` + weights callables `_mar_weights` /
  `_mnar_weights`). The mechanism↔selector mapping already exists in the causal missingness
  generator docstring: `BernoulliSelector` = MCAR (value-blind),
  `TopQuantileSelector` driver-mode = MAR (probability driven by ANOTHER column), self-mode = MNAR
  (probability grows with the OWN value).
- **Target**: one selector-driven missingness mechanism where MCAR, MAR and MNAR become selector
  configurations/presets (value-blind / driver-column / self-value coupling).
- **Acceptance criteria**:
  - The three mechanisms are reproducible as selector configurations with the same statistical
    semantics (which cells can go NaN, and the coupling direction).
  - The dummy MAR driver draw (a random OTHER column per column, legacy RNG call order) is either
    expressible in the selector contract or replaced by an explicit driver-column parameter.
  - Deficit bookkeeping unchanged: `missing_values_indices_` + NaN writing stay generator-side.
- **Open decisions**:
  - What to do with the three dummy generators (specific cases): keep / remove / keep as
    deprecated aliases once the selector-based generator covers them.
  - Exact-count vs iid-rate semantics for MCAR; how `percentage_missing` maps onto the selector
    presets' intensity knobs (`rate`, `quantile`, ...).

## 3. `RandomSamplingFeaturesGenerator` must use a selector ✅ DONE

- **Current**
  ([imbalance/imbalance.py](badgers/generators/tabular_data/imbalance/imbalance.py)): resamples
  rows with per-row probabilities from a free `sampling_proba_func` (default: min-max normalized
  FIRST column) — a value-weighted resample whose probability rule bypasses the selector contract.
- **Target**: replace `sampling_proba_func` with a selector returning per-row scores in [0, 1]
  (e.g. `ProportionalSelector("col", column_mapping)` reproduces the current default). Scores are
  realized as keep weights `1 - p`, exactly like `RandomSamplingTargetsGenerator` (score
  validation in [0, 1], noisy-OR over multiple variables, boolean scores = deterministic
  subsample).
- **Acceptance criteria**:
  - One code path shared with `RandomSamplingTargetsGenerator` for score realization.
  - `imbalance_labels_` semantics preserved (True = row dropped from the output).
  - Defined behavior for `selector=None` (either required, or None = uniform resampling).
- **Open decisions**:
  - Merge `RandomSamplingFeaturesGenerator` and `RandomSamplingTargetsGenerator` (both are
    score-driven resamplers) or document their difference.
- **Implementation status**: done — the free-probability twin was removed (breaking):
  `_column_sampling_proba` and `RandomSamplingFeaturesGenerator` deleted from
  [imbalance.py](badgers/generators/tabular_data/imbalance/imbalance.py) and its `__init__`;
  `RandomSamplingTargetsGenerator` is the single score-driven resampler (`selector` REQUIRED,
  `selector=None` raises; boolean selectors such as `ClassSelector` realize an exact per-class
  subsample, fractional selectors such as `ProportionalSelector` reproduce the former
  first-column-weighted default). The class docstring documents the two selector flavors; the
  Imbalance tutorial now shows both flavors on the one generator. Covered by rewritten/new tests
  in [test_imbalance.py](tests/generators/tabular_data/test_imbalance.py) (37 pass).

## 4. `RandomShiftGenerator` and `FixedShiftGenerator` must take a selector ✅ DONE

- **Current** ([drift/drift.py](badgers/generators/tabular_data/drift/drift.py)): both shift EVERY
  row — random variant: per-column translation $\epsilon_i \sim \mathcal{N}(0, \text{shift\_std})$
  after standardization; fixed variant: exact per-column shift (reproducible). The `*Classes`
  counterparts already require a selector.
- **Target**: add `selector=None` (None = all rows fire, backward compatible) so the whole drift
  family is uniformly selector-driven; fired rows = OR of the selector's masks.
- **Acceptance criteria**:
  - Same standardize → shift → inverse-transform pipeline; shifts drawn/applied ONLY for fired
    rows; unfired rows bit-identical to the input; masks computed on the un-mutated input.
  - Per-class shift/noise lookup stays in the `*Classes` variants.
  - Tests: `selector=None` reproduces current whole-data behavior; a `ClassSelector` restricting
    to one class shifts only that class's rows.
- **Implementation status**: done — both generators accept keyword-only `selector=None`
  (`AlwaysSelector`, `BernoulliSelector`, `TopQuantileSelector`, `ClassSelector`, ... all work);
  shifts are drawn/applied only for fired rows, masks are OR-ed from the selector output and
  computed on the un-mutated input. Covered by 11 new tests in
  [test_drift.py](tests/generators/tabular_data/test_drift.py) (all 93 pass).
- **Open decisions**:
  - Merge Random* and Fixed* into one generator with a deterministic/random switch (they differ
    only in whether the shift is drawn), and/or converge the base and Classes variants into a
    single selector-driven shift generator with optional per-class parameters.

## 5. Outliers: refactor around pluggable point-generation objects (core Samplers)

- **Current** ([outliers/](badgers/generators/tabular_data/outliers)): 8 classical generators.
  Since the 0.0.15 refactor the three distribution-based ones (`ZScoreSamplingGenerator`,
  `HypersphereSamplingGenerator`, `HyperCubeSampling`) delegate to the core Samplers in
  [badgers/core/sampling.py](badgers/core/sampling.py) (logic defined ONCE there, per the
  acceptance criteria below); `StructuredOutlierGenerator` remains the reduce → sample →
  inverse-transform composition wrapper. Still inlining their strategies:
  `UniformInstanceAttributeSampling`/`VarianceInstanceSampling` (instance recombination) and the
  histogram trio (`IndependentHistogramsGenerator`, `HistogramSamplingGenerator`,
  `LowDensitySamplingGenerator`) — the KDE and histogram-dimension Sampler implementations listed
  in the acceptance criteria do not exist yet.
- **Target**: one outlier generator appending `n_outliers` rows produced by a pluggable core
  Sampler (out-of-distribution point generation); the histogram/KDE/low-density strategies become
  new Sampler implementations in `badgers/core/sampling.py`; `StructuredOutlierGenerator` remains
  a composition wrapper (reduced space) around any strategy; the causal outlier generator uses the
  same objects — one shared vocabulary, no duplicated logic.
- **Acceptance criteria**:
  - Every classical generator is reachable as (unified generator + Sampler configuration) — as a
    thin subclass/alias or a deprecated wrapper; `get_outliers_labels()` works identically.
  - New Sampler implementations: low-density sampling (KDE rejection with `max_samples` budget)
    and histogram-based strategies, each carrying its own parameters (bins, threshold, scale...).
  - ZScore / Hypersphere / Uniform logic defined ONCE (in `badgers/core/sampling.py`).
  - Instance recombination (`UniformInstanceAttributeSampling`) either becomes an
    instance-based Sampler or is documented as intentionally not out-of-distribution.
- **Open decisions**:
  - NAMING: is "Sampler" the right word for these objects (candidate alternatives: outlier
    strategy / point-generation strategy)? Note `badgers/core/sampling.py` already names them
    Samplers, so renaming is a cross-cutting API decision.
  - `UniformOutOfDistributionSampler` vs `HyperCubeSampling` polarity (beyond-range vs
    within-box): one sampler with a flag, or two?
  - Value-level compatibility: must pre-refactor outputs be reproducible bit-for-bit for the same
    seed, or is API compatibility enough?
- **Implementation status**: partially done (updated 2026-10-25, generator review) — the three
  distribution-based generators now construct the core Samplers instead of duplicating their math
  (`fire_and_seed` in [outliers/_firing.py](badgers/generators/tabular_data/outliers/_firing.py)
  realizes the selector/`n_outliers` count and seeds the anchor rows). Remaining gaps: the
  histogram/KDE strategies are still generator-local (no core Sampler), the count-resolution
  helper `_resolve_count_and_targets` in
  [low_density_sampling.py](badgers/generators/tabular_data/outliers/low_density_sampling.py)
  duplicates the `fire_and_seed` contract for the selector-keys convention, and the open decisions
  above are unresolved.

## 6. All generators must provide a uniform "affected rows" function (ground truth)

- **Current** (various): ground-truth reporting is inconsistent across families and modalities:
  - Outliers: ALL (direct + causal) report via `get_outliers_labels()` — row labels aligned with
    the OUTPUT (`True` = appended outlier row, `False` = original row kept as-is).
  - Imbalance: ALL report via `get_imbalance_labels()` — aligned with the INPUT (`True` = input row
    dropped from the output); the causal variant marks every output row `True` (trivially, since its
    output *is* the affected subset).
  - Missingness: the direct MCAR/MAR/MNAR generators record `missing_values_indices_` (exact cell
    positions `(row, col)`), the causal variant reports `get_missing_labels()` (rows with ≥ 1 NaN).
  - Drift: the 4 direct shift generators report NOTHING; only `CausalDriftGenerator` reports
    (`get_drift_labels()`).
  - Noise: `AdditiveNoiseGenerator` and `GaussianNoiseClassesGenerator` report NOTHING; only the
    causal variants report (`get_noise_labels()`).
- **Target**: EVERY generator (all modalities, all families, direct and causal) provides one
  uniform function answering "which rows of the input were affected by the last generated
  deficit", with consistent semantics. Free ground truth must not depend on which family the user
  picked.
- **Acceptance criteria**:
  - One uniform getter on the shared base (`GeneratorMixin` or a labels mixin), raising
    `RuntimeError` before the first `generate()` call (current behavior to preserve).
  - Per-family getters (`get_outliers_labels()`, `get_imbalance_labels()`, ...) are kept as thin
    aliases/deprecations during the transition — no breaking change for existing users.
  - Alignment semantics defined ONCE and applied everywhere (see open decisions); documented in
    each docstring.
  - Missingness additionally keeps cell-level information (`missing_values_indices_`) as a
    complementary accessor; the row-level getter reports rows with ≥ 1 missing cell.
  - Subsampling generators (imbalance) and row-expanding generators (noise `n_copies > 1`) must
    remain recoverable: from labels alone it must be possible to tell which input rows were
    dropped / which output rows are copies.
- **Open decisions**:
  - Alignment semantics: labels aligned to the INPUT ("this input row was affected") or to the
    OUTPUT ("this output row carries the deficit")? They differ whenever rows are appended
    (outliers), duplicated or removed (imbalance, noise `n_copies > 1`). Note the current split:
    outliers are output-aligned, direct imbalance generators are input-aligned.
  - Naming: one generic name (e.g. `get_deficit_labels()`, matching the causal base's
    `deficit_labels_` attribute) vs. keeping per-family names permanently.
  - Whether "affected" means "deficit fired on this row" (selection) or "row value changed"
    (e.g. a causal descendant recomputed after an upstream defect) — the causal generators
    currently label only rows where the defect fired, not rows passively recomputed.

# Future work

- Re-introduce the `CausalEdgePerturbationGenerator` (relationship-level
  defect: scale one edge coefficient by `weight_factor`, recompute the
  child marginal-preserving, propagate the descendants through the DGP).
  It was removed before the 0.0.15 release; the underlying DGP edit
  (`edits.scale_edge` in `badgers.core.dgp.edits`) is still shipped and
  tested, so the generator can be re-added on top of it — see the
  git history (`docs/tutorials/tabular-data/Edge-Perturbation.ipynb`,
  `badgers/generators/tabular_data/relationship/causal.py`).

# Generator review 2026-10-25 (systematic pass over all 38 generator files)

Findings from reading every generator (tabular, causal, time_series, graph, text)
and probing the suspected bugs at runtime. Ordered by severity. Item numbers are
referenceable as R1..R14 in review discussions.

## Bugs (real, reachable)

- **R1** ✅ FIXED (2026-10-25): `AdditiveLinearTrendGenerator.generate` now draws
  `start = int(...)` and documents the `start=None`/`end=None` semantics (commit
  `ddc7b8b`); regression test `test_additive_linear_trend__default_start_end` in
  [test_trends.py](tests/generators/time_series/test_trends.py) asserts the drawn int start
  (drawn value is unknowable from the output — the ramp's leading 0 lands on the drawn row),
  the constant positive ramp step, and the final ramp value. Related (separate item R15):
  the scalar-slope × multi-column broadcast bug in the same slice is NOT fixed by R1.
- **R15** [trends.py](badgers/generators/time_series/trends.py): `AdditiveLinearTrendGenerator.generate`
  with a SCALAR `slope` and multi-column `X` crashes: `np.linspace(0, slope * (end - start),
  end - start)` is 1-D and cannot broadcast into the `(rows, n_cols)` slice assignment →
  `ValueError: could not broadcast input array from shape (n,) into shape (n,2)` (verified;
  the docstring explicitly claims "Can be a single float value or a list of slopes for each
  feature"). With an ARRAY slope the linspace broadcasts fine (the array is inserted on the
  trailing axis) — the bug is specific to the scalar case. WORSE: for SQUARE X (n == n_cols)
  the scalar path does NOT crash — it silently applies the ramp along COLUMNS instead of rows
  (each row j gets the constant value ramp[j]; verified). `GlobalAdditiveLinearTrendGenerator`
  is ALSO affected (verified): scalar slope on (100,2) raises
  `ValueError: operands could not be broadcast together with shapes (100,2) (100,)`, and on
  square X it silently applies the ramp along columns. Only the array-slope path is correct
  in both generators.
  Fix candidate: broadcast the 1-D ramp against `X.shape[1]` (e.g. reshape + `np.ones`, or
  `np.broadcast_to`).
- **R2** ✅ FIXED (2026-10-25): the capacity guard is now
  `max_width_patterns * n_patterns <= signal_size` (commit `3995c79`); regression tests for the
  exact boundary (works) and the previously-crashing 11-signal call (now raises). Note the guard
  is a conservative budget for the SEGMENT structure (each segment must hold a full-width
  pattern), not a tight packing condition.
- **R3** ✅ FIXED (2026-10-25): indices are drawn DISTINCT (sampling without replacement via
  `rng.choice(..., replace=False)`) and `changepoints` is reported sorted by index with the
  documented composition semantics (each reported change IS the realized level jump at that
  position) (commit `7787279`). Same-seed outputs changed (different draw mechanism).
- **R4** ✅ FIXED (2026-10-25): every class value present among the fired rows must now have an
  entry in `noise_std_per_class` (missing → `ValueError`, was silent skip); dict keys are
  normalized against the class column's dtype (plain-int keys work for a float-encoded class
  column; wrong-kind keys, e.g. strings for numeric classes, raise instead of silently
  disabling the noise) (commit `443daba`). Contract change: the documented-and-tested
  "absent class = no noise" partial-mapping behavior is gone.

## API inconsistencies (same concept, different spelling)

- **R5** ✅ FIXED (2026-10-25, ruling: "the selector owns the information about the class"):
  `class_column=` is removed from `RandomShiftClassesGenerator.generate` and
  `FixedShiftClassesGenerator.generate`; the class column is resolved ONLY from the selector
  (`resolve_class_column`). A selector that exposes neither `class_column` nor `column` raises
  (the old "all fired rows treated as one class" fallback is gone), and an out-of-range
  position raises a clean ValueError — `ClassSelector.__call__` now validates the position
  itself (the old generator-side check was the only guard; without it the selector crashed
  with a raw IndexError). Dict knobs follow the R4 contract: a fired class value absent from
  the dict raises, keys are dtype-normalized, wrong-kind keys raise. Mirror tests added for
  both generators (commit `eff208b`).
- **R6** ✅ FIXED (2026-10-25, ruling: "option A — default to a relevant selector"):
  `selector=None` semantics are now uniform per the relevant-default rule. The three all-rows
  generators (`AdditiveNoiseGenerator`, `RandomShiftGenerator`, `FixedShiftGenerator`)
  substitute the documented default `AlwaysSelector(columns=range(X.shape[1]))` — the
  ad-hoc early-return/fallback paths are gone. Where no relevant default exists, `None`
  still raises with a per-site rationale (per-class drift/noise generators: the selector owns
  the class column per R5; causal generators: the selector owns the user's intervention
  targets; `RandomSamplingTargetsGenerator`: resampling needs the user's deficit scores);
  `RandomSamplingClassesGenerator` keeps its documented no-op ("fires nowhere"). Supporting
  change in `fired()`: constant scalar specs (exactly 0.0/1.0) are realized WITHOUT a rng
  draw (fractional scalars/bool arrays still draw), so the new AlwaysSelector defaults cost
  no draws and keep the old draw-free None path's stream alignment — pinned causal-noise test
  seed updated for the shifted stream. Default-vs-explicit equivalence tests added for all
  three generators plus two fired() contract tests (commit `abb4199`).
- **R7** ✅ FIXED (2026-10-25, ruling: "hard rename"): `FixedShiftGenerator.generate`'s
  deterministic parameter is renamed `shift_std` → `shift`, matching
  `FixedShiftClassesGenerator.generate(X, shift)` (the random counterparts keep `shift_std`
  correctly). No deprecation shim — the old kwarg now raises a TypeError like any unknown
  keyword. All tests and the Drift tutorial updated.
- **R8** Validation style: bare `assert` (vanishes under `python -O`) in outliers/missingness/
  text/time-series generators (`assert expansion >= 0`, `assert 0 <= percentage_missing <= 1`,
  `assert bins > 0`, ...) vs explicit `raise ValueError` in `AdditiveNoiseGenerator` and
  `CausalNoiseGenerator`. Also stale docstrings claiming counts are "capped at len(X)" where the
  code raises (distribution_sampling x3, low-density trio, `UniformInstanceAttributeSampling`,
  `HistogramSamplingGenerator`); `instance_sampling.py` still references `UniformInstanceSampling`
  in a docstring (the class was renamed `UniformInstanceAttributeSampling`).
- **R9** Ground-truth attribute zoo (superset of TODO item 6): direct families use
  `<family>_labels_` (`outlier_labels_`, `imbalance_labels_`), the causal base uses
  `deficit_labels_` while `CausalOutlierPropagationGenerator` additionally keeps its own
  `outlier_labels_` (two attributes for one concept), missingness has only cell-level
  `missing_values_indices_`, drift/noise direct generators report nothing, time-series exposes raw
  lists with five different shapes (`outliers_indices_`, `missing_indices_`, `changepoints`,
  `patterns_indices_`, `drops_indices_`...), and every causal generator hand-copies an identical
  `get_*_labels()` getter that only differs in the error message. Item 6's acceptance criteria
  already prescribe the fix (one labels mixin on the shared base + per-family aliases).

## Refactor opportunities (no behavior change)

- **R10** [drift/drift.py](badgers/generators/tabular_data/drift/drift.py): the Random*/Fixed* and
  base/Classes variants are near-copies (the Classes pair share ~90% of their body). Item 4's open
  decision (merge into one generator with a deterministic/random switch) also resolves R5/R7.
- **R11** [noise/noise.py](badgers/generators/tabular_data/noise/noise.py): the hand-rolled
  `covered = range(X.shape[1])` fallback carries a "same default as _resolve_targets ... keep in
  sync" comment — call `_resolve_targets(selector, X, None)` instead of duplicating the default.
- **R12** [noise/causal.py](badgers/generators/tabular_data/noise/causal.py): causal generator
  signatures disagree — `CausalNoiseGenerator.generate(selector=None...)` raises at runtime while
  drift/missingness/outliers declare `selector` as a required parameter; `noise_std` supports
  dict-by-target only in `CausalNoiseGenerator` (`_param_by_target`) while
  `CausalAdditiveNoiseGenerator.noise_std` is a plain scalar. Unify (required selector; dict-or-
  scalar via `_param_by_target` everywhere).
- **R13** Constructor boilerplate: every generator redefines `__init__` only to re-document and
  call `super().__init__` — and `default_rng(seed=0)` in the default argument is evaluated ONCE at
  import, so all instances created without an explicit rng SHARE one Generator object (verified:
  two `AdditiveNoiseGenerator()` instances advance each other's streams). Latent footgun for
  users who create several generators and expect independent streams; fix = `None` default +
  per-instance `default_rng()` (behavior change for anyone relying on the shared stream — needs a
  changelog note).
- **R14** Cosmetics: [graph/missingness.py](badgers/generators/graph/missingness.py)
  raises `NotImplementedError` for non-nx.Graph input (should be `TypeError`);
  [causal/_base.py](badgers/generators/tabular_data/causal/_base.py) imports `warnings` inside
  `_column_stds`; [utils.py](badgers/generators/time_series/utils.py) has the typo "thant", a
  duplicated `:return:` line in `get_patterns_uniform_probability`, and an obscure `for...else`
  that reads the loop variable after the loop; `MissingAtRandomGenerator` (time_series) is
  actually MCAR (misnomer); text generators disagree on mutation (SwapLettersGenerator mutates the
  input list, LeetSpeakGenerator returns a new one); several family base-class docstrings are
  copy-paste from the wrong module (drift base says "add noise", ts `MissingValuesGenerator` says
  "generate point outliers").


  # UNIFIY DOCSTINGS!