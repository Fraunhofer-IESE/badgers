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
  on the un-mutated input) and combined with the `perturbation_nodes` column mask. The shared
  row-selection helper lives in `badgers.core.selectors` (`fired_mask`/`fired_rows`; also used by
  the drift and imbalance consumers). Covered by 6 new tests in
  [test_additive.py](tests/generators/tabular_data/noise/test_additive.py); full suite green.
- **Consequence / open decisions**:
  - `GaussianNoiseClassesGenerator` (a.k.a. GaussianWhiteNoise) then degenerates to
    `AdditiveNoiseGenerator` with `distribution=norm(...)` — a special case. DECIDE after this
    refactor: remove it, or keep/re-implement it as a thin configuration over the unified
    generator. Note: its per-class `noise_std_per_class` lookup is NOT expressible through a
    selector (a selector decides which rows fire, not the noise magnitude), so a true per-class
    std needs an explicit mechanism (e.g. per-class distribution configs or per-column std).
  - Decide whether noise generators should expose `get_noise_labels()` like the causal ones.

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

- **Current** ([outliers/](badgers/generators/tabular_data/outliers)): 8 classical generators, each
  inlining its own out-of-distribution point-generation logic (ZScore, Hypersphere, HyperCube
  uniform, UniformInstanceAttribute recombination, IndependentHistograms, Histogram dd, KDE
  LowDensitySampling) plus `StructuredOutlierGenerator` (reduce → sample → inverse-transform
  wrapper). Meanwhile the CAUSAL outlier generator already delegates to the `Sampler` objects in
  [badgers/core/sampling.py](badgers/core/sampling.py) — with near-duplicates of the classical
  logic: `ZScoreSampler` ≈ `ZScoreSamplingGenerator`, `HypersphereSampler` ≈
  `HypersphereSamplingGenerator`, `UniformOutOfDistributionSampler` ≈ `HyperCubeSampling`
  (difference: the core one samples strictly BEYOND [min, max], `HyperCubeSampling` samples WITHIN
  the scaled box, expansion included).
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