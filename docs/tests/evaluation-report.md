# Test Suite Evaluation Report — Badgers

**Date:** 2026-09-15  
**Evaluator:** AI Coding Agent (test-evaluator skill)  
**Verdict:** **Accept with conditions**

The test suite provides reasonable coverage for most generators, but has several critical issues: 30 tests in `test_causal.py` are **failing** due to an API mismatch (the `outlier_magnitude` parameter was removed from `CausalOutlierPropagationGenerator.generate()`), and several test files contain weak oracles that only check shape preservation without verifying behavioral correctness.

---

## 1. Scope and Assumptions

### Artifacts Reviewed

- **Source code:** `badgers/core/`, `badgers/generators/` (tabular_data, time_series, text, graph)
- **Test suite:** `tests/core/`, `tests/generators/`, `tests/benchmarks/`
- **Documentation:** `docs/index.md`, `docs/architecture.md`, `AGENTS.md`
- **Specification:** The project README and architecture docs serve as the primary specification. The core contract is: each generator provides a `generate(X, y, **params)` method returning `(Xt, yt)`.

### Environment

- Python 3.12.6, pytest 9.0.2, Windows
- Test run: 148 tests collected, 118 passed, 30 failed

### Assumptions

- The `GeneratorMixin` interface (`generate(X, y, **params) -> Tuple`) is the authoritative contract.
- The architecture docs describe the intended behavior of each generator.
- The `AGENTS.md` performance patterns are normative for implementation quality but not for behavioral correctness.

---

## 2. Requirement Coverage

### Core Module

| Requirement | Tests | Status |
|---|---|---|
| `GeneratorMixin` abstract interface | None directly | Not tested |
| `Pipeline` chains generators | `test_pipelines.py::test_generate` | **Weak** — no assertions |
| `normalize_proba` normalizes to sum=1 | `test_utils.py::test_normalize_proba` | Covered |
| `random_sign` produces ±1 | `test_utils.py::test_random_signs` | Covered |
| `random_spherical_coordinate` lies on sphere | `test_utils.py::test_random_spherical_coordinate` | Covered |
| `Sampler` base class abstract | `test_sampling.py::test_sampler_is_abstract` | Covered |
| `NormalSampler` mean/std approx | `test_sampling.py::test_normal_sampler_*` | Covered |
| `UniformSampler` within bounds | `test_sampling.py::test_uniform_sampler_*` | Covered |
| `ZScoreSampler` ≥3σ from mean | `test_sampling.py::test_zscore_sampler_*` | Covered |
| `HypersphereSampler` ≥3σ distance | `test_sampling.py::test_hypersphere_sampler_*` | Covered |
| `UniformOutOfDistributionSampler` beyond bounds | `test_sampling.py::test_uniform_ood_sampler_*` | Covered |
| Sampler factory functions | `test_sampling.py::test_create_*` | Covered |
| Causal graph utilities (DAG validation, topology, parents, etc.) | `test_causal_graph.py` (14 tests) | **Well covered** |

### Tabular Data Generators

| Requirement | Tests | Status |
|---|---|---|
| `GaussianNoiseGenerator` preserves shape | `test_noise.py::test_gaussian_noise__preserves_shape` | **Weak** — shape-only |
| `GaussianNoiseGenerator` increases variance | `test_noise.py::test_gaussian_noise__increases_variance` | Covered |
| `GaussianNoiseClassesGenerator` preserves shape | `test_noise.py::test_gaussian_noise_classes__preserves_shape` | **Weak** — shape-only |
| `RandomShiftGenerator` preserves shape (scalar/array) | `test_drift.py` (2 tests) | **Weak** — shape-only |
| `RandomShiftClassesGenerator` preserves shape (scalar/array/2D) | `test_drift.py` (3 tests) | **Weak** — shape-only |
| `MissingValueGenerator` subclasses insert NaNs | `test_missingness.py` (1 test) | Partially covered |
| `RandomSamplingClassesGenerator` preserves shape | `test_imbalance.py` (1 test) | **Weak** — shape-only |
| `RandomSamplingFeaturesGenerator` preserves shape | `test_imbalance.py` (1 test) | **Weak** — shape-only |
| `RandomSamplingTargetsGenerator` preserves shape | `test_imbalance.py` (1 test) | **Weak** — shape-only |
| Outlier generators shape/labels | `test_outliers.py` (multiple) | Covered |
| Outlier reproducibility | `test_outliers.py::test_outliers__reproducibility_given_same_seed` | Covered |
| Outlier scores worse than original | `test_outliers.py::test_outliers__scores_worse_than_original` | Covered |
| `CausalOutlierPropagationGenerator` | `test_causal.py` (30 tests) | **ALL FAILING** — API mismatch |

### Time Series Generators

| Requirement | Tests | Status |
|---|---|---|
| `LocalGaussianNoiseGenerator` runs | `test_noise.py::test_local_gaussian_noise__generates` | **Weak** — no assertions |
| `GlobalGaussianNoiseGenerator` runs | `test_noise.py::test_global_gaussian_noise__generates` | **Weak** — no assertions |
| `RandomZerosGenerator` shape/count | `test_outliers.py` | Covered |
| `LocalZScoreGenerator` shape/count/no NaN | `test_outliers.py` | Covered |
| `MissingAtRandomGenerator` shape/count | `test_missingness.py` | Covered |
| `RandomChangeInMeanGenerator` modifies data | `test_changepoints.py` | **Weak** — only checks `any(X != Xt)` |
| Trend generators match expected | `test_trends.py` (3 tests) | **Well covered** |
| Season generator matches expected | `test_seasons.py` | Covered |
| Pattern utilities | `test_patterns.py` (8 tests) | **Well covered** |
| Transmission error generators | `test_transmission_errors.py` (13 tests) | **Well covered** |
| Pattern index generation | `test_utils.py` (3 tests) | Covered |

### Text Generators

| Requirement | Tests | Status |
|---|---|---|
| `SwapLettersGenerator` transforms long words | `test_typos.py` | Covered |
| `LeetSpeakGenerator` same length | `test_typos.py` | **Weak** — length-only |
| `SwapCaseGenerator` uppercases | `test_typos.py` | Covered |

### Graph Generators

| Requirement | Tests | Status |
|---|---|---|
| `NodesMissingCompletelyAtRandom` removes nodes | `test_missingness.py` (2 tests) | Covered |
| `EdgesMissingCompletelyAtRandom` removes edges | `test_missingness.py` (1 test) | Covered |

---

## 3. Oracle Assessment

### Strong, Independent Oracles

- **`test_sampling.py`**: `ZScoreSampler` and `HypersphereSampler` tests verify mathematical properties (≥3σ, Euclidean distance) derived from the specification, not from the implementation.
- **`test_causal_graph.py`**: Topological order, parent/descendant/ancestor relationships are verified against graph-theoretic properties independent of implementation.
- **`test_trends.py`**: Expected outputs are computed from the mathematical definition of linear trends, not from running the code.
- **`test_seasons.py`**: Expected sinusoidal output computed from `sin(t * 2π / period)`.
- **`test_patterns.py`**: Pattern transformations verified against arithmetic expectations.
- **`test_transmission_errors.py`**: Length changes, value preservation verified against specification.

### Weak or Circular Oracles

- **`test_pipelines.py::test_generate`**: **No assertions at all.** The test only calls `pipeline.generate()` without checking any result. This is the weakest possible oracle — it only verifies the code doesn't crash.
- **`test_noise.py` (time series)**: Both `test_local_gaussian_noise__generates` and `test_global_gaussian_noise__generates` have **zero assertions**. They only verify the code runs without exception.
- **`test_drift.py` (all 5 tests)**: Only check `len(X) == len(Xt)`. No verification that drift actually occurred or that the shift magnitude is correct.
- **`test_imbalance.py` (all 3 tests)**: Only check shape preservation. No verification that sampling probabilities are respected or that the output distribution matches expectations.
- **`test_noise.py` (tabular, `GaussianNoiseClassesGenerator`)**: Only checks shape. No verification that per-class noise was applied correctly.
- **`test_changepoints.py`**: Only checks `any(X != Xt)`. A generator that adds 1e-15 to one cell would pass.
- **`test_typos.py::test_leet_speak__generates_same_length`**: Only checks length. No verification that leet speak transformation actually occurred.

### Missing Observations

- No test verifies that `GaussianNoiseGenerator` with `noise_std=0` produces identical output.
- No test verifies that `RandomShiftGenerator` with `shift_std=0` produces identical output.
- No test verifies the `MissingCompletelyAtRandom` vs `DummyMissingAtRandom` behavioral difference.
- No test verifies that `RandomSamplingClassesGenerator` actually respects the `proportion_classes` distribution.
- No test verifies that `RandomSamplingFeaturesGenerator` actually respects the `sampling_proba_func`.

---

## 4. Fault-Discrimination Results

### Surviving Semantic Mutants (Hypothetical)

The following plausible incorrect implementations would pass the current test suite:

1. **Constant-return noise generator**: A `GaussianNoiseGenerator` that always returns `X + 0.001` would pass the "increases variance" test (since variance increases slightly) and the shape test. No test checks the noise magnitude.

2. **No-op drift generator**: A `RandomShiftGenerator` that ignores `shift_std` and always adds 0 would fail no test — the shape tests only check `len(X) == len(Xt)`.

3. **No-op imbalance generator**: A `RandomSamplingClassesGenerator` that returns the input unchanged would pass the shape tests. No test checks the output distribution.

4. **No-op leet speak generator**: A `LeetSpeakGenerator` that returns input unchanged would pass the length-only test.

5. **No-op time series noise**: Both time series noise tests have zero assertions — any implementation that doesn't crash passes.

### Known Failures

- **30 tests in `test_causal.py`** fail because they pass `outlier_magnitude=3.0` to `CausalOutlierPropagationGenerator.generate()`, but the parameter was removed from the method signature (replaced by sampler-based perturbation control).

---

## 5. Domain and Boundary Gaps

### Missing Partitions

- **Zero noise/drift**: No test verifies behavior with `noise_std=0` or `shift_std=0`.
- **Edge case percentages**: `percentage_missing=0` and `percentage_missing=1` are not tested for `MissingValueGenerator`.
- **Single-sample input**: No test verifies behavior with `n_samples=1`.
- **High-dimensional data**: No test with >100 features.
- **Empty input**: No test with `X` having 0 rows.
- **NaN in input**: No test verifies behavior when input already contains NaN.
- **Integer-typed input**: All fixtures use float data; integer input behavior is untested.

### Missing State Transition Tests

- No test verifies that `generate()` is idempotent or that calling it twice produces different results with different seeds.
- No test verifies that generator internal state (e.g., `missing_values_indices_`, `outliers_indices_`) is reset between calls.

### Missing Property Tests

- **Roundtrip property**: For invertible transformations, `inverse(transform(X)) ≈ X`.
- **Monotonicity**: Larger `noise_std` → larger variance increase.
- **Distribution preservation**: For MCAR missingness, the distribution of non-missing values should match the original.

---

## 6. Reliability and Maintainability

### Issues Found

1. **`test_pipelines.py` uses `unittest.TestCase`** while all other tests use pytest-style functions. Inconsistent style.

2. **`test_utils.py` uses `unittest.TestCase`** — same inconsistency.

3. **`test_pipelines.py::test_generate` has no assertions** — the test is effectively dead code.

4. **Duplicate `COMMON_GENERATORS` definition** in `test_outliers.py` — defined twice with slightly different imports.

5. **`test_outliers.py` has duplicate imports** — `numpy`, `pandas`, `default_rng`, `make_blobs`, etc. are imported twice.

6. **Hardcoded magic values**: `n_outliers=10`, `noise_std=1`, `shift_std=0.1` appear throughout without explanation.

7. **No `@pytest.mark.parametrize` for noise_std values** — only one value tested per generator.

### Flaky Test Risk

- Tests using `IsolationForest` (in `test_outliers.py`) depend on sklearn's random state and could be fragile across sklearn versions.
- Tests using `np.linalg.lstsq` (in `test_causal.py`) could be fragile with near-singular matrices.

---

## 7. Prioritized Recommendations

### Critical (Must Fix)

| # | Issue | File | Action |
|---|---|---|---|
| C1 | 30 failing tests due to `outlier_magnitude` API removal | `tests/generators/tabular_data/outliers/test_causal.py` | Remove `outlier_magnitude=...` from all `generator.generate()` calls. The parameter no longer exists; perturbation magnitude is now controlled via the `out_of_distribution_sampler` parameter. |
| C2 | `test_pipelines.py` has zero assertions | `tests/core/test_pipelines.py` | Add assertions: verify `Xt` shape, verify pipeline applies both generators (e.g., check that noise was added AND sampling occurred). |

### High (Should Fix)

| # | Issue | File | Action |
|---|---|---|---|
| H1 | Time series noise tests have zero assertions | `tests/generators/time_series/test_noise.py` | Add assertions: verify shape preserved, verify variance increased, verify noise magnitude matches `noise_std`. |
| H2 | Drift tests only check shape | `tests/generators/tabular_data/test_drift.py` | Add assertions: verify that shifted data differs from original, verify shift magnitude is proportional to `shift_std`. |
| H3 | Imbalance tests only check shape | `tests/generators/tabular_data/test_imbalance.py` | Add assertions: verify output class distribution matches `proportion_classes`, verify sampling with replacement changes row count correctly. |
| H4 | Leet speak test only checks length | `tests/generators/text/test_typos.py` | Add assertion: verify at least some characters were transformed to leet equivalents. |

### Medium (Consider Fixing)

| # | Issue | File | Action |
|---|---|---|---|
| M1 | No zero-parameter edge case tests | Multiple | Add tests for `noise_std=0`, `shift_std=0`, `percentage_missing=0`, `percentage_missing=1`. |
| M2 | Duplicate imports and COMMON_GENERATORS | `tests/generators/tabular_data/test_outliers.py` | Consolidate imports and remove duplicate `COMMON_GENERATORS` definition. |
| M3 | Inconsistent test style (unittest vs pytest) | `tests/core/test_pipelines.py`, `tests/core/test_utils.py` | Convert to pytest-style functions for consistency. |
| M4 | Missing `DummyMissingAtRandom` test | `tests/generators/tabular_data/test_missingness.py` | The test only covers `MissingCompletelyAtRandom` via `__subclasses__()`. Add a test that verifies MAR behavior differs from MCAR. |

### Low (Nice to Have)

| # | Issue | File | Action |
|---|---|---|---|
| L1 | No property-based tests | All | Consider adding `hypothesis`-based tests for invariants (e.g., `noise_std=0 → Xt == X`). |
| L2 | No parametrized edge values | Multiple | Use `@pytest.mark.parametrize` for `noise_std=[0, 0.1, 1, 10]`, `percentage_missing=[0, 0.5, 1]`. |
| L3 | Magic numbers lack documentation | Multiple | Extract magic numbers to named constants or document in test docstrings. |

---

## 8. Residual Risk

1. **Causal outlier generator is untested**: All 30 causal tests fail. The `CausalOutlierPropagationGenerator` has zero effective test coverage despite being one of the most complex generators.

2. **Weak oracles mask behavioral bugs**: Generators for noise (time series), drift, and imbalance could have significant behavioral bugs that the current shape-only tests would not detect.

3. **No integration tests**: There are no tests that chain multiple generators through a `Pipeline` and verify the combined effect.

4. **No regression tests for historical bugs**: The test suite doesn't reference any historical defects, so there's no evidence of regression protection.

5. **No concurrency/thread-safety tests**: If generators are used in parallel pipelines, thread safety is unverified.

6. **No performance regression tests in CI**: While benchmarks exist, they are not run as part of the CI pipeline.

---

## 9. Tests That Need Immediate Updates (Summary)

The following test files **must** be updated:

1. **`tests/generators/tabular_data/outliers/test_causal.py`** — Remove `outlier_magnitude=` from all 30 `generator.generate()` calls. This is the only change needed; the parameter was removed from the API.

2. **`tests/core/test_pipelines.py`** — Add actual assertions to `test_generate`. Currently it calls the pipeline but verifies nothing.

The following test files **should** be updated to strengthen weak oracles:

3. **`tests/generators/time_series/test_noise.py`** — Add assertions beyond "doesn't crash."
4. **`tests/generators/tabular_data/test_drift.py`** — Add behavioral assertions beyond shape checks.
5. **`tests/generators/tabular_data/test_imbalance.py`** — Add distribution verification assertions.
6. **`tests/generators/tabular_data/test_noise.py`** — Add zero-noise and magnitude verification tests.
7. **`tests/generators/text/test_typos.py`** — Add transformation verification for leet speak test.
8. **`tests/generators/tabular_data/test_outliers.py`** — Clean up duplicate imports and COMMON_GENERATORS.
