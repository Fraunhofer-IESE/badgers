# Baseline Branch for v0.0.13 — Design Spec

**Date**: 2026-06-24
**Branch**: `baseline_0_0_13` (to be created from `dev_0_0_13`)

## Goal

Create a baseline branch that captures the performance characteristics of all tabular data generators at version 0.0.13, using the existing `benchmarks` module. This baseline will serve as the reference point for future optimization work on outlier generators.

## Scope

- Add missing tabular generator registrations to the benchmark framework
- Run benchmarks and save a named baseline
- Commit everything to a dedicated branch

Out of scope: any generator code changes, optimization, or non-tabular generators.

## Steps

### 1. Create branch

```bash
git checkout -b baseline_0_0_13 dev_0_0_13
```

### 2. Add missing benchmark registrations

Add `register()` calls to the existing `benchmarks/generators/tabular_data/_*.py` files for all concrete generator classes not yet registered.

#### `_outliers.py` — add 4 generators

| Generator | Default params | Scenarios |
|---|---|---|
| `HistogramSamplingGenerator` | `n_outliers=10, bins=10, threshold_low_density=0.1` | `SCENARIO_SMALL_BLOBS` only (raises on >5D) |
| `ZScoreSamplingGenerator` | `n_outliers=10, scale=1.0` | `SCENARIO_SMALL_BLOBS`, `SCENARIO_MEDIUM_BLOBS` |
| `LowDensitySamplingGenerator` | `n_outliers=10, threshold_low_density=0.1` | `SCENARIO_SMALL_BLOBS`, `SCENARIO_MEDIUM_BLOBS` |
| `HypersphereSamplingGenerator` | `n_outliers=10, scale=1.0` | `SCENARIO_SMALL_BLOBS`, `SCENARIO_MEDIUM_BLOBS` |

#### `_noise.py` — add 1 generator

| Generator | Default params | Scenarios |
|---|---|---|
| `GaussianNoiseClassesGenerator` | `noise_std_per_class={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}` | `SCENARIO_SMALL_BLOBS`, `SCENARIO_MEDIUM_BLOBS` |

#### `_drift.py` — add 1 generator

| Generator | Default params | Scenarios |
|---|---|---|
| `RandomShiftClassesGenerator` | `shift_std=0.1` | `SCENARIO_SMALL_BLOBS`, `SCENARIO_MEDIUM_BLOBS` |

#### `_imbalance.py` — add 2 generators

| Generator | Default params | Scenarios |
|---|---|---|
| `RandomSamplingClassesGenerator` | `proportion_classes={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}` | `SCENARIO_SMALL_BLOBS` |
| `RandomSamplingTargetsGenerator` | `{}` (no params; uses built-in default `lambda y: normalize_proba(y)`) | `SCENARIO_SMALL_BLOBS` |

### 3. Run benchmarks

```bash
python -m benchmarks run --iterations 10
```

Using 10 iterations for more stable baseline statistics.

### 4. Save baseline

```bash
python -m benchmarks baseline save --name v0.0.13
```

### 5. Commit

```bash
git add benchmarks/generators/tabular_data/_*.py benchmarks/baselines/v0.0.13.json
git commit -m "bench: add missing tabular generator registrations and v0.0.13 baseline"
```

## Verification

- `python -m benchmarks run` completes without errors
- `python -m benchmarks baseline list` shows `v0.0.13`
- `pytest -v tests` passes (no generator code changes, so this is a sanity check)
