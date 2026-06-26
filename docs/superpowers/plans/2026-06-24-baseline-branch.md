# Baseline Branch for v0.0.13 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create branch `baseline_0_0_13`, add missing tabular generator benchmark registrations, run benchmarks, and save a v0.0.13 baseline.

**Architecture:** Follows the existing benchmark registration pattern — each `benchmarks/generators/tabular_data/_*.py` file imports generator classes and calls `register(GeneratorBenchmark(...))`. No new files or infrastructure.

**Tech Stack:** Python 3.11+, badgers 0.0.13, existing benchmarks module

## Global Constraints

- Python >= 3.11 (per pyproject.toml)
- No generator code changes — only benchmark registration additions
- Follow existing patterns in `benchmarks/generators/tabular_data/_*.py`
- `HistogramSamplingGenerator` must only use `SCENARIO_SMALL_BLOBS` (2D) — it raises on >5D
- Classes-based generators (`GaussianNoiseClassesGenerator`, `RandomSamplingClassesGenerator`) need class labels matching blobs scenario (5 centers → classes 0-4)
- `RandomSamplingTargetsGenerator` takes no params (uses built-in default lambda)

---

### Task 1: Create baseline branch

**Files:**
- None (git operation only)

**Interfaces:**
- Produces: branch `baseline_0_0_13` from `dev_0_0_13`

- [ ] **Step 1: Create and switch to the baseline branch**

```bash
git checkout -b baseline_0_0_13 dev_0_0_13
```

Expected: `Switched to a new branch 'baseline_0_0_13'`

- [ ] **Step 2: Verify current branch**

```bash
git branch --show-current
```

Expected: `baseline_0_0_13`

- [ ] **Step 3: Commit** (no code changes, just branch creation — skip commit)

---

### Task 2: Add missing outlier benchmark registrations

**Files:**
- Modify: `benchmarks/generators/tabular_data/_outliers.py`

**Interfaces:**
- Consumes: `GeneratorBenchmark`, `register`, scenarios from `benchmarks.scenarios.tabular`
- Produces: 4 new `register()` calls for `HistogramSamplingGenerator`, `ZScoreSamplingGenerator`, `LowDensitySamplingGenerator`, `HypersphereSamplingGenerator`

- [ ] **Step 1: Add imports for the 4 missing outlier generators**

Add these import lines after the existing outlier imports in `benchmarks/generators/tabular_data/_outliers.py`:

```python
from badgers.generators.tabular_data.outliers.low_density_sampling import (
    HistogramSamplingGenerator,
    LowDensitySamplingGenerator,
)
from badgers.generators.tabular_data.outliers.distribution_sampling import (
    ZScoreSamplingGenerator,
    HypersphereSamplingGenerator,
)
```

- [ ] **Step 2: Add register() calls for the 4 generators**

Append after the existing `IndependentHistogramsGenerator` registration:

```python
register(GeneratorBenchmark(
    generator_cls=HistogramSamplingGenerator,
    name="HistogramSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "bins": 10, "threshold_low_density": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS],  # only 2D — raises on >5D
))

register(GeneratorBenchmark(
    generator_cls=ZScoreSamplingGenerator,
    name="ZScoreSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "scale": 1.0},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=LowDensitySamplingGenerator,
    name="LowDensitySampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "threshold_low_density": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=HypersphereSamplingGenerator,
    name="HypersphereSampling",
    module_path="tabular_data.outliers",
    default_params={"n_outliers": 10, "scale": 1.0},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
```

- [ ] **Step 3: Verify the file parses correctly**

```bash
python -c "import benchmarks.generators.tabular_data._outliers"
```

Expected: no output (no errors)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generators/tabular_data/_outliers.py
git commit -m "bench: register missing outlier generators (HistogramSampling, ZScoreSampling, LowDensitySampling, HypersphereSampling)"
```

---

### Task 3: Add missing noise benchmark registration

**Files:**
- Modify: `benchmarks/generators/tabular_data/_noise.py`

**Interfaces:**
- Consumes: `GeneratorBenchmark`, `register`, scenarios from `benchmarks.scenarios.tabular`
- Produces: 1 new `register()` call for `GaussianNoiseClassesGenerator`

- [ ] **Step 1: Add import for GaussianNoiseClassesGenerator**

Add after the existing `GaussianNoiseGenerator` import in `benchmarks/generators/tabular_data/_noise.py`:

```python
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator, GaussianNoiseClassesGenerator
```

- [ ] **Step 2: Add register() call**

Append after the existing `GaussianNoise` registration:

```python
register(GeneratorBenchmark(
    generator_cls=GaussianNoiseClassesGenerator,
    name="GaussianNoiseClasses",
    module_path="tabular_data.noise",
    default_params={"noise_std_per_class": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
```

- [ ] **Step 3: Verify the file parses correctly**

```bash
python -c "import benchmarks.generators.tabular_data._noise"
```

Expected: no output (no errors)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generators/tabular_data/_noise.py
git commit -m "bench: register GaussianNoiseClassesGenerator"
```

---

### Task 4: Add missing drift benchmark registration

**Files:**
- Modify: `benchmarks/generators/tabular_data/_drift.py`

**Interfaces:**
- Consumes: `GeneratorBenchmark`, `register`, scenarios from `benchmarks.scenarios.tabular`
- Produces: 1 new `register()` call for `RandomShiftClassesGenerator`

- [ ] **Step 1: Add import for RandomShiftClassesGenerator**

Add after the existing `RandomShiftGenerator` import in `benchmarks/generators/tabular_data/_drift.py`:

```python
from badgers.generators.tabular_data.drift import RandomShiftGenerator, RandomShiftClassesGenerator
```

- [ ] **Step 2: Add register() call**

Append after the existing `RandomShift` registration:

```python
register(GeneratorBenchmark(
    generator_cls=RandomShiftClassesGenerator,
    name="RandomShiftClasses",
    module_path="tabular_data.drift",
    default_params={"shift_std": 0.1},
    scenarios=[SCENARIO_SMALL_BLOBS, SCENARIO_MEDIUM_BLOBS],
))
```

- [ ] **Step 3: Verify the file parses correctly**

```bash
python -c "import benchmarks.generators.tabular_data._drift"
```

Expected: no output (no errors)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generators/tabular_data/_drift.py
git commit -m "bench: register RandomShiftClassesGenerator"
```

---

### Task 5: Add missing imbalance benchmark registrations

**Files:**
- Modify: `benchmarks/generators/tabular_data/_imbalance.py`

**Interfaces:**
- Consumes: `GeneratorBenchmark`, `register`, scenarios from `benchmarks.scenarios.tabular`
- Produces: 2 new `register()` calls for `RandomSamplingClassesGenerator`, `RandomSamplingTargetsGenerator`

- [ ] **Step 1: Add imports for the 2 missing imbalance generators**

Add after the existing `RandomSamplingFeaturesGenerator` import in `benchmarks/generators/tabular_data/_imbalance.py`:

```python
from badgers.generators.tabular_data.imbalance import (
    RandomSamplingFeaturesGenerator,
    RandomSamplingClassesGenerator,
    RandomSamplingTargetsGenerator,
)
```

- [ ] **Step 2: Add register() calls**

Append after the existing `RandomSamplingFeatures` registration:

```python
register(GeneratorBenchmark(
    generator_cls=RandomSamplingClassesGenerator,
    name="RandomSamplingClasses",
    module_path="tabular_data.imbalance",
    default_params={"proportion_classes": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}},
    scenarios=[SCENARIO_SMALL_BLOBS],
))

register(GeneratorBenchmark(
    generator_cls=RandomSamplingTargetsGenerator,
    name="RandomSamplingTargets",
    module_path="tabular_data.imbalance",
    default_params={},
    scenarios=[SCENARIO_SMALL_BLOBS],
))
```

- [ ] **Step 3: Verify the file parses correctly**

```bash
python -c "import benchmarks.generators.tabular_data._imbalance"
```

Expected: no output (no errors)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generators/tabular_data/_imbalance.py
git commit -m "bench: register RandomSamplingClassesGenerator and RandomSamplingTargetsGenerator"
```

---

### Task 6: Run benchmarks and save baseline

**Files:**
- Create: `.benchmarks/results/run_baseline_0_0_13_<timestamp>.json` (auto-generated by CLI, gitignored)
- Create: `.benchmarks/baselines/v0.0.13.json` (via `baseline save`, gitignored)

**Interfaces:**
- Consumes: all registered `GeneratorBenchmark` instances (from Tasks 2-5)
- Produces: baseline JSON file at `.benchmarks/baselines/v0.0.13.json` (gitignored)

- [ ] **Step 1: Run benchmarks with 10 iterations**

```bash
python -m benchmarks run --iterations 10
```

Expected: output showing generators and scenarios count, results saved to `.benchmarks/results/run_baseline_0_0_13_<timestamp>.json`

- [ ] **Step 2: Save the results as a named baseline**

```bash
python -m benchmarks baseline save --name v0.0.13
```

Expected: `Baseline 'v0.0.13' saved from run_baseline_0_0_13_<timestamp>.json`

- [ ] **Step 3: Verify the baseline exists**

```bash
python -m benchmarks baseline list
```

Expected: `v0.0.13` appears in the list

- [ ] **Step 4: Commit** (skip — baselines are gitignored, no files to commit)

---

### Task 7: Final verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v tests
```

Expected: all tests pass (no generator code was changed, so this is a sanity check)

- [ ] **Step 2: Verify the baseline can be used for comparison**

```bash
python -m benchmarks compare --baseline v0.0.13
```

Expected: comparison report showing all generators as "new generator" (since there's no older baseline to compare against — this just validates the JSON is well-formed and loadable)

- [ ] **Step 3: Push the branch**

```bash
git push -u origin baseline_0_0_13
```
