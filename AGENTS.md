# AGENTS.md — AI Coding Agent Instructions

## Project: Badgers

Badgers is a Python library for generating bad (low-quality) data for testing and benchmarking data science pipelines.

## Before Submitting a PR

Always run these checks locally before opening a pull request:

### 1. Run the full test suite with tox

```bash
tox
```

This runs tests against all supported Python versions (3.11, 3.12, 3.13, 3.14).

### 2. Run tests directly with pytest

```bash
pytest -v tests
```

### 3. Run linting

```bash
flake8 badgers tests --count --select=E9,F63,F7,F82 --show-source --statistics
```

This must produce **zero** errors before submitting.

### 4. PR Submission Checklist

- [ ] `tox` passes on all environments
- [ ] `pytest -v tests` passes locally
- [ ] `flake8` produces zero syntax/name errors
- [ ] New functionality has tests
- [ ] Documentation is updated (docstrings, README, or mkdocs)
- [ ] Branch is rebased on latest `main`

## Supported Python Versions

- Python 3.11, 3.12, 3.13, 3.14
- Python 3.8, 3.9, and 3.10 are **no longer supported** (EOL or incompatible with NumPy 2.x)

## Testing

- Test framework: **pytest**
- Tests live in `tests/`
- Fixtures are defined in `tests/conftest.py`
- Run a subset: `pytest -v tests/generators/tabular_data/`

## CI/CD

- GitHub Actions workflow: `.github/workflows/tests.yml`
- Runs on push/PR to `main`
- Matrix: Python 3.11–3.14 × Ubuntu, macOS, Windows
- Linting: flake8 (syntax errors + complexity)
- Tests: `tox -e py`

## Performance Patterns

### Prefer Vectorized NumPy Over Python Loops

Badgers generators often produce many samples at once. Always prefer vectorized
NumPy operations over Python-level `for` loops, especially when `n_samples` can
be large.

**Rule of thumb:** If you see a list comprehension building a `np.array` row by
row, it can almost certainly be vectorized.

#### Pattern 1: Replace per-row RNG calls with batched calls

```python
# ❌ Slow — calls random_sign + exponential n_outliers times
outliers = np.array([
    random_sign(rng, size=d) * (3.0 + rng.exponential(scale=s, size=d))
    for _ in range(n_outliers)
])

# ✅ Fast — one batched call each
signs = random_sign(rng, size=(n_outliers, d))
exponentials = rng.exponential(scale=s, size=(n_outliers, d))
outliers = signs * (3.0 + exponentials)
```

#### Pattern 2: Replace per-row .iloc/.choice with integer array indexing

```python
# ❌ Slow — calls .iloc[:, i] and .choice per column in a Python loop
np.stack([rng.choice(X.iloc[:, i], size=n) for i in range(X.shape[1])]).T

# ✅ Fast — generate all indices at once, index into .values
row_idx = rng.integers(0, len(X), size=(n, X.shape[1]))
X.values[row_idx, np.arange(X.shape[1])]
```

#### Pattern 3: Provide batch variants for utility functions

When a utility function operates on a single sample (e.g.,
`random_spherical_coordinate`), also provide a batch version
(`random_spherical_coordinates`) that accepts arrays and uses `axis=`
parameters. This avoids forcing callers into Python loops.

#### When NOT to vectorize

- When `n_samples` is always small (≤10) — the allocation overhead of 2D arrays
  can outweigh the benefit. Measure first.
- When the algorithm inherently requires sequential state (e.g., rejection
  sampling with unknown iteration count).

### Benchmark Before and After

Use the project's benchmark framework to measure performance changes:

```bash
# Run all benchmarks
python -m benchmarks run --iterations 10

# Run a subset
python -m benchmarks run --generators tabular_data.outliers --iterations 10

# Compare against a baseline
python -m benchmarks compare --baseline v0.0.13
```

Results are saved to `.benchmarks/results/`. Baselines live in
`.benchmarks/baselines/`.

### Avoid Expensive pandas .iloc in Loops

`DataFrame.iloc[:, i]` inside a loop is expensive because each call creates a
new view object and triggers index lookups. When you need per-column random
access, work with the underlying `.values` NumPy array instead, or use
`.to_numpy()`.
