# ADR-0002: Per-data-type pytest fixture architecture

- **Date:** 2026-06-23
- **Status:** Accepted

## Context

Tests were written as `unittest.TestCase` classes with a monolithic fixture module. Fixtures
were far from the tests that used them, and parametrizing across input types was awkward.

## Decision

Use per-data-type `conftest.py` files with parametrized pytest fixtures.

```
tests/
  conftest.py                          # shared: rng fixture only
  generators/
    tabular_data/conftest.py           # tabular_data, tabular_data_labeled
    time_series/conftest.py            # time_series_sine, time_series_walk
    graph/conftest.py                  # graph_erdos_renyi
    text/conftest.py                   # text_word_list
```

Naming conventions:

- Fixtures are prefixed with the data type (`tabular_data`, `time_series_sine`,
  `graph_erdos_renyi`, `text_word_list`).
- Test functions are `test_<subject>__<behavior>` with a double underscore separator, e.g.
  `test_gaussian_noise__increases_variance`.
- No `unittest.TestCase` classes — flat pytest functions only.
- Only numpy and pandas input types (no `list_1D` / `list_2D`).

## Consequences

- Fixtures are auto-discovered by pytest's `conftest.py` mechanism, scoped to the nearest
  directory, so test data stays close to the tests that use it.
- No single monolithic fixture file.
- pytest's built-in parametrization covers the input-type matrix.
