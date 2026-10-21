# Task 1 Report: X-only Generator and Pipeline Contract

## Changes

- Changed `GeneratorMixin.generate` to accept `X` and operation parameters only, removed its `Tuple` import, and documented a single `Xt` return.
- Changed `Pipeline.generate` to accept `X` and optional named parameters only. Each generator's single result becomes the next generator's `X`, and the pipeline returns the final `X` directly.
- Reworked pipeline contract tests to use simple X-only fakes, covering input/output chaining, direct return identity, insertion order, parameter routing, and empty pipelines.
- No legacy generator or selector implementation was changed.

## TDD

### RED

Command:

```text
python -m pytest tests/core/test_pipelines.py -v
```

Output:

```text
FAILED tests/core/test_pipelines.py::test_pipeline__passes_and_returns_single_X_result
TypeError: Pipeline.generate() missing 1 required positional argument: 'y'
1 failed, 12 passed in 7.13s
```

The new X-only call failed for the expected reason: the existing pipeline still required `y`.

The bare command `pytest tests/core/test_pipelines.py -v` was also attempted, but `pytest` was not available as a shell command. Using `python -m pytest` successfully invoked the installed test runner.

### GREEN

Command:

```text
python -m pytest tests/core/test_pipelines.py tests/generators/tabular_data/causal/test_base.py -v
```

Output:

```text
23 passed in 0.21s
```

## Additional Validation

Full suite:

```text
python -m pytest -q tests
423 passed, 90 skipped, 4 warnings, 3 subtests passed in 8.37s
```

Lint:

```text
python -m flake8 badgers tests --count --select=E9,F63,F7,F82 --show-source --statistics
0
```

`git diff --check` passed. The full test run emitted an environment `RequestsDependencyWarning` and four existing `LowDensitySamplingGenerator` warnings; no test failures resulted.
