# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given (apropos credit, this file is largely inspired from https://github.com/audreyfeldroy/cookiecutter-pypackage/blob/master/CONTRIBUTING.rst, thanks for the amazing work!).

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Fraunhofer-IESE/badgers/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Badgers could always use more documentation, whether as part of the
official Badgers docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at URL.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `badgers` for local development.

1. Fork the `badgers` repo on GitHub.
2. Clone your fork locally

```
    $ git clone https://github.com/Fraunhofer-IESE/badgers
```

3. Install dependencies and start your virtualenv:

```
    $ pip install -e ".[dev]"
```

5. Create a branch for local development:

```
    $ git checkout -b name-of-your-bugfix-or-feature
```

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

```
    $ tox
```

   You can also run tests directly with pytest:

```
    $ pytest -v tests
```

7. Commit your changes and push your branch to GitHub:

```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.11, 3.12, 3.13, and 3.14. Check
   https://github.com/Fraunhofer-IESE/badgers/actions and make sure that the tests pass for all supported Python versions.

## Testing

Tests live under `tests/` and use **pytest** with function-based tests (no `unittest.TestCase`
classes). The test structure mirrors the source layout:

```
tests/
├── conftest.py                  # shared fixtures (e.g., rng)
├── core/                        # tests for badgers.core
├── benchmarks/                  # tests for the benchmarks package
└── generators/
    ├── tabular_data/
    │   ├── conftest.py          # tabular-specific fixtures
    │   ├── test_outliers.py
    │   ├── test_noise.py
    │   └── ...
    ├── time_series/
    │   ├── conftest.py          # time-series fixtures (sine, random walk)
    │   └── ...
    ├── graph/
    │   ├── conftest.py          # graph fixtures (Erdős-Rényi)
    │   └── ...
    └── text/
        ├── conftest.py          # text fixtures (word list)
        └── ...
```

### Key conventions

- **Fixtures over classes**: each data type has a `conftest.py` with reusable fixtures (e.g.
  `tabular_small`, `time_series_sine`, `graph_erdos_renyi`, `text_word_list`). Fixtures return
  the input data `X` directly (with class values as a column of `X` where labeled data is
  needed) and use deterministic random seeds.
- **Flat test functions**: tests are plain `test_<subject>__<behavior>` functions (double
  underscore separator), not methods on `TestCase` classes.
- **Generator tests are independent**: generator correctness is tested directly via pytest, not
  through the benchmark framework's functional checks. This keeps tests fast, focused, and free
  of benchmark infrastructure coupling.

See [ADR-0001](docs/architecture/adr/0001-tests-vs-benchmarks.md) and
[ADR-0002](docs/architecture/adr/0002-per-data-type-fixtures.md) for the reasoning.

### Running tests

```bash
# Run all tests
pytest tests/

# Run only generator tests
pytest tests/generators/

# Run a specific category
pytest tests/generators/tabular_data/ -v
```

## Architecture contracts

The layering described in the [architecture docs](docs/architecture.md) is enforced by
[`import-linter`](https://import-linter.readthedocs.io/). Run it before opening a PR:

```bash
pip install import-linter
lint-imports
```

A broken contract means an import violates the documented layering — fix the import, not the
contract. See [ADR-0007](docs/architecture/adr/0007-import-contracts.md).

## PR Submission Checklist

Before opening a pull request, ensure you have:

- [ ] Run `tox` locally and all environments pass
- [ ] Run `pytest -v tests` locally and all tests pass
- [ ] Run `flake8 badgers tests --count --select=E9,F63,F7,F82 --show-source --statistics` with zero errors
- [ ] Run `lint-imports` with zero broken contracts
- [ ] Added tests for any new functionality
- [ ] Updated documentation (docstrings, README, or mkdocs) for any new features
- [ ] Verified your branch is up to date with `main`


