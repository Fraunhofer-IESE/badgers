# AGENTS.md — AI Coding Agent Instructions

## Project: Badgers

Badgers is a Python library for generating bad (low-quality) data for testing and benchmarking data science pipelines.

## Before Submitting a PR

Always run these checks locally before opening a pull request:

### 1. Run the full test suite with tox

```bash
tox
```

This runs tests against all supported Python versions (3.11, 3.12, 3.13).

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

- Python 3.11, 3.12, 3.13
- Python 3.8, 3.9, and 3.10 are **no longer supported** (EOL or incompatible with NumPy 2.x)

## Testing

- Test framework: **pytest**
- Tests live in `tests/`
- Fixtures are defined in `tests/conftest.py`
- Run a subset: `pytest -v tests/generators/tabular_data/`

## CI/CD

- GitHub Actions workflow: `.github/workflows/tests.yml`
- Runs on push/PR to `main`
- Matrix: Python 3.11–3.13 × Ubuntu, macOS, Windows
- Linting: flake8 (syntax errors + complexity)
- Tests: `tox -e py`
