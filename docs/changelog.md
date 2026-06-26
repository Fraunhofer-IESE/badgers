# Changelog

See https://github.com/Fraunhofer-IESE/badgers/releases

## dev_0_0_14 (unreleased)

### Added

- **Complete tabular generator benchmark coverage**: All 16 tabular generators are now registered in the benchmark framework (up from 7). Added registrations for `HistogramSamplingGenerator`, `ZScoreSamplingGenerator`, `LowDensitySamplingGenerator`, `HypersphereSamplingGenerator`, `GaussianNoiseClassesGenerator`, `RandomShiftClassesGenerator`, `RandomSamplingClassesGenerator`, and `RandomSamplingTargetsGenerator`.

### Changed

- **Benchmark outputs moved to `.benchmarks/`**: Results and baselines are now stored in `.benchmarks/results/` and `.benchmarks/baselines/` (gitignored) instead of `benchmarks/results/` and `benchmarks/baselines/`.

## dev_0_0_13 (released)

### Changed

- **Separated tests from benchmarks**: Generator correctness tests now live in `tests/generators/` as standalone pytest functions with fixtures, independent of the benchmark framework. The `benchmarks/checks/` directory and `FunctionalCheck` infrastructure have been removed. Benchmarks now focus exclusively on performance measurement (time/memory). (9 commits: `943e230`..`56fee5d`)

### Added

- Pytest fixtures for each data type: `tabular_small`/`tabular_large`, `time_series_sine`/`time_series_walk`, `graph_erdos_renyi`, `text_word_list`
- Test conftest files in `tests/generators/{tabular_data,time_series,graph,text}/`

### Removed

- `benchmarks/checks/` directory and all `FunctionalCheck` infrastructure
- `--type functional` option from benchmark CLI
- `functional_checks` parameter from `GeneratorBenchmark` model