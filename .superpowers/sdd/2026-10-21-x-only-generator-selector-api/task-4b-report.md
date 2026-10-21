# Task 4b Report: Tabular Imbalance Generators

## Status

Implemented the X-only migration for ordinary and causal tabular imbalance
generators. The changes are limited to the requested generator modules and
focused tests.

## Changes

- Removed `y` parameters, `yt` slicing, and tuple returns from the imbalance
  generators and their documentation.
- Changed selector invocation to `selector(X)`.
- Preserved sampling behavior and `get_imbalance_labels()` semantics.
- Removed separate target-row filtering from causal imbalance; generated rows
  are now represented by the filtered X alone.
- Updated focused tests to invoke generators with X only, capture one return
  value, and check class counts and dropped-row alignment using X columns and
  imbalance labels.

## TDD Evidence

Updated tests before production code. The initial focused test run failed as
expected because the old implementation returned an `(Xt, yt)` tuple, rather
than the single X object asserted by the updated test.

## Verification

- `python -m pytest -v tests/generators/tabular_data/test_imbalance.py` —
  **36 passed**.
- `python -m pytest -v tests/generators/tabular_data/imbalance/test_causal.py`
  — **7 skipped** because `dowhy.gcm` is not installed.
- A standalone causal X-only selector smoke test passed, including output-row
  selection and getter alignment.
- Problems reported no errors in the four changed source/test files.
- Focused Flake8 syntax/name check (`E9,F63,F7,F82`) — **0 errors**.
- `git diff --check` — clean.

## Concerns

The causal test module's existing tests all skip in this environment due to the
missing optional `dowhy.gcm` dependency; the selector branch was additionally
smoke-tested without that dependency. An untracked `docs/examples/` directory was present at final check and could
not be removed because another process held it open; it is not included in the
task commit.

## Regression Finding Follow-up

- Added `test_imbalance_labels__classes_generator_tracks_most_recent_call`,
  an X-only regression test that invokes the same generator twice with
  different drop masks and verifies both outputs and the getter after the
  second call.
- The current implementation correctly replaces `imbalance_labels_` on each
  generation, so no production code change was needed. For RED evidence, the
  new test was run against a temporary stale-label mutant that retained the
  first mask; it failed at the second-call getter assertion as intended. The
  production implementation was restored before GREEN verification.
- `python -m pytest -v tests/generators/tabular_data/test_imbalance.py` —
  **37 passed**.
- `python -m pytest -v tests/generators/tabular_data/imbalance/test_causal.py`
  — **7 skipped** because optional `dowhy.gcm` is unavailable.
- A direct causal X-only selector smoke test invoked `generate()` twice,
  checked the second output, and confirmed the getter's labels align with that
  latest output.
- Focused Flake8 syntax/name checks — **0 errors**; `git diff --check` — clean.
