"""Golden pins: bit-identical behavior of the classic missingness
generators across the selector refactor. Self-healing: creates the golden
file on first run, asserts against it afterwards."""

from pathlib import Path

import numpy as np
import pytest

from badgers.generators.tabular_data.missingness.missingness import (
    DummyMissingAtRandom,
    DummyMissingNotAtRandom,
    MissingCompletelyAtRandom,
)

GOLDEN = Path(__file__).parent / "golden_classic_missingness.npz"
CLASSES = {
    "mcar": MissingCompletelyAtRandom,
    "mar": DummyMissingAtRandom,
    "mnar": DummyMissingNotAtRandom,
}
# Deterministic input: 40 rows x 3 cols, no constant columns (constant
# columns divide by zero in the classic MAR/MNAR normalizations).
RNG = np.random.default_rng(123)
X0 = RNG.normal(size=(40, 3)) @ np.diag([1.0, 5.0, 0.5]) + np.array([0.0, 10.0, -3.0])


def _entries():
    out = {}
    for seed in (0, 1, 2):
        for key, cls in CLASSES.items():
            gen = cls(random_generator=np.random.default_rng(seed))  # explicit seed!
            Xt = gen.generate(X0.copy(), percentage_missing=0.25)
            out[f"{key}_{seed}_nanmask"] = np.isnan(Xt)
            idx = sorted(gen.missing_values_indices_)
            out[f"{key}_{seed}_indices"] = np.array(idx, dtype=int)
    return out


@pytest.mark.skipif(GOLDEN.exists(), reason="golden file exists: assert mode")
def test_create_golden():
    np.savez(GOLDEN, **_entries())


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden file missing: capture mode")
def test_assert_golden():
    golden = np.load(GOLDEN)
    for key, value in _entries().items():
        np.testing.assert_array_equal(value, golden[key])
