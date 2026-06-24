"""Predefined scenarios for time series generators."""
import numpy as np
from benchmarks.models import Scenario


def _sine_wave_factory(rng):
    t = np.linspace(0, 4 * np.pi, 200)
    X = np.sin(t).reshape(-1, 1)
    return X, None


def _random_walk_factory(rng):
    steps = rng.normal(0, 1, size=(200, 1))
    X = np.cumsum(steps, axis=0)
    return X, None


SCENARIO_SINE_WAVE = Scenario(
    name="sine_wave_200",
    data_type="time_series",
    factory=_sine_wave_factory,
    tags=["small", "1D", "periodic"],
)

SCENARIO_RANDOM_WALK = Scenario(
    name="random_walk_200",
    data_type="time_series",
    factory=_random_walk_factory,
    tags=["small", "1D", "stochastic"],
)