import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.contour import QuadContourSet
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "tutorials"
    / "use-cases"
    / "defect_impact_showcase.ipynb"
)


def load_notebook_function(notebook_path, function_name, namespace):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and any(
            isinstance(node, ast.FunctionDef) and node.name == function_name
            for node in ast.parse("".join(cell.get("source", []))).body
        )
    )
    function_node = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace[function_name]


def test_regression_metrics_compare_predictions_with_continuous_y():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def evaluate(" in "".join(cell.get("source", []))
    )
    namespace = {
        "np": np,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    class Predictor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def predict(self, X):
            return self.values[: len(X)]

        def score_samples(self, X):
            return np.zeros(len(X))

    Xt = np.array([[0.0, 10.0, 0.0], [1.0, 20.0, 1.0]])
    predictors = {
        "clf": Predictor([0, 1]),
        "reg": Predictor([9.0, 20.0]),
        "det": Predictor([1, 1]),
    }
    scenario = SimpleNamespace(
        name="synthetic", family="test", observed=None
    )

    metrics, _ = namespace["evaluate"](
        scenario, predictors, Xt, np.array([0, 1])
    )

    assert metrics["reg_mse"] == 0.5


def test_classifier_renderer_uses_caller_axes_and_draws_map_and_points():
    overlay_calls = []

    def overlay(ax, X2, y_true, observed=None, wrong=None):
        overlay_calls.append((ax, X2.copy(), y_true.copy(), wrong.copy()))
        return ax.scatter(X2[:, 0], X2[:, 1])

    namespace = {
        "np": np,
        "overlay": overlay,
        "wrong_clf": lambda frame: (
            frame["prediction_clf"].to_numpy()
            != frame["cls"].to_numpy()
        ),
    }
    plot_classifier = load_notebook_function(
        NOTEBOOK, "plot_classifier", namespace
    )
    predictions = pd.DataFrame(
        {
            "x": [-0.5, 0.5],
            "y": [-0.25, 0.25],
            "cls": [0, 1],
            "prediction_clf": [0, 0],
        }
    )
    grid = SimpleNamespace(
        xx0=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        xx1=np.array([[-1.0, -1.0], [1.0, 1.0]]),
    )
    rgba = np.ones((2, 2, 4))
    fig, ax = plt.subplots()
    open_figures = set(plt.get_fignums())

    try:
        result = plot_classifier(ax, predictions, grid, rgba)

        assert result is ax
        assert set(plt.get_fignums()) == open_figures
        assert len(ax.images) == 1
        assert ax.images[0].axes is ax
        assert len(ax.collections) == 1
        assert ax.collections[0].axes is ax
        assert overlay_calls[0][0] is ax
        np.testing.assert_array_equal(overlay_calls[0][3], [False, True])
    finally:
        plt.close(fig)


def test_detector_renderer_uses_caller_axes_for_score_contours_and_points():
    overlay_calls = []

    def overlay(ax, X2, y_true, observed=None, wrong=None):
        overlay_calls.append(ax)
        return ax.scatter(X2[:, 0], X2[:, 1])

    namespace = {
        "np": np,
        "overlay": overlay,
    }
    plot_detector = load_notebook_function(
        NOTEBOOK, "plot_detector", namespace
    )
    predictions = pd.DataFrame(
        {
            "x": [-0.5, 0.5],
            "y": [-0.25, 0.25],
            "cls": [0, 1],
        }
    )
    xx0, xx1 = np.meshgrid([-1.0, 1.0], [-1.0, 1.0])
    grid = SimpleNamespace(
        xx0=xx0,
        xx1=xx1,
        response=np.array([[-0.2, 0.2], [-0.2, 0.2]]),
    )
    fig, ax = plt.subplots()
    open_figures = set(plt.get_fignums())

    try:
        result = plot_detector(ax, predictions, grid)

        assert result is ax
        assert set(plt.get_fignums()) == open_figures
        assert sum(
            isinstance(collection, QuadContourSet)
            for collection in ax.collections
        ) == 2
        assert sum(
            isinstance(collection, PathCollection)
            for collection in ax.collections
        ) == 1
        assert all(collection.axes is ax for collection in ax.collections)
        assert overlay_calls == [ax]
    finally:
        plt.close(fig)


def test_regression_renderer_conditions_four_step_curves_on_class():
    class Regressor:
        def __init__(self):
            self.inputs = []

        def predict(self, X):
            self.inputs.append(X.copy())
            return X[:, 0] + 10 * X[:, 1]

    regressor = Regressor()
    predictions = {
        "x": [-1.0, -0.5, 0.5, 1.0],
        "y": [-0.8, -0.4, 0.7, 0.9],
        "cls": [0, 1, 2, 3],
        "prediction_reg": [-0.9, -0.3, 0.8, 1.1],
    }
    namespace = {
        "np": np,
        "CLASS_COLORS": ["#2878b5", "#e69f00", "#009e73", "#d55e00"],
        "CLASS_MARKERS": ["o", "s", "^", "D"],
        "wrong_reg": lambda frame: np.abs(
            frame["prediction_reg"].to_numpy() - frame["y"].to_numpy()
        ) > 0.5,
        "darken": lambda color: color,
    }
    plot_regression = load_notebook_function(
        NOTEBOOK, "plot_regression", namespace
    )
    frame = pd.DataFrame(predictions)
    x_grid = np.linspace(-1.0, 1.0, 5)
    fig, ax = plt.subplots()
    open_figures = set(plt.get_fignums())

    try:
        result = plot_regression(ax, frame, regressor, x_grid)

        assert result is ax
        assert set(plt.get_fignums()) == open_figures
        assert len(regressor.inputs) == 4
        assert len(ax.lines) == 4
        assert all(line.get_linewidth() < 2.0 for line in ax.lines)
        for cls, model_input in enumerate(regressor.inputs):
            np.testing.assert_array_equal(
                model_input[:, 1], np.full(len(x_grid), cls)
            )
            np.testing.assert_allclose(
                ax.lines[cls].get_ydata(),
                model_input[:, 0] + 10 * cls,
            )
        assert len(ax.collections) > 0
        observed_points = np.concatenate(
            [collection.get_offsets() for collection in ax.collections]
        )
        np.testing.assert_allclose(
            observed_points,
            frame[["x", "y"]].to_numpy(),
        )
    finally:
        plt.close(fig)
