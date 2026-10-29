import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
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
    target_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and any(
            isinstance(node, ast.FunctionDef) and node.name == function_name
            for node in ast.parse("".join(cell.get("source", []))).body
        )
    )
    # Pre-populate the namespace with every helper defined in other cells
    # (e.g. scatter_classes / outlier_marks live in the setup cell), so the
    # lifted functions resolve through the notebook's own namespace plus the
    # supplied externals.
    for cell in notebook["cells"]:
        if cell is target_cell or cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        helper_nodes = [
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        ]
        if helper_nodes:
            exec(compile(ast.Module(body=helper_nodes, type_ignores=[]),
                         str(notebook_path), "exec"), namespace)
    source = "".join(target_cell.get("source", []))
    tree = ast.parse(source)
    # module-level constants the lifted functions reference as default
    # values (e.g. overlay's ms=POINT_MS) must exist before the defs run
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) for t in node.targets
        ):
            names = {t.id for t in node.targets}
            if names <= {"POINT_MS", "POINT_EDGE", "OUTLIER_MS",
                         "OUTLIER_EDGE"}:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             str(notebook_path), "exec"), namespace)
    function_nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    module = ast.Module(body=function_nodes, type_ignores=[])
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace[function_name]


def marker_lines(ax):
    """Marker-carrying Line2D artists: point rows and appended x-marks."""
    return [line for line in ax.lines
            if line.get_marker() not in (None, "None")]


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
    class_colors = ["#2878b5", "#e69f00", "#009e73", "#d55e00"]
    namespace = {
        "np": np,
        "Line2D": Line2D,
        "CLASS_COLORS": class_colors,
        "CLASS_MARKERS": ["o", "s", "^", "D"],
        "wrong_clf": lambda frame: (
            frame["prediction_clf"].to_numpy()
            != frame["cls"].to_numpy()
        ),
    }
    plot_classifier = load_notebook_function(
        NOTEBOOK, "plot_classifier", namespace
    )
    darken = namespace["darken"]
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
        # row 0: class 0 predicted correctly -> plain class-0 point
        # row 1: class 1 predicted as class 0 -> darkened class-1 point
        points = [line for line in marker_lines(ax)
                  if len(line.get_xdata())]
        assert len(points) == 2
        assert all(line.axes is ax for line in points)
        np.testing.assert_array_equal(points[0].get_xdata(), [-0.5])
        np.testing.assert_array_equal(points[1].get_xdata(), [0.5])
        # class-shaped markers; colors: plain class 0, darkened class 1
        assert [line.get_marker() for line in points] == ["o", "s"]
        assert to_hex(points[0].get_color()) == to_hex(class_colors[0])
        assert to_hex(points[1].get_color()) == to_hex(darken(class_colors[1]))
    finally:
        plt.close(fig)


def test_detector_renderer_uses_caller_axes_for_score_contours_and_points():
    namespace = {
        "np": np,
        "Line2D": Line2D,
        "CLASS_COLORS": ["#2878b5", "#e69f00", "#009e73", "#d55e00"],
        "CLASS_MARKERS": ["o", "s", "^", "D"],
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
        assert all(collection.axes is ax for collection in ax.collections)
        # both rows observed (no appended rows): one point line per class
        # with a non-empty marker position (absent classes yield empty lines)
        points = [line for line in marker_lines(ax) if len(line.get_xdata())]
        assert len(points) == 2
        assert all(line.axes is ax for line in points)
        np.testing.assert_array_equal(points[0].get_xdata(), [-0.5])
        np.testing.assert_array_equal(points[1].get_xdata(), [0.5])
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
        "Line2D": Line2D,
        "CLASS_COLORS": ["#2878b5", "#e69f00", "#009e73", "#d55e00"],
        "CLASS_MARKERS": ["o", "s", "^", "D"],
        "wrong_reg": lambda frame: np.abs(
            frame["prediction_reg"].to_numpy() - frame["y"].to_numpy()
        ) > 0.5,
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
        steps = [
            line for line in ax.lines
            if line.get_marker() in (None, "None")
        ]
        assert len(steps) == 4
        assert all(line.get_linewidth() < 2.0 for line in steps)
        for cls, model_input in enumerate(regressor.inputs):
            np.testing.assert_array_equal(
                model_input[:, 1], np.full(len(x_grid), cls)
            )
            np.testing.assert_allclose(
                steps[cls].get_ydata(),
                model_input[:, 0] + 10 * cls,
            )
        points = marker_lines(ax)
        assert len(points) > 0
        observed = np.concatenate(
            [np.column_stack([line.get_xdata(), line.get_ydata()])
             for line in points],
            axis=0,
        )
        expected = frame[["x", "y"]].to_numpy()
        assert len(observed) == len(expected)
        for row in expected:
            hits = (np.isclose(observed[:, 0], row[0])
                    & np.isclose(observed[:, 1], row[1]))
            assert hits.sum() == 1
    finally:
        plt.close(fig)
