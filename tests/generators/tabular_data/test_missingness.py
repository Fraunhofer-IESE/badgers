import numpy as np

from badgers.generators.tabular_data.missingness import MissingValueGenerator


def test_missingness__preserves_shape_and_inserts_nans(tabular_data):
    """All MissingValueGenerator subclasses preserve shape and insert correct NaN count."""
    X, y = tabular_data
    X_np = np.asarray(X)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 1
    total_cells = X_np.shape[0] * n_features
    percentage_missing = 0.1

    for cls in MissingValueGenerator.__subclasses__():
        transformer = cls()
        Xt, _ = transformer.generate(X.copy(), y, percentage_missing=percentage_missing)
        assert Xt.shape[1] == n_features
        assert np.sum(np.isnan(Xt).sum()) == int(percentage_missing * total_cells)
