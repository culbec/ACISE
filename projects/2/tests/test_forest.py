import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from acise_p2_cidt.tree import forest
from acise_p2_cidt.tree.cart import CART
from acise_p2_cidt.tree.forest import RandomForest

SMALL_GRID = {"max_depth": [3], "min_samples_leaf": [5], "class_weight": [None]}


def synthetic(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(600, 8, n_informative=4, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{j}" for j in range(8)]), y


class LoggedSplit:
    """Split-like object that logs each `X_test` read (duck-types cv.Split)."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, events: list[str]):
        ys = pd.Series(y)
        self.X_fit, self.y_fit = X.iloc[:300], ys.iloc[:300]
        self.X_val, self.y_val = X.iloc[300:450], ys.iloc[300:450]
        self.src_fit = self.src_val = self.src_test = None  # single source
        self._X_test, self._events = X.iloc[450:], events

    @property
    def X_test(self) -> pd.DataFrame:
        self._events.append("X_test")
        return self._X_test


def test_one_tree_without_randomness_is_cart():
    # No bootstrap and all columns per node: the forest is exactly one CART.
    X, y = synthetic()
    rf = RandomForest(
        n_estimators=1, max_features=None, bootstrap=False, max_depth=5
    ).fit(X, y)
    tree = CART(max_depth=5).fit(X, y)
    for a, b in zip(
        (rf.estimators_[0].tree_.feature, rf.estimators_[0].tree_.threshold),
        (tree.tree_.feature, tree.tree_.threshold),
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(rf.predict_proba(X), tree.predict_proba(X))


@pytest.mark.parametrize(
    "max_features, expected",
    [
        ("sqrt", 2),  # int(sqrt(8))
        (0.5, 4),
        (3, 3),
        (20, 8),  # capped at the column count
        (None, 8),
    ],
)
def test_columns_drawn_per_node(max_features, expected):
    assert RandomForest(max_features=max_features)._n_drawn(8) == expected


def test_cart_column_draw_is_seeded_and_changes_the_tree():
    X, y = synthetic()
    full = CART(max_depth=4).fit(X, y)
    assert np.array_equal(
        CART(max_depth=4, max_features=8).fit(X, y).tree_.feature, full.tree_.feature
    )  # drawing all 8 columns is the plain tree
    a = CART(max_depth=4, max_features=1, random_state=1).fit(X, y)
    b = CART(max_depth=4, max_features=1, random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
    assert not np.array_equal(a.tree_.feature[:3], full.tree_.feature[:3])


def test_forest_is_deterministic_and_beats_a_single_tree():
    X, y = synthetic()
    Xf, yf, Xt, yt = X.iloc[:450], y[:450], X.iloc[450:], y[450:]
    p1 = RandomForest(n_estimators=30, random_state=3).fit(Xf, yf).predict_proba(Xt)
    p2 = RandomForest(n_estimators=30, random_state=3).fit(Xf, yf).predict_proba(Xt)
    np.testing.assert_array_equal(p1, p2)
    acc_rf = np.mean((p1[:, 1] > 0.5) == yt)
    acc_tree = np.mean(CART().fit(Xf, yf).predict(Xt) == yt)
    assert acc_rf > acc_tree


def test_fit_predict_reads_test_once_after_tuning():
    X, y = synthetic()
    events: list[str] = []
    split = LoggedSplit(X, y, events)
    best, grid = forest.grid_search(split, SMALL_GRID)  # pyright: ignore[reportArgumentType]
    assert events == [] and len(grid) == 1 and best.alpha == 0.0
    res = forest.fit_predict(split, best.params | {"n_estimators": 20})  # pyright: ignore[reportArgumentType]
    assert events == ["X_test"]
    assert res.pred.dtype == bool and len(res.pred) == 150
    assert res.payload["grid"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"max_features": 0.0},  # empty share
        {"max_features": "log2"},  # not supported
        {"class_weight": "balanced_subsample"},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError, match="invalid"):
        RandomForest(**kwargs)
