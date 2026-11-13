import numpy as np
import pandas as pd
import pytest

from acise_p2_cidt.tree import tune
from acise_p2_cidt.tree.cart import CART

SMALL_GRID = {
    "max_depth": [3, 8],
    "min_samples_leaf": [5],
    "criterion": ["gini"],
    "class_weight": [None],
}


# GuardedSplit duck-types cv.Split, so the pyright ignores below are on purpose.
class GuardedSplit:
    """
    Split-like object that logs when `X_test` is read, so a test can check that
    tuning never touches the test rows.
    """

    def __init__(self, parts: dict[str, pd.DataFrame | pd.Series], events: list):
        self._parts, self._events = parts, events
        self.X_fit, self.y_fit = parts["X_fit"], parts["y_fit"]
        self.X_val, self.y_val = parts["X_val"], parts["y_val"]
        self.src_fit = self.src_val = self.src_test = None  # single source

    @property
    def X_test(self) -> pd.DataFrame:
        self._events.append("X_test")
        return self._parts["X_test"]


def depth2_parts(seed: int = 0) -> dict[str, pd.DataFrame | pd.Series]:
    """y = (a > 0) & (b > 0) with 10% labels flipped, plus 3 noise columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(1500, 5)), columns=list("abcde"))
    y = ((X.a > 0) & (X.b > 0)).astype(int)
    flip = rng.random(len(y)) < 0.1
    y = pd.Series(np.where(flip, 1 - y, y), name="failed")
    s = slice(0, 900), slice(900, 1200), slice(1200, 1500)
    return {
        "X_fit": X.iloc[s[0]],
        "y_fit": y.iloc[s[0]],
        "X_val": X.iloc[s[1]],
        "y_val": y.iloc[s[1]],
        "X_test": X.iloc[s[2]],
        "y_test": y.iloc[s[2]],
    }


def test_choose_threshold_hand_computed():
    # t=0.2 -> tp 3, fp 1, tn 2, fn 0: MCC = 6 / sqrt(4*3*3*2) = 0.7071
    # t=0.6 -> tp 2, fp 0, tn 3, fn 1: MCC = 6 / sqrt(2*3*3*4) = 0.7071 (tie)
    # t=0.5 -> tp 2, fp 1, tn 2, fn 1: MCC = 3 / 9 = 0.333
    # The tie goes to 0.6, nearer 0.5.
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    assert tune.choose_threshold(proba, y) == 0.6


def test_alpha_prunes_deep_tree_back():
    parts = depth2_parts()
    split = GuardedSplit(parts, [])
    deep = CART(max_depth=8, min_samples_leaf=5).fit(split.X_fit, split.y_fit)
    assert deep.depth_ == 8
    alpha = tune.choose_alpha(deep, split)
    assert alpha > 0 and deep.pruned(alpha).depth_ <= 3


def test_grid_search_and_fit_predict_never_read_test(monkeypatch):
    events: list[str] = []
    fit = CART.fit

    def logged_fit(self, *args, **kwargs):
        events.append("fit")
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(CART, "fit", logged_fit)
    parts = depth2_parts()
    split = GuardedSplit(parts, events)
    best, grid = tune.grid_search(split, SMALL_GRID)  # pyright: ignore[reportArgumentType]
    assert "X_test" not in events and len(grid) == 2
    assert (grid.depth <= 3).all()  # depth 8 is pruned back as well
    assert best.params["max_depth"] == 3  # equal MCC: first grid point wins

    events.clear()
    res = tune.fit_predict(split, best.params)  # pyright: ignore[reportArgumentType]
    assert events == ["fit", "X_test"]  # test rows read once, after the only fit
    assert res.payload["grid"] is None
    assert res.pred.dtype == bool and len(res.pred) == len(parts["X_test"])
    assert np.mean(res.pred == parts["y_test"].to_numpy().astype(bool)) > 0.8


def test_empty_grid_raises():
    split = GuardedSplit(depth2_parts(), [])
    with pytest.raises(ValueError, match="no points"):
        tune.grid_search(split, SMALL_GRID | {"criterion": []})  # pyright: ignore[reportArgumentType]


def _source_split(proba_val, y_val, src_val):
    """Just what the per-source threshold code reads."""

    class S:
        pass

    s = S()
    s.y_val, s.src_val = pd.Series(y_val), np.asarray(src_val)
    s.src_test = np.array(["a", "b", "new"])
    return s


def test_per_source_thresholds():
    # source a fails above 0.3, source b above 0.7; c has too few failures to tune
    rng = np.random.default_rng(0)
    pa, pb, pc = rng.random(200), rng.random(200), rng.random(50)
    proba = np.r_[pa, pb, pc]
    y = np.r_[pa > 0.3, pb > 0.7, np.arange(50) < 3].astype(int)
    src = np.repeat(["a", "b", "c"], [200, 200, 50])
    th = tune.choose_thresholds(proba, y, src, pooled=0.5)
    assert th["a"] < 0.35 and th["b"] > 0.65 and th["c"] == 0.5
    c = tune.Choice({}, 0.0, 0.5, 0.0)
    choice, pred, variants = tune.predict_per_source(
        proba, np.array([0.4, 0.4, 0.4]), _source_split(proba, y, src), c
    )
    # a flags 0.4, b does not, an unseen source falls back to the pooled 0.5
    assert variants["source_thr"].tolist() == [True, False, False]
    assert pred.tolist() == [False, False, False]  # pooled 0.5
    assert choice.thresholds == th


def test_single_source_keeps_pooled_threshold():
    s = _source_split(np.r_[0.1, 0.9], [0, 1], ["a", "a"])
    c = tune.Choice({}, 0.0, 0.5, 0.0)
    choice, pred, variants = tune.predict_per_source(np.r_[0.1, 0.9], np.r_[0.4, 0.6, 0.6], s, c)
    assert variants is None and choice.thresholds is None
    assert pred.tolist() == [False, True, True]


def test_source_rate_flags_the_failing_source():
    class S:
        pass

    s = S()
    s.src_fit = np.repeat(["bad", "good"], 100)
    s.y_fit = pd.Series(np.r_[np.ones(80), np.zeros(20), np.zeros(90), np.ones(10)])
    s.src_val = np.repeat(["bad", "good"], 20)
    s.y_val = pd.Series(np.r_[np.ones(16), np.zeros(4), np.zeros(18), np.ones(2)])
    s.src_test = np.array(["bad", "good", "unseen"])
    assert tune.source_rate(s).tolist() == [True, False, False]
