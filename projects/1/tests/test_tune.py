import numpy as np
import pandas as pd

from acise_p1_cigp.data.load import Dataset
from acise_p1_cigp.solutions.flow_control import model
from acise_p1_cigp.solutions.flow_control.tune import (
    count_statements,
    fit_predict,
    grid,
    tune,
)
from acise_p1_cigp.training.cv import dev_split, folds, prepare, run_cv


def dataset(n=600, seed=0):
    """Label = x >= 50, with 5% label noise; z is noise; some x missing."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 100, n).astype(float)
    y = (x >= 50) ^ (rng.random(n) < 0.05)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    X.loc[::17, "x"] = np.nan
    return Dataset(
        X=X,
        y=pd.Series(y.astype(int), name="failed"),
        dev_mask=np.arange(n) < n // 5,
        meta={"projects": ["p"], "target": "T2"},
        groups=pd.Series("p", index=X.index),
    )


def step(b, t) -> bool:
    return b["x"] >= t["cut"]


def test_grid_is_distinct_quantiles():
    g = grid(pd.Series([0, 0, 0, 0, 1, 1, 2]))
    assert list(g) == [0, 1]  # observed values only; the max 2 is above the 95% quantile


def test_tune_finds_cut():
    ds = dataset()
    t, best = tune(step, {"cut": ("x", 10.0)}, ds.X.fillna(0), ds.y)
    assert abs(t["cut"] - 50) <= 6 and best > 0.7


def test_fit_predict_in_cv():
    ds = dataset()
    res, payloads = run_cv(
        ds, ["x", "z"], lambda s: fit_predict(s, step, {"cut": ("x", None)}), "b"
    )
    assert (res["mcc"] > 0.6).all()
    for p in payloads.values():
        assert p["source"] == "tuned" and abs(p["thresholds"]["cut"] - 50) <= 6


def test_val_pick_never_worse_than_default():
    ds = dataset()
    split = prepare(ds, ["x", "z"], folds(ds)[0])
    for d in (20.0, 50.0, 80.0):
        out = fit_predict(split, step, {"cut": ("x", d)})
        # ignores t, so its val score is the default's
        alone = fit_predict(split, lambda b, t, d=d: b["x"] >= d, {"cut": ("x", d)})
        assert out.payload["val_mcc"] >= alone.payload["val_mcc"]
        assert out.pred.shape == (len(split.X_test),)


def test_count_statements():
    def f(b, t):
        if b["a"] >= 1 and not b["c"]:
            return False
        elif 0 < b["d"] < 3:
            return True
        return b["e"] >= t["x"]

    # if 2; comparisons 1 + 2 + 1; and 1; not 1
    assert count_statements(f) == {"if": 2, "nodes": 8}


def test_candidates_within_budget():
    for predict, th in model.CANDIDATES.values():
        assert count_statements(predict)["nodes"] <= 10
        assert all(isinstance(feat, str) for feat, _ in th.values())


def test_dev_split_has_raw_rows():
    ds = dataset()
    s = dev_split(ds, ["x", "z"])
    assert list(s.raw_fit.columns) == ["x", "z"] and len(s.raw_val) == len(s.X_val)
